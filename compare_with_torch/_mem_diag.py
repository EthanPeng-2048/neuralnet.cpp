# ── _mem_diag.py — 估算显存占用（理论值），不依赖实际运行 ──
"""诊断 GPT 训练时各层级的显存占用。

基于 gpt_model.bin 的规格 (768/12/12/3072, vocab=32768, seq_len=1024)。
"""

import sys

def mb(n):
    return n / (1024**2)

def fmt(n):
    return f"{mb(n):.1f} MB"

# 从 _diag_spec.py 运行的输出
DM = 768
NH = 12
DK = DM // NH  # 64
NL = 12
DFF = 3072
VOCAB = 32768
SEQ = 1024

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOTAL = BATCH * SEQ  # 每步 token 总数
BH = BATCH * NH

print(f"=== 参数: batch={BATCH}, seq={SEQ}, total_tokens={TOTAL}, d_model={DM}, heads={NH}, layers={NL} ===")
print()

# ── 永久参数 ──
print("── 永久参数 ──")
token_emb = VOCAB * DM * 4
pos_emb = SEQ * DM * 4
print(f"  token_emb ({VOCAB},{DM}): {fmt(token_emb)}")
print(f"  pos_emb ({SEQ},{DM}): {fmt(pos_emb)}")

per_block_params = (
    4 * DM * DM * 4      # w_q, w_k, w_v, w_o: (768,768)*4
    + 2 * DM * 4         # norm1 gamma+beta
    + DFF * DM * 4       # fc1: (3072,768)
    + DM * DFF * 4       # fc2: (768,3072)
    + 2 * DM * 4         # norm2 gamma+beta
)
print(f"  per_block_params: {fmt(per_block_params)}")

ln_f_params = DM * 4  # RMSNorm: gamma only
lm_head = VOCAB * DM * 4 + VOCAB * 4
print(f"  ln_f (RMSNorm): {fmt(ln_f_params)}")
print(f"  lm_head: {fmt(lm_head)}")

total_params = token_emb + pos_emb + NL * per_block_params + ln_f_params + lm_head
print(f"  总参数: {fmt(total_params)}")

# AdamW 动量 (2x)
adam_m = total_params * 2
print(f"  AdamW 动量 (m+v): {fmt(adam_m)}")

# 梯度
grads = total_params
print(f"  梯度: {fmt(grads)}")

permanent = total_params + adam_m + grads
print(f"  永久总占用: {fmt(permanent)}")
print()

# ── 每层前向缓存 ──
print("── 每层前向缓存（存为成员变量，反向传播时需要）──")
q_cache = DM * TOTAL * 4
k_cache = DM * TOTAL * 4
v_cache = DM * TOTAL * 4
print(f"  Q_cache ({DM},{TOTAL}): {fmt(q_cache)}")
print(f"  K_cache ({DM},{TOTAL}): {fmt(k_cache)}")
print(f"  V_cache ({DM},{TOTAL}): {fmt(v_cache)}")

attn_cache = BH * SEQ * SEQ * 4
print(f"  attn_cache ({BH}*{SEQ},{SEQ}) = ({BH*SEQ},{SEQ}): {fmt(attn_cache)}")

# norm1 (LayerNorm): normed + std
norm1_cache = 2 * DM * TOTAL * 4
print(f"  norm1_cache (normed+std): {fmt(norm1_cache)}")

# norm2 (LayerNorm): normed + std
norm2_cache = 2 * DM * TOTAL * 4
print(f"  norm2_cache (normed+std): {fmt(norm2_cache)}")

# ff: gate, sigmoid, up
ff_cache = 3 * DFF * TOTAL * 4
print(f"  ff_cache (gate+sigmoid+up): {fmt(ff_cache)}")

residual1 = DM * TOTAL * 4
residual2 = DM * TOTAL * 4
print(f"  residual1: {fmt(residual1)}")
print(f"  residual2: {fmt(residual2)}")

per_block_cache = (q_cache + k_cache + v_cache + attn_cache +
                   norm1_cache + norm2_cache + ff_cache +
                   residual1 + residual2)
print(f"  每层缓存合计: {fmt(per_block_cache)}")
all_block_cache = NL * per_block_cache
print(f"  {NL} 层缓存合计: {fmt(all_block_cache)}")
print()

# ── 每层前向临时张量（batch 录制期间被 defer，不归还池）──
print("── 每层前向临时张量（batch 录制期间延迟释放）──")
# Linear 输出: w_q, w_k, w_v (每个 forward 产生一个临时输出)
# 注意: Q_cache/K_cache/V_cache 是这些临时输出的 move，不额外占用
# 但在 batch 模式中，Linear 内部也有临时张量...

# 注意力临时张量:
# scores (batched_matmul 输出) → scale_inplace 创建新 scores → apply_mask 创建 masked
# 每个都被 defer 直到 batch 结束
scores_size = BH * SEQ * SEQ * 4
print(f"  scores (batched_matmul): {fmt(scores_size)}")
print(f"  scores (scale_inplace 新): {fmt(scores_size)}")
print(f"  masked (apply_mask): {fmt(scores_size)}")
attn_temps = 3 * scores_size
print(f"  注意力临时合计: {fmt(attn_temps)}")

# concat_out (batched_matmul) → concat (rearrange)
concat_out = DM * TOTAL * 4
concat_rearr = DM * TOTAL * 4
print(f"  concat_out: {fmt(concat_out)}")
print(f"  concat (rearrange): {fmt(concat_rearr)}")

# Linear 内部临时（w*x 的中间结果，被 defer）
# 每个 Linear 内部: w (out,in) * x (in, batch*seq) → 临时张量 (out, batch*seq)
# w_q, w_k, w_v 各有 1 个临时；w_o 有 1 个临时；fc1 有 1 个临时；fc2 有 1 个临时
# 但 w_q/w_k/w_v 的输出被 move 到 Q_cache/K_cache/V_cache，所以临时张量被释放（defer）
linear_temp_q = DM * TOTAL * 4
linear_temp_k = DM * TOTAL * 4
linear_temp_v = DM * TOTAL * 4
linear_temp_o = DM * TOTAL * 4
linear_temp_fc1 = DFF * TOTAL * 4
linear_temp_fc2 = DM * TOTAL * 4
print(f"  Linear 临时 (q,k,v,o): {fmt(linear_temp_q * 4)}")
print(f"  Linear 临时 (fc1): {fmt(linear_temp_fc1)}")
print(f"  Linear 临时 (fc2): {fmt(linear_temp_fc2)}")

# norm 临时: elementwise 操作产生的临时
norm_temps = 2 * DM * TOTAL * 4  # norm1 + norm2 的中间结果
print(f"  norm 临时: {fmt(norm_temps)}")

# residual add 临时: elementwise_binary(Add, ...) 产生的新张量
residual_add_temps = 2 * DM * TOTAL * 4
print(f"  residual add 临时: {fmt(residual_add_temps)}")

per_block_deferred = (attn_temps + concat_out + concat_rearr +
                      linear_temp_q + linear_temp_k + linear_temp_v + linear_temp_o +
                      linear_temp_fc1 + linear_temp_fc2 +
                      norm_temps + residual_add_temps)
print(f"  每层 defer 临时合计: {fmt(per_block_deferred)}")
all_deferred = NL * per_block_deferred
print(f"  {NL} 层 defer 临时合计: {fmt(all_deferred)}")
print()

# ── GPTModel 前向 ──
print("── GPTModel 前向额外占用 ──")
all_emb = TOTAL * DM * 4
stored_tokens = TOTAL * 4
all_T = DM * TOTAL * 4
pos_gathered = TOTAL * DM * 4
pos_T = DM * TOTAL * 4
x_with_pos = DM * TOTAL * 4
stored_x = DM * TOTAL * 4
ln_f_out = DM * TOTAL * 4
logits = VOCAB * TOTAL * 4
print(f"  all_emb ({TOTAL},{DM}): {fmt(all_emb)}")
print(f"  stored_tokens: {fmt(stored_tokens)}")
print(f"  all_T ({DM},{TOTAL}): {fmt(all_T)}")
print(f"  pos_gathered: {fmt(pos_gathered)}")
print(f"  pos_T: {fmt(pos_T)}")
print(f"  x_with_pos: {fmt(x_with_pos)}")
print(f"  stored_x: {fmt(stored_x)}")
print(f"  ln_f_out: {fmt(ln_f_out)}")
print(f"  logits ({VOCAB},{TOTAL}): {fmt(logits)}")

gpt_deferred = all_emb + all_T + pos_gathered + pos_T + x_with_pos
print(f"  GPTModel defer 临时: {fmt(gpt_deferred)}")
print()

# ── 反向传播 ──
print("── 反向传播临时（与 forward 规模相当）──")
bwd_deferred = all_deferred + gpt_deferred
print(f"  反向 defer 临时（估算 = forward）: {fmt(bwd_deferred)}")
print()

# ── 总计 ──
print("=" * 60)
print(f"  永久参数:       {fmt(permanent)}")
print(f"  前向缓存:       {fmt(all_block_cache)}")
print(f"  前向 defer:     {fmt(all_deferred + gpt_deferred)}")
print(f"  反向 defer:     {fmt(bwd_deferred)}")

# 峰值出现时机：反向传播时，前向缓存仍存活，反向 defer 也存活
# 前向 defer 在 flush_batch 后释放
peak = permanent + all_block_cache + bwd_deferred
print(f"  ─────────────────────────────")
print(f"  峰值 (反向时):  {fmt(peak)}")
print(f"  ─────────────────────────────")
print()

# 逐项明细
print("── 峰值组成明细 ──")
print(f"  {'类别':<25} {'大小':>12} {'占比':>8}")
print(f"  {'─'*25} {'─'*12} {'─'*8}")
for name, size in [
    ("模型参数", total_params),
    ("AdamW 动量", adam_m),
    ("梯度", grads),
    ("层缓存 (Q/K/V/attn/norm/ff)", all_block_cache),
    ("反向临时 (≈前向临时)", bwd_deferred),
]:
    print(f"  {name:<25} {fmt(size):>12} {size/peak*100:>7.1f}%")