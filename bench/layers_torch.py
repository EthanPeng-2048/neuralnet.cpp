# ── layers_torch.py — PyTorch CUDA 层基准（对照 neuralnet.cpp layer_bench）────
#
# 层实现与 neuralnet.cpp 的 Layer 一一对应（f32，与 layer_bench 的 Matrix 一致）：
#   linear        Linear(in→out)  输入 (tokens, in) → (tokens, out)
#   layernorm     LayerNorm(d_model)（eps=1e-5）
#   softmax       Softmax（按行）
#   mha           多头注意力（独立 Q/K/V/O 投影，无掩码）
#   causal_attn   因果自注意力（上三角 -inf 掩码）
#   feedforward   Linear(d→ff) → QuickGeLU → Linear(ff→d)
#   gpt_block     Pre-Norm: x+Attn(LN1(x)); x+FFN(LN2(x))
#   transformer   Post-Norm: LN(x+Attn(x)); LN(x+FFN(x))（编码器层）
#
# 计时：CUDA event，fwd 与 fwd+bwd 分别测，warmup≥20、best-of-iter（与
# layer_bench 口径一致；layer_bench 是 wall-clock 含提交开销，见报告口径说明）。
# 形状映射：layer_bench 输入 (feature_dim, batch*seq) 列主序 ↔ torch
# (batch, seq, feature) 行主序，数学等价、FLOPs 相同。
#
# 用法: python layers_torch.py --group model|large [--rounds 3] [--warmup 20]
#        [--iter 50] [--out results.json]
#   model 组: d_model=64  heads=4  d_ff=256  seq=256 batch=64   （被测模型配置）
#   large 组: d_model=512 heads=8  d_ff=2048 seq=1024 batch=8   （大尺寸）
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compare_with_torch"))
from model import QuickGeLU  # noqa: E402  （与 C++ GeLU 一致）


# ═══════════════════════════════════════════════════════════════════════════
#  各层实现（f32）
# ═══════════════════════════════════════════════════════════════════════════
class LinearLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.fc = nn.Linear(din, dout)

    def forward(self, x):        # x: (T, din)
        return self.fc(x)


class SoftmaxLayer(nn.Module):
    def forward(self, x):        # x: (rows, cols) 按行 softmax
        return F.softmax(x, dim=-1)


class MultiHeadAttention(nn.Module):
    """独立 Q/K/V/O 投影 + 缩放点积（无掩码）—— 对应 C++ MultiHeadAttention。"""

    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.scale = self.d_k ** -0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):        # x: (B, T, C)
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)


class CausalSelfAttention(nn.Module):
    """因果自注意力 —— 对应 C++ CausalSelfAttention（复用 model.py 结构）。"""

    def __init__(self, d_model, heads, max_len):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.scale = self.d_k ** -0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = att + self.mask[:T, :T]
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)


class FeedForwardLayer(nn.Module):
    """Linear → QuickGeLU → Linear —— 对应 C++ FeedForward。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = QuickGeLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class GPTBlockPreNorm(nn.Module):
    """x = x + Attn(LN1(x)); x = x + FFN(LN2(x)) —— 对应 C++ GPTBlock。"""

    def __init__(self, d_model, heads, d_ff, seq):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, heads, seq)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForwardLayer(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerEncoderPostNorm(nn.Module):
    """LN(x + Attn(x)); LN(x + FFN(x)) —— 对应 C++ TransformerEncoderLayer。"""

    def __init__(self, d_model, heads, d_ff, seq):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  基准执行
# ═══════════════════════════════════════════════════════════════════════════
def bench_layer(name, build, in_shape, fwd_flops, tokens, warmup, iters, rounds):
    """build() → (module, input)。fwd 与 fwd+bwd 分别计时（best-of）。"""
    fwd_ms_rounds, train_ms_rounds = [], []
    for _ in range(rounds):
        torch.cuda.synchronize()
        mod, x = build()

        def fwd():
            return mod(x)

        # warmup + fwd 计时
        for _i in range(warmup):
            fwd()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        best_f = float("inf")
        for _i in range(iters):
            s.record(); fwd(); e.record(); torch.cuda.synchronize()
            best_f = min(best_f, s.elapsed_time(e))

        # fwd+bwd 计时（zero_grad(set_to_none) + forward + backward）
        def train_step():
            mod.zero_grad(set_to_none=True)
            y = mod(x)
            if y.grad_fn is None:
                y = y.detach().requires_grad_(True)
            y.backward(torch.ones_like(y))

        for _i in range(warmup):
            train_step()
        torch.cuda.synchronize()
        best_t = float("inf")
        for _i in range(iters):
            s.record(); train_step(); e.record(); torch.cuda.synchronize()
            best_t = min(best_t, s.elapsed_time(e))
        fwd_ms_rounds.append(best_f)
        train_ms_rounds.append(best_t)
        del mod, x
        torch.cuda.empty_cache()

    bf, bt = min(fwd_ms_rounds), min(train_ms_rounds)
    fwd_gf = fwd_flops / (bf / 1e3) / 1e9
    tok_s = tokens / (bt / 1e3)
    print(f"  {name:<16}: fwd {bf:8.3f} ms  {fwd_gf:9.1f} GFLOPS"
          f"  |  train {bt:8.3f} ms  {tok_s:9.0f} tok/s"
          f"   rounds fwd={'/'.join(f'{x:.2f}' for x in fwd_ms_rounds)}"
          f" train={'/'.join(f'{x:.2f}' for x in train_ms_rounds)}")
    return {"layer": name, "fwd_ms": bf, "train_ms": bt,
            "fwd_rounds_ms": fwd_ms_rounds, "train_rounds_ms": train_ms_rounds,
            "fwd_gflops": fwd_gf, "tok_per_s": tok_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["model", "large"], required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)

    if args.group == "model":
        d, heads, dff, seq, batch = 64, 4, 256, 256, 64
        T = batch * seq
    else:
        d, heads, dff, seq, batch = 512, 8, 2048, 1024, 8
        T = batch * seq

    print(f"=== layers_torch ({dev}, fp32, group={args.group}: "
          f"d={d} heads={heads} dff={dff} seq={seq} batch={batch}) ===")

    g = torch.Generator(device="cuda").manual_seed(1234)

    def rand(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    # FLOPs 公式与 layer_bench 一致
    flops_linear = 2.0 * 64 * 256 * T if args.group == "model" else 2.0 * d * dff * T
    flops_norm = 6.0 * d * T
    flops_softmax = 6.0 * d * T
    flops_attn = 6.0 * d * d * T + 4.0 * T * T * d
    flops_ff = 4.0 * d * dff * T
    flops_block = flops_attn + flops_ff + flops_norm * 2.0

    cases = []

    # linear: layer_bench shape (in, batch) — 用 tokens=T 规模（in=64/512→d 档）
    def build_linear():
        mod = LinearLayer(d, dff).cuda()
        x = rand(T, d)
        return mod, x
    cases.append(("linear", build_linear, (T, d), 2.0 * d * dff * T, T))

    def build_layernorm():
        mod = nn.LayerNorm(d, eps=1e-5).cuda()
        # nn 侧输入 (dmodel, batch*seq) 列主序、按列归一 → torch 等价 (T, d)
        x = rand(T, d)
        return mod, x
    cases.append(("layernorm", build_layernorm, (d, T), flops_norm, T))

    def build_softmax():
        mod = SoftmaxLayer().cuda()
        x = rand(d, T)
        return mod, x
    cases.append(("softmax", build_softmax, (d, T), flops_softmax, T))


    def build_mha():
        mod = MultiHeadAttention(d, heads).cuda()
        x = rand(batch, seq, d)
        return mod, x
    cases.append(("mha", build_mha, (batch, seq, d), flops_attn, T))

    def build_causal():
        mod = CausalSelfAttention(d, heads, seq).cuda()
        x = rand(batch, seq, d)
        return mod, x
    cases.append(("causal_attn", build_causal, (batch, seq, d), flops_attn, T))

    def build_ff():
        mod = FeedForwardLayer(d, dff).cuda()
        x = rand(batch, seq, d)
        return mod, x
    cases.append(("feedforward", build_ff, (batch, seq, d), flops_ff, T))

    def build_gpt_block():
        mod = GPTBlockPreNorm(d, heads, dff, seq).cuda()
        x = rand(batch, seq, d)
        return mod, x
    cases.append(("gpt_block", build_gpt_block, (batch, seq, d), flops_block, T))

    def build_transformer():
        mod = TransformerEncoderPostNorm(d, heads, dff, seq).cuda()
        x = rand(batch, seq, d)
        return mod, x
    cases.append(("transformer", build_transformer, (batch, seq, d), flops_block, T))

    out_all = []
    for name, build, shape, flops, tok in cases:
        r = bench_layer(name, build, shape, flops, tok,
                        args.warmup, args.iter, args.rounds)
        out_all.append(r)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"device": dev, "dtype": "fp32", "group": args.group,
                       "config": {"d_model": d, "heads": heads, "d_ff": dff,
                                  "seq": seq, "batch": batch},
                       "layers": out_all}, f, indent=2)
        print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
