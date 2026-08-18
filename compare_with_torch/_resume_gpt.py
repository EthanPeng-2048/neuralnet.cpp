# ── _resume_gpt.py — 把 C++ gpt_model.bin 权重加载进 torch 并继续训练 ─────────
#
# 目的：直接回答「gpt_model.bin 的 loss 是不是卡死在 8.5？」。
#   1. 按 C++ V3 序列化格式读入全部 198 个参数矩阵（行优先，无需转置）
#   2. 载入与 C++ 架构一致的 torch GPTModel（model.py）
#   3. 用用户推荐超参（lr=3e-4, AdamW wd=0.01, grad clip=1.0, warmup+cosine）
#      在同一个 token 流（.tokcache）上继续训练，打印 loss 曲线
#
# 若加载正确：初始 loss 应 ≈ 8.5（与用户观察一致），随后应随训练继续下降。
# 若初始 loss 明显偏离 8.5 → 说明序列化映射有误或模型本身异常。
# ─────────────────────────────────────────────────────────────────────────

import argparse
import math
import os
import random
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys_path)
from model import GPTModel, QuickGeLU

PAD_ID = 0

# ── 读取 C++ v4 自描述序列化模型 ──────────────────────────────────────
def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_f32_matrix(f):
    rows = read_u64(f)
    cols = read_u64(f)
    n = rows * cols
    data = struct.unpack(f"<{n}f", f.read(4 * n))
    return rows, cols, list(data)


def read_kv(f):
    """读取一个 KeyValueRecord（长度前缀在文件层已读出）→ dict。"""
    pos = [0]

    def u32():
        v = struct.unpack_from("<I", buf, pos[0])[0]
        pos[0] += 4
        return v

    def u64():
        v = struct.unpack_from("<Q", buf, pos[0])[0]
        pos[0] += 8
        return v

    count = u32()
    out = {}
    for _ in range(count):
        key_len = u32()
        key = buf[pos[0]:pos[0] + key_len].decode()
        pos[0] += key_len
        typ = buf[pos[0]]
        pos[0] += 1
        value_len = u32()
        value = buf[pos[0]:pos[0] + value_len]
        pos[0] += value_len
        if typ == 0:
            out[key] = u64()
        elif typ == 1:
            out[key] = value.decode(errors="replace")
        elif typ == 2:
            out[key] = [u64() for _ in range(len(value) // 8)]
    return out


def load_bin_weights(path: str):
    """返回按 C++ parameters() 顺序排列的 (rows, cols, row-major data) 列表。"""
    global buf
    mats = []
    with open(path, "rb") as f:
        # header: magic u32 | version u32 | precision u8 | spec_len u64 | spec KeyValueRecord
        magic = read_u32(f)
        version = read_u32(f)
        precision = f.read(1)[0]
        assert magic == 0x4E4E4E4E, f"bad magic {magic:#x}"
        assert version == 4, f"unexpected version {version}（仅支持 v4 自描述格式）"
        assert precision == 0, f"expected f32 (precision={precision})"

        spec_len = read_u64(f)
        buf = f.read(spec_len)
        kv = read_kv(f)
        mtype = kv.get("type", 0)
        if mtype == 3 or mtype == 4:  # GPT / ALiBi_GPT
            spec = dict(vocab=kv.get("vocab_size", 0),
                        d_model=kv.get("d_model", 0),
                        seq_len=kv.get("seq_len", 0),
                        num_heads=kv.get("num_heads", 0),
                        d_ff=kv.get("d_ff", 0),
                        num_layers=kv.get("num_layers", 0),
                        pos_enc=kv.get("pos_encoding", 0),
                        activation=kv.get("activation", 0),
                        norm=kv.get("norm_type", 0))
        else:
            raise ValueError(f"unsupported model_type {mtype}")

        # 参数矩阵（直到 tokenizer 数据前）。期望数量（gpt_model.bin 实测 197）：
        #   token_emb(1) + pos_emb(1) + num_layers*16(块=LayerNorm, norm1+norm2 各 2)
        #   + ln_f(1, RMSNorm) + lm_head(2) = 2 + 12*16 + 1 + 2 = 197
        expected = 2 + spec["num_layers"] * 16 + 3
        for _ in range(expected):
            mats.append(read_f32_matrix(f))
    print(f"[load] spec={spec}")
    print(f"[load] {len(mats)} matrices loaded from {path}")
    return spec, mats


# ── 载入 torch 模型 ─────────────────────────────────────────────────────
def apply_weights(model: GPTModel, spec, mats) -> int:
    """按 C++ 参数顺序映射到 model.py。返回校验用的参数量。"""
    i = 0
    n = 0

    def load(w, r, c):
        nonlocal i, n
        data = torch.tensor(mats[i][2], dtype=torch.float32).view(r, c)
        assert tuple(w.shape) == (r, c), f"shape mismatch {tuple(w.shape)} vs {(r, c)}"
        w.data.copy_(data)
        i += 1
        n += r * c

    def load_linear(lin, out, inp):
        nonlocal i, n
        # C++ Linear: w (out,in), b (out,1)
        load(lin.weight, out, inp)
        b = torch.tensor(mats[i][2], dtype=torch.float32).view(out, 1)
        i += 1
        lin.bias.data.copy_(b.view(-1))
        n += out

    def load_layernorm(ln, dim):
        nonlocal i, n
        # C++ LayerNorm: gamma (dim,1), beta (dim,1)
        g = torch.tensor(mats[i][2], dtype=torch.float32).view(dim, 1)
        i += 1
        b = torch.tensor(mats[i][2], dtype=torch.float32).view(dim, 1)
        i += 1
        ln.weight.data.copy_(g.view(-1))
        ln.bias.data.copy_(b.view(-1))
        n += 2 * dim

    # 1. token_emb (vocab, d_model)
    load(model.token_emb.weight, spec["vocab"], spec["d_model"])
    # 2. pos_emb (seq_len, d_model)
    load(model.pos_emb.weight, spec["seq_len"], spec["d_model"])

    # 3. blocks（块内 norm1/norm2 = LayerNorm，各 gamma+beta，与文件 16 个矩阵/块一致）
    for blk in model.blocks:
        # self_attn: w_q, w_k, w_v, w_o (each w+b)
        load_linear(blk.self_attn.w_q, spec["d_model"], spec["d_model"])
        load_linear(blk.self_attn.w_k, spec["d_model"], spec["d_model"])
        load_linear(blk.self_attn.w_v, spec["d_model"], spec["d_model"])
        load_linear(blk.self_attn.w_o, spec["d_model"], spec["d_model"])
        # norm1 (LayerNorm: gamma+beta)
        load_layernorm(blk.norm1, spec["d_model"])
        # ff: fc1 (d_ff, d_model), fc2 (d_model, d_ff)
        load_linear(blk.ff.fc1, spec["d_ff"], spec["d_model"])
        load_linear(blk.ff.fc2, spec["d_model"], spec["d_ff"])
        # norm2 (LayerNorm: gamma+beta)
        load_layernorm(blk.norm2, spec["d_model"])

    # 4. ln_f (RMSNorm: 仅 gamma 一个矩阵)
    g = torch.tensor(mats[i][2], dtype=torch.float32).view(spec["d_model"], 1)
    i += 1
    model.ln_f.weight.data.copy_(g.view(-1))
    n += spec["d_model"]
    # 5. lm_head: (vocab, d_model)
    load_linear(model.lm_head, spec["vocab"], spec["d_model"])

    assert i == len(mats), f"consumed {i}/{len(mats)} matrices"
    return n


# ── 训练数据（复用 _crossval_cache 的逻辑） ─────────────────────────────
def load_tokcache(path: str) -> list[int]:
    with open(path, "rb") as f:
        h = f.read(40)
        magic, ver, ssz, ts, vs, n = struct.unpack("<4sII4xQQQ", h)
        assert magic == b"TKCH", f"bad magic {magic}"
        data = f.read(n * 8)
    return list(struct.unpack(f"<{n}Q", data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt_model.bin")
    ap.add_argument("--cache", default="datasets/tinystories_20k.txt.tokcache")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    spec, mats = load_bin_weights(args.model)
    vocab = spec["vocab"]
    print(f"vocab = {vocab}  random CE = {math.log(vocab):.3f}")

    model = GPTModel(
        vocab_size=vocab,
        d_model=spec["d_model"],
        seq_len=spec["seq_len"],        # 必须与文件 pos_emb 行数一致 (1024)
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        num_layers=spec["num_layers"],
        block_norm_type=0,              # 块内 = LayerNorm（gpt_model.bin 实测）
        final_norm_type=1,              # ln_f = RMSNorm（spec.norm=1，但仅 ln_f 生效）
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loaded = apply_weights(model, spec, mats)
    print(f"model params = {n_params/1e6:.1f}M, loaded raw elems = {loaded:,}")
    assert abs(n_params - loaded) < n_params * 0.01, "param count mismatch!"

    flow = load_tokcache(args.cache)
    print(f"n_tokens = {len(flow):,}")

    stride = args.seq_len
    windows = [(pos, min(args.seq_len, len(flow) - pos))
               for pos in range(0, len(flow), stride)]
    print(f"windows = {len(windows):,}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )

    n_windows = len(windows)
    steps_per_epoch = max(1, n_windows // args.batch_size)
    idx = list(range(n_windows))
    random.shuffle(idx)

    # 第一步：纯评估 checkpoint 当前 loss（不更新）
    model.eval()
    with torch.no_grad():
        x = torch.full((args.batch_size, args.seq_len), PAD_ID, dtype=torch.long, device=device)
        y = torch.full((args.batch_size, args.seq_len), PAD_ID, dtype=torch.long, device=device)
        mask = torch.zeros((args.batch_size, args.seq_len), dtype=torch.float32, device=device)
        for b in range(args.batch_size):
            pos, win_len = windows[idx[b]]
            n = win_len - 1
            seg = flow[pos:pos + win_len]
            x[b, :n] = torch.tensor(seg[:n], dtype=torch.long)
            y[b, :n] = torch.tensor(seg[1:n + 1], dtype=torch.long)
            mask[b, :n] = 1.0
        _, eval_loss = model(x, targets=y, loss_mask=mask)
    print(f"\n[EVAL] gpt_model.bin 当前 loss = {eval_loss.item():.4f}\n", flush=True)
    model.train()

    t0 = time.time()
    for step in range(args.steps):
        if step % steps_per_epoch == 0:
            random.shuffle(idx)
        off = (step % steps_per_epoch) * args.batch_size
        sel = idx[off:off + args.batch_size]
        bs = len(sel)

        x = torch.full((bs, args.seq_len), PAD_ID, dtype=torch.long, device=device)
        y = torch.full((bs, args.seq_len), PAD_ID, dtype=torch.long, device=device)
        mask = torch.zeros((bs, args.seq_len), dtype=torch.float32, device=device)
        for b in range(bs):
            pos, win_len = windows[sel[b]]
            n = win_len - 1
            if n <= 0:
                continue
            seg = flow[pos:pos + win_len]
            x[b, :n] = torch.tensor(seg[:n], dtype=torch.long)
            y[b, :n] = torch.tensor(seg[1:n + 1], dtype=torch.long)
            mask[b, :n] = 1.0

        logits, loss = model(x, targets=y, loss_mask=mask)

        min_lr = args.lr / 10.0
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / max(1, args.warmup_steps)
        else:
            prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, prog)))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        loss.backward()
        if args.max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        opt.step()

        if (step + 1) % 20 == 0 or step == 0:
            dt = time.time() - t0
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  lr={lr:.2e}  time={dt:.0f}s", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
