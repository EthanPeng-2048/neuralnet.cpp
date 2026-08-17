# ── _crossval_cache.py — 用 C++ .tokcache 直接在 torch 里复现 C++ GPT 训练 ──
#
# 目的：判断 gpt_model.bin 训练 loss 卡在 8.5 的问题，到底是 C++ 框架的 bug，
#       还是数据/词表/超参本身的问题。
#
# 方法：直接读取 C++ 训练时使用的 .tokcache（与 C++ 完全相同的 token 流，
#       由 bytebpe.json 分词生成），并按与 C++ text_train.cpp 完全一致的
#       滑动窗口 + padding + loss-mask 逻辑喂给 torch GPT。
#   - 若 torch loss 能平滑降到 8.5 以下 → 数据/词表没问题，问题在 C++ 框架。
#   - 若 torch 也卡在 8.5 附近 → 是数据/词表/超参的问题，与框架无关。
#
# 与 C++ 完全一致的约定：
#   - token_flow = 全部行拼接后的 [BOS]+tokens+[EOS] 流（来自 .tokcache）
#   - window: pos=0, stride, 2*stride, ... 每个窗口 seq_len 长
#   - x[t] = flow[win_pos+t], y[t] = flow[win_pos+t+1]
#   - 窗口末尾不足部分 pad_id=0，mask=0（不参与 loss），其余 mask=1
#   - loss = -Σ log_softmax / num_valid（与 C++ forward_sparse 一致）
#   - AdamW lr=3e-4, wd=0.01(仅2D), grad clip=1.0, warmup+cosine
# ─────────────────────────────────────────────────────────────────────────

import argparse
import math
import os
import random
import struct
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from model import GPTModel

PAD_ID = 0


def load_tokcache(path: str) -> list[int]:
    """读取 C++ TokCacheHeader(40B, MSVC 对齐) + size_t 数组。"""
    with open(path, "rb") as f:
        h = f.read(40)
        magic, ver, ssz, ts, vs, n = struct.unpack("<4sII4xQQQ", h)
        assert magic == b"TKCH", f"bad magic {magic}"
        assert ssz == struct.calcsize("Q") == 8, "non-64bit size_t cache"
        data = f.read(n * 8)
    assert len(data) == n * 8, "truncated cache"
    return list(struct.unpack(f"<{n}Q", data))


def build_windows(flow: list[int], seq_len: int, stride: int):
    """与 C++ 相同：stride 滑动窗口，返回 (offset, valid_len) 列表。"""
    n = len(flow)
    offs = []
    for pos in range(0, n, stride):
        offs.append((pos, min(seq_len, n - pos)))
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="datasets/tinystories.txt.tokcache")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=0)  # 0 = 等于 seq_len
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--vocab", type=int, default=32868)
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
    print(f"vocab = {args.vocab}  random CE = {math.log(args.vocab):.3f}")

    stride = args.stride if args.stride > 0 else args.seq_len
    t0 = time.time()
    print(f"loading cache {args.cache} ...", flush=True)
    flow = load_tokcache(args.cache)
    print(f"  n_tokens = {len(flow):,}  ({time.time()-t0:.0f}s)", flush=True)

    windows = build_windows(flow, args.seq_len, stride)
    print(f"  windows = {len(windows):,}  (n_tokens/stride={len(flow)//stride:,})", flush=True)

    model = GPTModel(
        vocab_size=args.vocab,
        d_model=args.d_model,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params/1e6:.1f}M", flush=True)

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
            n = win_len - 1  # 可用的 (x,y) 对数（与 C++ participate = t+1 < win_len 一致）
            if n <= 0:
                continue
            seg = flow[pos:pos + win_len]
            x[b, :n] = torch.tensor(seg[:n], dtype=torch.long)
            y[b, :n] = torch.tensor(seg[1:n + 1], dtype=torch.long)
            mask[b, :n] = 1.0

        logits, loss = model(x, targets=y, loss_mask=mask)

        # warmup + cosine（与推荐配置一致）
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
