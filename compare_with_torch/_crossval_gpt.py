# ── _crossval_gpt.py — torch 交叉验证：判断问题在框架还是数据/词表 ────────────
#
# 目标：用 PyTorch（可信参考实现）在「真实 CharBPE-32862 词表 + 真实 TinyStories
# 数据切片」上训练一个小型 GPT，观察 loss 是否能正常下降。
#   - 若 torch loss 能稳定降到 8.5 以下且持续下降 → 数据/词表/超参没问题，
#     问题出在 C++ 训练框架（或该框架在大配置下的行为）。
#   - 若 torch 也在 8.5 附近停滞 → 不是框架问题，是数据/词表/超参的问题。
#
# 与 C++ 保持一致的约定：
#   - 每行 = 独立样本: [BOS] + tokens + [EOS]
#   - pad_id=0, 目标 = 输入左移一位
#   - loss mask: 只在有效 token 位置计算 loss（与 C++ forward_sparse 一致）
#   - AdamW lr=3e-4, weight_decay=0.01, grad clip=1.0（用户推荐参数）
# ─────────────────────────────────────────────────────────────────────────

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import load_charbpe_from_file
from model import GPTModel

PAD_ID = 0


def load_samples(text_path: str, tokenizer, n_lines: int):
    """读取前 n_lines 行，每行编码为 [BOS]+tokens+[EOS] 样本。"""
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    samples = []
    total_tokens = 0
    with open(text_path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            toks = tokenizer.encode(line)
            s = []
            if bos_id >= 0:
                s.append(bos_id)
            s.extend(toks)
            if eos_id >= 0:
                s.append(eos_id)
            total_tokens += len(s)
            samples.append(s)
    print(f"samples={len(samples)} total_tokens={total_tokens} avg_len={total_tokens / max(1, len(samples)):.1f}")
    return samples


def build_batch(samples, indices, batch_size, seq_len, device):
    """与 C++ 一致的 next-token 批次构建（不足部分 pad，mask 掉 pad 位置）。"""
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    for b in range(batch_size):
        s = samples[indices[b]]
        n = min(len(s) - 1, seq_len)  # 最多 seq_len 个 (x,y) 对
        if n <= 0:
            continue
        x[b, :n] = torch.tensor(s[:n], dtype=torch.long)
        y[b, :n] = torch.tensor(s[1:n + 1], dtype=torch.long)
        mask[b, :n] = 1.0
    return x, y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="datasets/tinystories.txt")
    ap.add_argument("--vocab", default="bytebpe.json")
    ap.add_argument("--n-lines", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    tok = load_charbpe_from_file(args.vocab)
    assert tok is not None, f"cannot load {args.vocab}"
    vocab = tok.vocab_size()
    print(f"tokenizer vocab_size={vocab} (C++ model 使用 vocab={vocab + 6} 的同类词表)")

    samples = load_samples(args.text, tok, args.n_lines)
    assert len(samples) >= args.batch_size, "too few samples"

    model = GPTModel(
        vocab_size=vocab,
        d_model=args.d_model,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params/1e6:.2f}M, random-init CE≈{vocab:g} -> {__import__('math').log(vocab):.3f}")

    # 只对需要 weight decay 的参数（2D 权重）应用，其余 0（标准 AdamW 约定）
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )

    indices = list(range(len(samples)))
    random.shuffle(indices)
    steps_per_epoch = len(indices) // args.batch_size
    print(f"steps_per_epoch={steps_per_epoch}")

    t0 = time.time()
    for step in range(args.steps):
        # 每个 epoch 内顺序取 batch；越界时重新 shuffle
        if step % steps_per_epoch == 0:
            random.shuffle(indices)
        off = (step % steps_per_epoch) * args.batch_size
        sel = indices[off:off + args.batch_size]

        x, y, mask = build_batch(samples, sel, args.batch_size, args.seq_len, device)
        logits, loss = model(x, targets=y, loss_mask=mask)

        # warmup + cosine 余弦衰减（与推荐配置一致）
        min_lr = args.min_lr if args.min_lr > 0 else args.lr / 10.0
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

        if (step + 1) % 10 == 0 or step == 0:
            dt = time.time() - t0
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  time={dt:.0f}s", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
