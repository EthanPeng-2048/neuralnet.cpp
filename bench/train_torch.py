# ── train_torch.py — PyTorch CUDA 训练速度/显存基准 ──────────────────────────
#
# 对照 neuralnet.cpp 的 text_train.exe（GPT、稀疏 CE、Adam、滑动窗口数据管线）。
# 数据管线逐项镜像 C++ 侧（src/text_train.cpp）：
#   - 每行独立 tokenize 后纯拼接（无 BOS/EOS），doc 边界在 C++ 侧由 doc_ids 掩码，
#     torch 侧为标准 causal attention（FLOPs 相同，掩码语义差异见报告）
#   - 窗口: pos = 0, seq, 2*seq, ...；x[t]=flow[pos+t]
#   - 目标: y[t]=flow[pos+t+1]，仅当 t+1 < win_len=min(seq, len-pos) 参与 loss
#     （注意：满窗的最后一个位置 t=seq-1 被屏蔽——与 C++ 逐位一致）
#   - 每 epoch shuffle 窗口索引；steps_per_epoch = ceil(windows/batch)
# 计时口径与 C++ 对齐：epoch 边界同步 GPU（C++ 在 epoch 末 drain loss 回读），
# epoch 内 host（构 batch + H2D）与 GPU 重叠，不做每步 synchronize。
# 精度：--dtype fp16 = 全 f16（参数/激活/掩码/CE 全 fp16，无 master weights），
# 对应 C++ --precision-param/compute/stable/optimizer 全 f16。
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compare_with_torch"))

import numpy as np
import torch

from tokenizer import get_or_train_tokenizer
from model import GPTModel


def parse_args():
    p = argparse.ArgumentParser(description="GPT 训练速度/显存基准 (PyTorch CUDA)")
    p.add_argument("text_path")
    p.add_argument("--vocab", default="bench_vocab_torch.json")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument(
        "--adam-eps", type=float, default=1e-8,
        help="Adam eps。fp16 下 1e-8 表示为 0 → 稀疏 embedding 零梯度行 0/0=NaN，"
             "须用 fp16 可表示值（建议 1e-4）；C++ 侧 eps 以 fp32 RParam 进 kernel，"
             "保持 1e-8",
    )
    p.add_argument("--out", default=None, help="结果 JSON 输出路径")
    return p.parse_args()


def main():
    cfg = parse_args()
    assert torch.cuda.is_available(), "CUDA 不可用"
    device = torch.device("cuda")

    # ── 读文本 + tokenize（不计时） ─────────────────────────────
    with open(cfg.text_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    lines = [ln for ln in lines if ln]

    t0 = time.perf_counter()
    tok = get_or_train_tokenizer(cfg.vocab, cfg.text_path, cfg.vocab_size)
    flow_list: list[int] = []
    for ln in lines:
        flow_list.extend(tok.encode(ln))
    flow = np.asarray(flow_list, dtype=np.int64)
    print(f"tokenize+concat: {len(lines)} 行, {flow.size} tokens, {time.perf_counter()-t0:.1f}s")

    L = int(flow.size)
    seq, bs_cfg = cfg.seq_len, cfg.batch_size
    windows = np.arange(0, L, seq, dtype=np.int64)
    steps_per_epoch = max(1, math.ceil(len(windows) / bs_cfg))
    vocab = tok.vocab_size()
    print(f"词表: {vocab}  窗口数: {len(windows)}  steps/epoch: {steps_per_epoch}")

    # ── 模型 + 优化器 ──────────────────────────────────────────
    model = GPTModel(
        vocab_size=vocab, d_model=cfg.d_model, seq_len=cfg.seq_len,
        num_heads=cfg.num_heads, d_ff=cfg.d_ff, num_layers=cfg.num_layers,
    ).to(device)
    if cfg.dtype == "fp16":
        model = model.half()  # 参数 + 浮点 buffer（causal mask）全部 fp16
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=cfg.adam_eps
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    print("========================================")
    print("  GPT 训练基准 (PyTorch CUDA)")
    print(f"  d_model={cfg.d_model} heads={cfg.num_heads} layers={cfg.num_layers} "
          f"d_ff={cfg.d_ff} seq={seq} batch={bs_cfg} vocab={vocab}")
    print(f"  dtype={cfg.dtype}  optimizer=adam lr={cfg.lr}  params={n_params}")
    print(f"  device={torch.cuda.get_device_name(device)}")
    print("========================================")

    rng = random.Random(42)
    idx_all = np.asarray(list(range(len(windows))), dtype=np.int64)
    ar_t = np.arange(seq, dtype=np.int64)
    results = []

    for epoch in range(cfg.epochs):
        # 每 epoch 重新洗牌（与 C++ std::shuffle 语义无需逐位一致，等分布即可）
        epoch_order = idx_all.tolist()
        rng.shuffle(epoch_order)
        idx = np.asarray(epoch_order, dtype=np.int64)
        # weighted loss 累加（设备端标量，避免每步同步）
        loss_weighted = torch.zeros((), device=device, dtype=torch.float32)
        valid_total = 0
        tokens_total = 0

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        for step in range(steps_per_epoch):
            sel = idx[step * bs_cfg:(step + 1) * bs_cfg]
            bs = len(sel)
            starts = windows[sel]
            pos = starts[:, None] + ar_t[None, :]           # (bs, seq)
            in_x = pos < L
            x = np.where(in_x, flow[np.minimum(pos, L - 1)], 0)
            # C++: participate = t+1 < win_len=min(seq, len-pos)
            valid = (ar_t[None, :] + 1 < seq) & (pos + 1 < L)
            y = np.where(valid, flow[np.minimum(pos + 1, L - 1)], 0)

            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y).to(device)
            mb = torch.from_numpy(valid.astype(np.float32)).to(device)

            logits, loss = model(xb, targets=yb, loss_mask=mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n_valid = int(valid.sum())
            loss_weighted = loss_weighted + loss.detach().float() * n_valid
            valid_total += n_valid
            tokens_total += bs * seq

        torch.cuda.synchronize()
        sec = time.perf_counter() - t_start
        avg_loss = (loss_weighted.item() / valid_total) if valid_total else float("nan")
        tps = tokens_total / sec
        results.append({
            "epoch": epoch + 1, "sec": sec, "steps": steps_per_epoch,
            "tokens": tokens_total, "tok_per_s": tps, "avg_loss": avg_loss,
        })
        print(f"  Epoch {epoch + 1}/{cfg.epochs}  avg_loss={avg_loss:.4f}  "
              f"time={sec:.1f}s  tok/s={tps:,.0f}")

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(f"\n显存: peak_allocated={peak_alloc / 1e6:.1f} MB  "
          f"peak_reserved={peak_reserved / 1e6:.1f} MB")

    steady = results[1:] if len(results) > 1 else results
    mean_tps = sum(r["tok_per_s"] for r in steady) / len(steady)
    mean_sec = sum(r["sec"] for r in steady) / len(steady)
    print(f"稳定态(除首 epoch) 平均: {mean_sec:.2f}s/epoch  {mean_tps:,.0f} tok/s")

    out = {
        "config": {**vars(cfg), "vocab_actual": vocab, "params": n_params,
                   "steps_per_epoch": steps_per_epoch, "windows": int(len(windows))},
        "device": torch.cuda.get_device_name(device),
        "results": results,
        "peak_alloc_mb": peak_alloc / 1e6,
        "peak_reserved_mb": peak_reserved / 1e6,
        "steady_mean_tok_per_s": mean_tps,
        "steady_mean_sec_per_epoch": mean_sec,
    }
    if cfg.out:
        os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
        with open(cfg.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入 {cfg.out}")


if __name__ == "__main__":
    main()
