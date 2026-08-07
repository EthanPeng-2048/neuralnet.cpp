# ── text_train.py — GPT 文本生成训练 (PyTorch CUDA) ────────────────────────
#
# 复制自 C++ 项目 src/text_train.cpp。
# 数据流：
#   文本 → Tokenizer.encode → token IDs
#   每 batch：token IDs → (batch, seq) LongTensor → GPTModel.forward
#     → logits (batch, seq, vocab) → CrossEntropyLoss → scalar
#     → backward → optimizer.step / zero_grad
#
# 关键约定（与 C++ 一致）：
#   - 每行 = 一个独立训练样本，[BOS] + tokens + [EOS]
#   - 短行用 pad_id=0 填充，目标 = 输入左移一位（next-token prediction）
#   - 不忽略 PAD 位置的 loss（与 C++ CrossEntropyLoss 行为一致）
#   - steps_per_epoch = min(num_samples / batch_size, 1000)
# ─────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

# 确保能 import 同目录下的模块（无论从哪个 cwd 运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import TokenizerAdapter, get_or_train_tokenizer, load_tokenizer_from_file
from model import GPTModel


# ── 默认超参数（与 C++ domain_gpt.hpp 一致） ──────────────────────────────

GPT_VOCAB_SIZE = 8000
GPT_D_MODEL = 96
GPT_NUM_HEADS = 4
GPT_D_FF = 192
GPT_NUM_LAYERS = 2
GPT_SEQ_LEN = 32


def parse_args():
    parser = argparse.ArgumentParser(description="GPT 文本生成训练程序 (PyTorch CUDA)")
    parser.add_argument("text_path", help="训练文本文件路径")
    parser.add_argument("--save", default="gpt_model_torch.pt", help="模型保存路径")
    parser.add_argument("--resume", default=None, help="从已有模型恢复训练")
    parser.add_argument("--vocab", default="gpt_bpe_torch.json", help="词表 JSON 路径（tokenizers 库标准格式，不存在则现场训练）")
    parser.add_argument("--vocab-size", type=int, default=GPT_VOCAB_SIZE, help="训练 BPE 时的目标词表大小")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=GPT_SEQ_LEN)
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["sgd", "sgd_momentum", "adam", "adamw"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减 (AdamW)")
    parser.add_argument("--d-model", type=int, default=GPT_D_MODEL)
    parser.add_argument("--num-heads", type=int, default=GPT_NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=GPT_NUM_LAYERS)
    parser.add_argument("--d-ff", type=int, default=GPT_D_FF)
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu / cuda:0（默认 cuda，不可用自动回退 cpu）",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=0.0,
        help="梯度裁剪最大全局 L2 范数 (默认 0=不裁剪，与 C++ --max-norm 一致)",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_samples(text: str, tokenizer: TokenizerAdapter) -> list[list[int]]:
    """每行 = 一个训练样本: [BOS] + tokens + [EOS]（与 C++ 一致）。"""
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    samples: list[list[int]] = []
    total_tokens = 0
    for line in text.splitlines():
        # 去除行首尾空白
        line = line.strip()
        if not line:
            continue

        tokens = tokenizer.encode(line)
        sample: list[int] = []
        if bos_id != TokenizerAdapter.npos:
            sample.append(bos_id)
        sample.extend(tokens)
        if eos_id != TokenizerAdapter.npos:
            sample.append(eos_id)
        total_tokens += len(sample)
        samples.append(sample)

    print(f"文本行数: {len(samples)}, {total_tokens} tokens (含 BOS/EOS)")
    return samples


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float = 0.0):
    """创建优化器（与 C++ create_optimizer 对应）。"""
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    if name == "sgd_momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "adam":
        # C++ Adam 默认: beta1=0.9, beta2=0.999, eps=1e-8（与 PyTorch 默认一致）
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "adamw":
        # AdamW: 解耦权重衰减（与 C++ AdamW 一致）
        return optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=weight_decay,
        )
    raise ValueError(f"未知优化器: {name}")


def build_batch(
    samples: list[list[int]],
    indices: list[int],
    batch_size: int,
    seq_len: int,
    pad_id: int,
    device: torch.device,
):
    """按给定索引取一个 batch 并构造 (x, y) 张量。

    改造点（相对 C++ 版）：
      - 不再每 step 独立随机采样，而是由调用方传入本 step 的样本索引列表
        （通常来自 epoch 级别的 shuffle 队列），保证每个样本每 epoch 只被访问一次
      - x[t] = sample[t], y[t] = sample[t+1]（next-token prediction）
      - 样本长度 > seq_len+1: 截断（取前 seq_len+1 个 token）
      - 样本长度 <= seq_len:  不足部分用 pad_id 填充
    """
    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)

    for b in range(batch_size):
        sample = samples[indices[b]]
        sample_len = len(sample)

        for t in range(seq_len):
            x[b, t] = sample[t] if t < sample_len else pad_id
            y[b, t] = sample[t + 1] if (t + 1) < sample_len else pad_id

    return x, y


def save_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: optim.Optimizer,
    epoch: int,
    tokenizer_json: str,
):
    """保存 checkpoint（含模型规格 + 嵌入 tokenizer JSON，与 C++ V3 格式精神一致）。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "vocab_size": model.vocab_size,
                "d_model": model.d_model,
                "seq_len": model.seq_len,
                "num_heads": model.num_heads,
                "d_ff": model.d_ff,
                "num_layers": model.num_layers,
            },
            "tokenizer_json": tokenizer_json,
        },
        path,
    )


def main():
    cfg = parse_args()

    # ── 加载文本 ─────────────────────────────────────────────
    print(f"加载文本: {cfg.text_path} ...")
    text = load_text(cfg.text_path)
    if not text:
        print("文本文件为空", file=sys.stderr)
        sys.exit(1)

    # ── 加载或训练分词器（tokenizers 库标准格式） ─────────────
    tokenizer = get_or_train_tokenizer(cfg.vocab, cfg.text_path, cfg.vocab_size)

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = 0  # 与 C++ 一致：所有分词器的 PAD_ID 都是 0

    # ── 按行编码 ─────────────────────────────────────────────
    samples = prepare_samples(text, tokenizer)
    print(f"词表大小: {tokenizer.vocab_size()}\n")

    # 过滤长度 < 2 的样本
    valid_samples = [s for s in samples if len(s) >= 2]
    if not valid_samples:
        print("无长度 >= 2 的训练样本", file=sys.stderr)
        sys.exit(1)
    print(f"有效样本数: {len(valid_samples)} (长度 >= 2)\n")

    # ── 设备选择 ─────────────────────────────────────────────
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU\n")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    # ── 打印配置 ─────────────────────────────────────────────
    print("========================================")
    print("  GPT 文本生成训练 (PyTorch CUDA)")
    print("========================================")
    print(f"  词表大小: {tokenizer.vocab_size()}")
    print(f"  模型维度: {cfg.d_model}")
    print(f"  注意力头: {cfg.num_heads}")
    print(f"  Transformer 层数: {cfg.num_layers}")
    print(f"  FFN 维度: {cfg.d_ff}")
    print(f"  序列长度: {cfg.seq_len}")
    print(f"  优化器: {cfg.optimizer}  学习率: {cfg.lr}  权重衰减: {cfg.weight_decay}")
    print(f"  梯度裁剪 max_norm: {cfg.max_norm}  (0=不裁剪)")
    print(f"  Loss mask: 启用 (屏蔽 padding 位置，与 C++ forward_sparse 一致)")
    print(f"  轮数: {cfg.epochs}  批大小: {cfg.batch_size}")
    print(f"  设备: {device}")
    print("========================================\n")

    # ── 构建模型 ─────────────────────────────────────────────
    model = GPTModel(
        vocab_size=tokenizer.vocab_size(),
        d_model=cfg.d_model,
        seq_len=cfg.seq_len,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
    ).to(device)

    # ── 优化器 ───────────────────────────────────────────────
    optimizer = build_optimizer(cfg.optimizer, model, cfg.lr, cfg.weight_decay)

    # ── 恢复训练 ─────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume:
        if not os.path.exists(cfg.resume):
            print(f"模型文件不存在: {cfg.resume}，将从头训练\n")
        else:
            ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"已加载模型: {cfg.resume} (从 epoch {start_epoch} 继续)\n")

    # ── 训练循环 ─────────────────────────────────────────────
    # 改造：每个样本每 epoch 被访问一次（按 shuffle 后的顺序切片），
    # 不再每 step 独立随机采样。
    import random as _random

    n_samples = len(valid_samples)
    # 一个 epoch 的 step 数 = 全部样本 / batch_size（向下取整，丢弃末尾不足一个 batch 的样本）
    # 与 C++ 一致：无上限
    steps_per_epoch = n_samples // cfg.batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1

    rng_epoch = _random.Random(42)  # epoch 级别 shuffle 的随机源

    model.train()
    t_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        ep_start = time.time()
        total_loss = 0.0

        # 每个 epoch 开始前 shuffle 样本索引队列
        indices = list(range(n_samples))
        rng_epoch.shuffle(indices)

        for step in range(steps_per_epoch):
            # 取本 step 的 batch_size 个索引（顺序切片，不再独立采样）
            batch_indices = indices[step * cfg.batch_size : (step + 1) * cfg.batch_size]

            x, y = build_batch(
                valid_samples,
                batch_indices,
                cfg.batch_size,
                cfg.seq_len,
                pad_id,
                device,
            )

            # 构建 loss_mask（与 C++ forward_sparse 一致）
            # 非对话模式：屏蔽 padding 位置，只对真实 token 计算 loss
            # mask[t][b] = 1.0 当 t+1 < sample_len（即目标不是 padding）
            loss_mask = torch.zeros(
                cfg.batch_size, cfg.seq_len, device=device
            )
            for b in range(cfg.batch_size):
                sample = valid_samples[batch_indices[b]]
                sample_len = len(sample)
                for t in range(cfg.seq_len):
                    if t + 1 < sample_len:
                        loss_mask[b, t] = 1.0

            logits, loss = model(x, targets=y, loss_mask=loss_mask)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（与 C++ --max-norm 一致，在 step 之前）
            if cfg.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_norm
                )

            optimizer.step()

            total_loss += loss.item()

            # 进度显示
            if (step + 1) % cfg.log_interval == 0 or step + 1 == steps_per_epoch:
                print(
                    f"\r  Epoch {epoch + 1}/{cfg.epochs}"
                    f"  step {step + 1}/{steps_per_epoch}"
                    f"  loss: {loss.item():.4f}   ",
                    end="",
                    flush=True,
                )

        ep_sec = time.time() - ep_start
        avg_loss = total_loss / steps_per_epoch
        print(
            f"\r  Epoch {epoch + 1}/{cfg.epochs}"
            f"  avg_loss={avg_loss:.4f}"
            f"  time={ep_sec:.1f}s"
        )

    total_sec = time.time() - t_start

    # ── 读取 tokenizer JSON 以便嵌入模型 ─────────────────────
    tokenizer_json = ""
    if os.path.exists(cfg.vocab):
        with open(cfg.vocab, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()

    # ── 保存模型（含规格 + 嵌入 tokenizer） ──────────────────
    save_checkpoint(cfg.save, model, optimizer, cfg.epochs - 1, tokenizer_json)
    print(
        f"\n训练完成! 总耗时: {total_sec:.1f}s"
        f"  词表已嵌入模型文件"
    )


if __name__ == "__main__":
    main()
