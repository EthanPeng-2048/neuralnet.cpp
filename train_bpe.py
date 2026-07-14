#!/usr/bin/env python3
"""
字节级 BPE 分词器训练脚本（纯 Python 标准库，零依赖）
优化版：合并与统计同步进行，速度提升约一倍。
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ── 预分词正则（GPT-2 风格）──────────────────────────────
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+""",
    re.IGNORECASE,
)


def pre_tokenize(text: str) -> list[str]:
    """按正则将文本切成块"""
    return PAT.findall(text)


def text_to_bytes_chunks(text: str) -> list[list[int]]:
    """预分词后，每块转为 UTF-8 字节序列"""
    chunks = []
    for piece in pre_tokenize(text):
        chunks.append(list(piece.encode("utf-8")))
    return chunks


def train_bpe(text: str, vocab_size: int, verbose: int = 1):
    """
    训练字节级 BPE（优化版：合并与统计一次完成）。

    返回:
      vocab: {id: bytes}  — 词表（id → 字节序列）
      merges: [(id_a, id_b, new_id), ...]  — 合并规则
    """
    # 1. 预分词 + 转字节
    chunks = text_to_bytes_chunks(text)
    if verbose >= 1:
        print(f"预分词后 {len(chunks)} 个块")

    # 2. 初始词表：256 个字节，映射到 ID 4~259
    vocab = {i + 4: bytes([i]) for i in range(256)}
    merges = []

    next_id = 260
    num_merges = vocab_size - 4 - 256
    if num_merges < 0:
        raise ValueError(f"vocab_size ({vocab_size}) 太小，至少需要 260")

    if verbose >= 1:
        print(f"目标词表: {vocab_size} (4 特殊 + 256 字节 + {num_merges} 合并)")
        print("构建初始 pair 频率...")

    # 3. 初始化 pair 计数器
    pair_counts = Counter()
    for chunk in chunks:
        for i in range(len(chunk) - 1):
            pair_counts[(chunk[i], chunk[i + 1])] += 1

    start_time = time.time()
    for step in range(num_merges):
        if not pair_counts:
            if verbose >= 1:
                print(f"第 {step} 步：没有 pair 可合并")
            break

        # 找最高频 pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]
        if best_count < 2:
            if verbose >= 1:
                print(f"第 {step} 步：最高频 pair 仅 {best_count} 次，停止")
            break

        a, b = best_pair

        # 4. 合并 + 同时重建 pair_counts（一次遍历完成）
        new_chunks = []
        new_pair_counts = Counter()
        for chunk in chunks:
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
                    new_chunk.append(next_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            # 统计新 chunk 的相邻 pair
            for j in range(len(new_chunk) - 1):
                new_pair_counts[(new_chunk[j], new_chunk[j + 1])] += 1
            new_chunks.append(new_chunk)

        chunks = new_chunks
        pair_counts = new_pair_counts

        # 记录合并规则
        vocab[next_id] = vocab[a + 4] + vocab[b + 4] 
        merges.append((a, b, next_id))

        # 进度输出
        if verbose >= 1 and (step % 50 == 0 or step < 10):
            try:
                preview = vocab[next_id].decode("utf-8")
            except UnicodeDecodeError:
                preview = repr(vocab[next_id])
            elapsed = time.time() - start_time
            if step > 0:
                speed = step / elapsed
                eta = (num_merges - step) / speed if speed > 0 else 0
                eta_str = f"ETA {eta:.1f}s"
            else:
                eta_str = ""
            sys.stdout.write(
                f"\r  step {step:4d}/{num_merges}: ({a},{b}) → {next_id}  "
                f"\"{preview}\"  ({best_count}次)  {eta_str}   "
            )
            sys.stdout.flush()

        next_id += 1

    if verbose >= 1:
        print()  # 换行
        print(f"训练完成: {len(vocab)} 个 token, {len(merges)} 条合并规则")
    return vocab, merges


def encode(text: str, merges: list[tuple[int, int, int]]) -> list[int]:
    """用训练好的 merges 编码文本"""
    chunks = text_to_bytes_chunks(text)

    # 构建合并优先级表
    merge_priority = {}
    for idx, (a, b, new_id) in enumerate(merges):
        merge_priority[(a, b)] = idx

    all_ids = []
    for chunk in chunks:
        ids = chunk[:]
        while len(ids) >= 2:
            best_idx = None
            best_priority = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in merge_priority and merge_priority[pair] < best_priority:
                    best_priority = merge_priority[pair]
                    best_idx = i
            if best_idx is None:
                break
            new_id = merges[best_priority][2]
            ids = ids[:best_idx] + [new_id] + ids[best_idx + 2:]
        all_ids.extend(ids)
    return all_ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """将 token ids 解码为文本"""
    raw = b""
    for tid in ids:
        if tid in vocab:
            raw += vocab[tid]
        else:
            raw += b"\xef\xbf\xbd"  # 未知 token 替换为 �
    return raw.decode("utf-8", errors="replace")


def save_bpe_json(vocab: dict, merges: list, path: str):
    """保存为 JSON（符合您的格式要求）"""
    vocab_json = {str(tid): bs.hex() for tid, bs in vocab.items()}
    merges_json = [[a, b, nid] for a, b, nid in merges]

    data = {
        "type": "freq_based_tokenizer",
        "vocab": vocab_json,
        "merges": merges_json,
        "vocab_size": len(vocab) + 4,   # 含特殊 token
        "byte_offset": 4,
        "special_tokens": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"已保存: {path} ({len(vocab)} 字节/子词 + 4 特殊 = {len(vocab)+4} 总词表)")


def main():
    parser = argparse.ArgumentParser(description="字节级 BPE 训练（纯标准库，优化版）")
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--vocab-size", type=int, default=10000, help="目标词表大小 (含特殊token，默认 10000)")
    parser.add_argument("--output", default="gpt_bpe.json", help="输出 JSON 路径")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="0=静默, 1=每50步, 2=每10步（仅用于控制打印间隔，当前固定为每50步）")
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"文本: {len(text):,} 字符, {len(text.encode('utf-8')):,} 字节")

    # 训练
    vocab, merges = train_bpe(text, args.vocab_size, verbose=args.verbose)

    # 快速验证
    test = "Alice was a very good girl, she said hello!"
    ids = encode(test, merges)
    decoded = decode(ids, vocab)
    print(f"\n验证:")
    print(f"  原文:  {test}")
    print(f"  编码:  {ids}")
    print(f"  解码:  {decoded}")
    print(f"  匹配:  {decoded == test}")

    save_bpe_json(vocab, merges, args.output)


if __name__ == "__main__":
    main()