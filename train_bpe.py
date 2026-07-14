"""
字节级 BPE 分词器训练脚本（纯 Python 标准库，零依赖）

灵感: Karpathy minbpe / GPT-2 byte-level BPE

词表构成:
  ID 0-3:    特殊 token (<pad> <unk> <bos> <eos>)
  ID 4-259:  256 个字节 (0x00-0xFF)
  ID 260+:   BPE 合并产生的子词

用法:
  python train_bpe.py dataset.txt --vocab-size 10000
  python train_bpe.py dataset.txt --vocab-size 5000 --output my_bpe.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ── 预分词正则（GPT-2 风格，仅用标准库 re）──────────────────
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


# ── BPE 训练 ──────────────────────────────────────────────────
def get_pair_counts(chunks: list[list[int]]) -> Counter:
    """统计所有相邻 pair 的频率"""
    pairs = Counter()
    for chunk in chunks:
        for i in range(len(chunk) - 1):
            pairs[(chunk[i], chunk[i + 1])] += 1
    return pairs


def merge_in_chunks(chunks: list[list[int]], pair: tuple[int, int], new_id: int) -> list[list[int]]:
    """将所有 chunk 中的 pair (a, b) 合并为 new_id"""
    merged = []
    for chunk in chunks:
        new_chunk = []
        i = 0
        while i < len(chunk):
            if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i + 1] == pair[1]:
                new_chunk.append(new_id)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        merged.append(new_chunk)
    return merged


def train_bpe(text: str, vocab_size: int, verbose: bool = True) -> tuple[dict, list]:
    """
    训练字节级 BPE。

    返回:
      vocab: {id: bytes}  — 词表（id → 字节序列）
      merges: [(id_a, id_b, new_id), ...]  — 合并规则
    """
    # 1. 预分词 + 转字节
    chunks = text_to_bytes_chunks(text)
    if verbose:
        print(f"预分词后 {len(chunks)} 个块")

    # 2. 初始词表：256 个字节
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # 特殊 token 占位（ID 0-3 在 C++ 端处理，这里不加入 vocab 字节映射）
    next_id = 256  # BPE 合并从 256 开始

    num_merges = vocab_size - 256
    if verbose:
        print(f"目标词表: {vocab_size} (256 字节 + {num_merges} 合并)")

    # 3. 迭代合并
    for step in range(num_merges):
        stats = get_pair_counts(chunks)
        if not stats:
            print(f"在第 {step} 步没有更多 pair 可合并")
            break

        best_pair = max(stats, key=stats.get)
        best_count = stats[best_pair]

        if best_count < 2:
            print(f"在第 {step} 步最高频 pair 只出现 {best_count} 次，停止")
            break

        # 执行合并
        chunks = merge_in_chunks(chunks, best_pair, next_id)

        # 记录：新 token 的字节 = 两个子 token 的字节拼接
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((best_pair[0], best_pair[1], next_id))

        if verbose and (step % 500 == 0 or step < 10):
            merged_bytes = vocab[next_id]
            try:
                preview = merged_bytes.decode("utf-8")
            except UnicodeDecodeError:
                preview = repr(merged_bytes)
            print(f"  step {step:4d}: 合并 ({best_pair[0]}, {best_pair[1]}) → {next_id}"
                  f"  \"{preview}\"  出现 {best_count} 次")

        next_id += 1

    if verbose:
        print(f"训练完成: {len(vocab)} 个 token, {len(merges)} 条合并规则")
    return vocab, merges


# ── 编码 / 解码 ───────────────────────────────────────────────
def encode(text: str, merges: list[tuple[int, int, int]]) -> list[int]:
    """用训练好的 merges 编码文本"""
    chunks = text_to_bytes_chunks(text)

    # 构建合并优先级表：pair → priority (越小越优先)
    merge_priority = {}
    for idx, (a, b, new_id) in enumerate(merges):
        merge_priority[(a, b)] = idx

    all_ids = []
    for chunk in chunks:
        # 对每个 chunk 按优先级贪心合并
        ids = chunk[:]
        while len(ids) >= 2:
            # 找当前最高优先级的 pair
            best_idx = None
            best_priority = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in merge_priority and merge_priority[pair] < best_priority:
                    best_priority = merge_priority[pair]
                    best_idx = i

            if best_idx is None:
                break

            # 执行合并
            a, b = ids[best_idx], ids[best_idx + 1]
            new_id = merges[best_priority][2]  # new_id from merge rule
            ids = ids[:best_idx] + [new_id] + ids[best_idx + 2:]

        all_ids.extend(ids)
    return all_ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """将 token ids 解码为文本"""
    raw = b""
    for tid in ids:
        if tid in vocab:
            raw += vocab[tid]
        elif 0 <= tid < 256:
            raw += bytes([tid])
        else:
            raw += b"\xef\xbf\xbd"  # UTF-8 replacement character
    return raw.decode("utf-8", errors="replace")


# ── 保存 / 加载 ───────────────────────────────────────────────
def save_bpe_json(vocab: dict, merges: list, path: str):
    """保存为 JSON（C++ 和 Python 都能读）"""
    # vocab: {id: hex_string}
    vocab_json = {}
    for tid, bs in vocab.items():
        vocab_json[tid] = bs.hex()

    # merges: [[a, b, new_id], ...]
    merges_json = [[a, b, nid] for a, b, nid in merges]

    data = {
        "vocab": vocab_json,
        "merges": merges_json,
        "vocab_size": len(vocab) + 4,  # +4 for special tokens
        "num_merges": len(merges),
        "special_tokens": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"已保存: {path} ({len(vocab)} 字节/子词 + {len(merges)} 条合并)")


# ── 主函数 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="字节级 BPE 训练（纯标准库）")
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--vocab-size", type=int, default=10000, help="目标词表大小 (默认: 10000)")
    parser.add_argument("--output", default="gpt_bpe.json", help="输出 JSON 路径")
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"文本: {len(text):,} 字符, {len(text.encode('utf-8')):,} 字节")

    vocab, merges = train_bpe(text, args.vocab_size)

    # 验证
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
