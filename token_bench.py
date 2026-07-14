#!/usr/bin/env python3
"""
分词器质量评估工具
用法：
    python evaluate_tokenizer.py tokenizer.json test.txt
    python evaluate_tokenizer.py tokenizer.json test.txt --max-len 8
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# 默认的字节偏移（与训练脚本保持一致）
BYTE_OFFSET = 4

# ── BPE 预分词正则（GPT-2 风格）──────────────────────────────
PAT_BPE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+""",
    re.IGNORECASE,
)


def bpe_pre_tokenize(text: str) -> list[str]:
    """按正则将文本切成块（BPE 预分词）"""
    return PAT_BPE.findall(text)


def bpe_text_to_bytes_chunks(text: str) -> list[list[int]]:
    """预分词后，每块转为 UTF-8 字节序列"""
    chunks = []
    for piece in bpe_pre_tokenize(text):
        chunks.append(list(piece.encode("utf-8")))
    return chunks


def encode_bpe(text: str, merges: list[tuple[int, int, int]]) -> list[int]:
    """BPE 编码：基于合并规则，按优先级逐步合并"""
    chunks = bpe_text_to_bytes_chunks(text)

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


def load_tokenizer(json_path: str) -> dict:
    """加载词表 JSON，返回 tokenizer 数据字典"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 确定词表大小
    vocab_size = data.get('vocab_size', 0)
    if not vocab_size:
        vocab_size = max(int(k) for k in data['vocab']) + 1
    # 补齐 vocab 到 vocab_size
    vocab = [b''] * vocab_size
    for id_str, hex_str in data['vocab'].items():
        tid = int(id_str)
        vocab[tid] = bytes.fromhex(hex_str)
    data['vocab'] = vocab
    return data


def is_bpe(tokenizer_data: dict) -> bool:
    """判断是否为 BPE 分词器（有 merges 字段）"""
    return 'merges' in tokenizer_data and len(tokenizer_data['merges']) > 0


def encode_auto(text: str, tokenizer_data: dict, max_len: int = 16) -> list[int]:
    """根据分词器类型自动选择编码器"""
    vocab = tokenizer_data['vocab']
    if is_bpe(tokenizer_data):
        merges = [tuple(m) for m in tokenizer_data['merges']]
        return encode_bpe(text, merges)
    else:
        lookup = build_lookup(vocab)
        return encode_fast(text, lookup, max_len)


def build_lookup(vocab: list[bytes]) -> dict[bytes, int]:
    """构建 {字节序列: token_id} 查找表，用于贪心编码"""
    return {bs: tid for tid, bs in enumerate(vocab) if bs}


def encode_fast(text: str, lookup: dict, max_len: int = 16) -> list[int]:
    data = text.encode("utf-8")
    n = len(data)
    ids = []
    pos = 0

    while pos < n:
        # 默认 fallback 到单字节
        best_id = data[pos] + BYTE_OFFSET
        best_len = 1
        max_search = min(max_len, n - pos)

        # 贪心寻找最长匹配
        for length in range(max_search, 1, -1):
            sub = data[pos:pos + length]
            tid = lookup.get(sub)
            if tid is not None:
                best_id = tid
                best_len = length
                break  # 找到最长匹配，立即停止
        
        # 直接加入结果，绝不强行拆分
        ids.append(best_id)
        pos += best_len

    return ids


def evaluate(text: str, tokenizer_data: dict, max_len: int = None) -> dict:
    """
    计算各项评估指标。
    若 max_len 为 None，自动取词表中最长 token 的长度。
    """
    vocab = tokenizer_data['vocab']
    use_bpe = is_bpe(tokenizer_data)

    # 自动推断最大匹配长度（仅 ByteZip 需要）
    if not use_bpe and max_len is None:
        max_len = max((len(bs) for bs in vocab if bs), default=1)

    ids = encode_auto(text, tokenizer_data, max_len or 16)
    num_tokens = len(ids)
    num_chars = len(text)
    num_bytes = len(text.encode('utf-8'))

    # 统计 token 长度（字节）
    token_lengths = [len(vocab[tid]) for tid in ids if 0 <= tid < len(vocab) and vocab[tid]]
    avg_len = sum(token_lengths) / num_tokens if num_tokens else 0

    # 压缩率
    tok_per_char = num_tokens / num_chars if num_chars else 0
    tok_per_byte = num_tokens / num_bytes if num_bytes else 0

    # 唯一 token 数
    unique_tokens = len(set(ids))

    # 覆盖率（词表内 token 的比例，理想情况为 1.0）
    covered = sum(1 for tid in ids if 0 <= tid < len(vocab) and vocab[tid])
    coverage = covered / num_tokens if num_tokens else 1.0

    return {
        'num_chars': num_chars,
        'num_bytes': num_bytes,
        'num_tokens': num_tokens,
        'avg_token_len': avg_len,
        'tok_per_char': tok_per_char,
        'tok_per_byte': tok_per_byte,
        'unique_tokens': unique_tokens,
        'coverage': coverage,
        'max_len': max_len,
        # 保留前 20 个 token 作为示例（可选）
        'sample_ids': ids[:20],
    }


def main():
    parser = argparse.ArgumentParser(description='评估分词器质量')
    parser.add_argument('tokenizer_json', help='词表 JSON 文件')
    parser.add_argument('test_text', help='测试文本文件')
    parser.add_argument('--max-len', type=int, default=None,
                        help='最大匹配长度（字节），仅 ByteZip 分词器有效，BPE 忽略此参数')
    args = parser.parse_args()

    # 加载分词器和测试文本
    tokenizer_data = load_tokenizer(args.tokenizer_json)
    vocab = tokenizer_data['vocab']
    text = Path(args.test_text).read_text(encoding='utf-8')

    # 评估
    metrics = evaluate(text, tokenizer_data, args.max_len)

    # 输出结果
    print("=" * 50)
    print("分词器评估结果")
    print("=" * 50)
    encoder_type = "BPE（合并规则）" if is_bpe(tokenizer_data) else "ByteZip（贪心最长匹配）"
    print(f"编码器类型:         {encoder_type}")
    print(f"测试文本字符数:     {metrics['num_chars']:,}")
    print(f"测试文本字节数:     {metrics['num_bytes']:,}")
    print(f"编码后 Token 数:    {metrics['num_tokens']:,}")
    print(f"平均 Token 长度 (字节): {metrics['avg_token_len']:.2f}")
    print(f"Token / 字符:       {metrics['tok_per_char']:.4f}")
    print(f"Token / 字节:       {metrics['tok_per_byte']:.4f}")
    print(f"唯一 Token 数:      {metrics['unique_tokens']:,}")
    print(f"词表覆盖率:         {metrics['coverage']*100:.2f}%")
    if not is_bpe(tokenizer_data):
        print(f"实际使用的最大匹配长度: {metrics['max_len']} 字节")

    # 显示分词示例（前20个token）
    sample_ids = metrics['sample_ids']
    if sample_ids:
        sample_tokens = [vocab[tid] if tid < len(vocab) else b'?' for tid in sample_ids]
        print("\n示例 Token（前20个）：")
        for i, (tid, token_bytes) in enumerate(zip(sample_ids, sample_tokens)):
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = repr(token_bytes)
            print(f"  {i:2d}: ID {tid:3d}  ->  {token_str}")


if __name__ == '__main__':
    main()