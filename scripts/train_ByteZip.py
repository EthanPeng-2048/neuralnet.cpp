"""
ByteZip v2.2 - 极速版（针对小词表跳过 V2，大词表采样压缩）
"""

import argparse
import json
import re
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────
SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
BYTE_OFFSET = 4
V1_MAX_LEN = 16
V2_MAX_LEN = 24              # 缩小范围，减少循环
MIN_FREQ = 2
SKIP_RATIO = 1.2
AFFIX_PROTECT_RATIO = 0.4
MAX_V2_SCAN_BYTES = 50_000_000  # 从 2MB 降至 300KB

PAT_EN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+""",
    re.IGNORECASE,
)

def pre_tokenize_chunks(text: str) -> list[bytes]:
    return [piece.encode("utf-8") for piece in PAT_EN.findall(text)]

# ── V1 扫描 ──────────────────────────────────────────────
def scan_v1(text: str, max_len: int):
    chunks = pre_tokenize_chunks(text)
    freq = Counter()
    left_ctx = defaultdict(set)
    right_ctx = defaultdict(set)

    for chunk in chunks:
        L = len(chunk)
        if L < 2:
            continue
        for start in range(L):
            max_end = min(start + max_len, L)
            if start + 2 > max_end:
                break
            for end in range(start + 2, max_end + 1):
                sub = bytes(chunk[start:end])
                freq[sub] += 1
                if start > 0:
                    left_ctx[sub].add(chunk[start-1])
                else:
                    left_ctx[sub].add(None)
                if end < L:
                    right_ctx[sub].add(chunk[end])
                else:
                    right_ctx[sub].add(None)
    return freq, left_ctx, right_ctx

def is_affix(sub: bytes, left_ctx: dict, right_ctx: dict):
    if sub not in left_ctx or sub not in right_ctx:
        return False, False
    len_l, len_r = len(left_ctx[sub]), len(right_ctx[sub])
    return (len_l >= 3 and len_l > len_r * 2.0), (len_r >= 3 and len_r > len_l * 2.0)

def build_v1(freq, left_ctx, right_ctx, target_size):
    vocab = [b""] * 4 + [bytes([i]) for i in range(256)]
    sorted_items = sorted(freq.items(), key=lambda x: (-(x[1] * (len(x[0]) - 1)), -len(x[0])))

    skip = set()
    added = 0
    for sub, count in sorted_items:
        if len(vocab) >= target_size:
            break
        if count < MIN_FREQ or len(sub) <= 1 or sub in skip:
            continue

        suffix_flag, prefix_flag = is_affix(sub, left_ctx, right_ctx)
        threshold = AFFIX_PROTECT_RATIO if (suffix_flag or prefix_flag) else SKIP_RATIO

        for i in range(len(sub)):
            for j in range(i + 2, len(sub) + 1):
                if i == 0 and j == len(sub):
                    continue
                child = sub[i:j]
                if freq.get(child, 0) <= count * threshold:
                    skip.add(child)

        vocab.append(sub)
        added += 1
        if added % 1000 == 0:
            print(f"  V1: 已添加 {added} 个词条 (总 {len(vocab)})")

    print(f"V1 完成: {len(vocab)} 词条 (新增 {added})")
    return vocab

def build_lookup(vocab, max_len):
    lookup = [{} for _ in range(max_len + 1)]
    for tid, bs in enumerate(vocab):
        if bs:
            L = len(bs)
            if L <= max_len:
                lookup[L][bs] = tid
    return lookup

def encode_bytes(data: bytes, lookup: list[dict], max_len: int):
    n = len(data)
    ids, positions = [], []
    pos = 0
    while pos < n:
        positions.append(pos)
        match_len = 1
        match_id = data[pos] + BYTE_OFFSET
        max_search = min(max_len, n - pos)
        for L in range(max_search, 1, -1):
            tid = lookup[L].get(data[pos:pos+L])
            if tid is not None:
                match_len, match_id = L, tid
                break
        ids.append(match_id)
        pos += match_len
    return ids, positions

def encode(text: str, lookup: list[dict], max_len: int) -> list[int]:
    ids, _ = encode_bytes(text.encode("utf-8"), lookup, max_len)
    return ids

def decode(ids, vocab):
    return b"".join(vocab[tid] for tid in ids).decode("utf-8", errors="replace")

# ── V2 残差（仅在大词表时启用） ──────────────────────────
def get_residual_spans(text: str, lookup_v1: list[dict], max_len: int) -> list[bytes]:
    data = text.encode("utf-8")
    ids, positions = encode_bytes(data, lookup_v1, max_len)
    n_tokens = len(ids)
    spans = []
    i = 0
    while i < n_tokens:
        if ids[i] < BYTE_OFFSET + 256:
            start_pos = positions[i]
            j = i
            while j < n_tokens and ids[j] < BYTE_OFFSET + 256:
                j += 1
            end_pos = positions[j] if j < n_tokens else len(data)
            ext_start = max(0, start_pos - 8)
            ext_end = min(len(data), end_pos + 8)
            spans.append(data[ext_start:ext_end])
            i = j
        else:
            i += 1
    return spans

def build_v2_from_spans(spans, max_len, max_items):
    total_bytes = sum(len(s) for s in spans)
    if total_bytes == 0 or max_items <= 0:
        return []

    # 采样
    if total_bytes > MAX_V2_SCAN_BYTES:
        sampled = bytearray()
        random.shuffle(spans)
        for span in spans:
            if len(sampled) >= MAX_V2_SCAN_BYTES:
                break
            remaining = MAX_V2_SCAN_BYTES - len(sampled)
            sampled.extend(span[:remaining])
        spans = [bytes(sampled)]

    freq_v2 = Counter()
    for span in spans:
        L = len(span)
        if L < 16:
            continue
        for start in range(L):
            max_end = min(start + max_len, L)
            if start + 16 > max_end:
                break
            for end in range(start + 16, max_end + 1):
                sub = bytes(span[start:end])
                freq_v2[sub] += 1

    sorted_v2 = sorted(freq_v2.items(), key=lambda x: (-(x[1] * (len(x[0]) - 1)), -len(x[0])))
    result = [sub for sub, cnt in sorted_v2 if cnt >= 2]
    return result[:max_items]

# ── 主流程 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--output", default="tokenizer_v2.json")
    parser.add_argument("--v1-len", type=int, default=16)
    parser.add_argument("--v2-len", type=int, default=24)
    parser.add_argument("--v2-reserve", type=int, default=0)
    args = parser.parse_args()

    global V1_MAX_LEN, V2_MAX_LEN
    V1_MAX_LEN, V2_MAX_LEN = args.v1_len, args.v2_len

    print(f"读取: {args.text_file}")
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"字符数: {len(text):,}")
    t0 = time.time()

    # ── 智能分配 V1/V2 槽位 ──
    if args.vocab_size <= 1500:
        # 小词表：跳过 V2，所有槽位给 V1，避免 70 秒慢扫描
        v2_reserve = 0
        v1_target = args.vocab_size
        print(f"小词表模式 (<=1500): 跳过 V2，全部给 V1 (目标 {v1_target})")
    else:
        if args.v2_reserve == 0:
            v2_reserve = min(1500, int(args.vocab_size * 0.12))
        else:
            v2_reserve = args.v2_reserve
        v1_target = args.vocab_size - v2_reserve
        if v1_target < BYTE_OFFSET + 256:
            v1_target = BYTE_OFFSET + 256
            v2_reserve = args.vocab_size - v1_target
        print(f"大词表模式: V1 目标 {v1_target}, V2 预留 {v2_reserve}")

    # ── V1 ──
    print("\n[V1] 统计...")
    freq, left_ctx, right_ctx = scan_v1(text, V1_MAX_LEN)
    print(f"  不同子串数: {len(freq):,}")
    vocab = build_v1(freq, left_ctx, right_ctx, v1_target)
    lookup_v1 = build_lookup(vocab, V1_MAX_LEN)

    # ── V2（仅当预留 > 0） ──
    if v2_reserve > 0 and args.vocab_size > 1500:
        print("\n[V2] 提取残差...")
        spans = get_residual_spans(text, lookup_v1, V1_MAX_LEN)
        print(f"  残差跨度: {len(spans):,}, 总字节: {sum(len(s) for s in spans):,}")
        if spans:
            print(f"  挖掘超长短语 (最多 {v2_reserve} 个)...")
            v2_words = build_v2_from_spans(spans, V2_MAX_LEN, v2_reserve)
            existing = set(vocab)
            for w in v2_words:
                if len(vocab) >= args.vocab_size:
                    break
                if w not in existing:
                    vocab.append(w)
                    existing.add(w)
            print(f"  V2 新增 {len(v2_words)} 个")
    else:
        print("\n[V2] 跳过（小词表或预留为 0）")

    print(f"\n最终词表: {len(vocab)}")
    t1 = time.time()
    print(f"总耗时: {t1 - t0:.1f} 秒")

    # 测试
    max_len = max(V1_MAX_LEN, V2_MAX_LEN)
    lookup_final = build_lookup(vocab, max_len)
    test = "Alice was a very good girl, she said hello! Sir Arthur Conan Doyle wrote Sherlock Holmes."
    ids = encode(test, lookup_final, max_len)
    dec = decode(ids, vocab)
    print(f"测试: {dec == test}  Token数: {len(ids)}")

    # 保存
    entries = {str(tid): vocab[tid].hex() for tid in range(BYTE_OFFSET, len(vocab)) if vocab[tid]}
    data = {
        "type": "freq_based_tokenizer",
        "vocab": entries,
        "vocab_size": len(vocab),
        "byte_offset": BYTE_OFFSET,
        "special_tokens": SPECIAL_TOKENS,
    }
    Path(args.output).write_text(json.dumps(data, indent=2))
    print(f"保存至: {args.output}")

if __name__ == "__main__":
    main()