"""
终极分词器训练脚本（纯 Python 标准库，零依赖）
结合了频率统计 + 去嵌套冗余，词表质量接近 BPE，速度极快。

用法:
  python train_tokenizer.py dataset.txt --vocab-size 10000
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ── 配置常量 ──────────────────────────────────────────────
SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
BYTE_OFFSET = 4          # 特殊 token 占 0-3，字节从 4 开始
MAX_SUBWORD_LEN = 16     # 限制子串最大长度（防止统计爆炸，英文 16 足够）
MIN_FREQ = 2             # 最低出现次数（过滤只出现 1 次的生僻碎片）

# ── 预分词正则 ──────────────────────────────────────────────
# 方案 A：专供英文（推荐英文语料，精度更高）
PAT_EN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+""",
    re.IGNORECASE,
)

# 方案 B：通用（支持中文/多语言，按空白和标点切块）
PAT_ANY = re.compile(r"""[^\s]+""")  # 匹配任意非空白字符序列

# 默认使用英文正则（如果你跑中文，把下面的 PAT 改成 PAT_ANY）
PAT = PAT_EN


def pre_tokenize_chunks(text: str) -> list[bytes]:
    """预分词：将文本切成块，每块转成 UTF-8 字节序列"""
    return [piece.encode("utf-8") for piece in PAT.findall(text)]


# ── 核心：构建高频词表（独立度智能去冗余） ──────────────────
def build_vocab(text: str, max_vocab_size: int, skip_ratio: float = 1.2) -> list[bytes]:
    """
    1. 统计所有长度 2~16 的子串频率
    2. 按压缩收益降序排序
    3. 取 Top-K，用「独立度」判断子串是否值得保留：
       - freq(sub) >> freq(parent) → 有独立语言学价值，不 skip
       - freq(sub) ≈ freq(parent)  → 纯冗余，skip
    """
    chunks = pre_tokenize_chunks(text)

    print("正在统计子串频率（单次扫描）...")
    freq = Counter()

    for chunk in chunks:
        L = len(chunk)
        if L < 2:
            continue
        for start in range(L):
            max_end = min(start + MAX_SUBWORD_LEN, L)
            if start + 2 > max_end:
                break
            for end in range(start + 2, max_end + 1):
                sub = bytes(chunk[start:end])
                freq[sub] += 1

    print(f"统计完成，共 {len(freq):,} 种不同子串，正在排序...")
    # 排序：压缩收益高的优先 (freq * (len - 1))，同收益时长的优先
    sorted_items = sorted(
        freq.items(), 
        key=lambda x: (-(x[1] * (len(x[0]) - 1)), -len(x[0]))
    )

    # 初始化词表：特殊 token(0-3) + 256个单字节(4-259)
    vocab = [b""] * 4 + [bytes([i]) for i in range(256)]
    
    # ── 独立度 skip 集合 ──────────────────────────────────────
    # 只有当子串的频率 ≤ parent × SKIP_RATIO 时才 skip（纯冗余）
    # 否则认为子串有独立语言学价值，允许后续被选入词表
    skip = set()
    SKIP_RATIO = skip_ratio  # α：子串频率需超过 parent 的 α 倍才算「独立」

    print(f"开始填充词表（目标 {max_vocab_size}，SKIP_RATIO={SKIP_RATIO}）...")
    added = 0
    for sub, count in sorted_items:
        if len(vocab) >= max_vocab_size:
            break
        if count < MIN_FREQ:
            break
        
        if len(sub) <= 1 or sub in skip:
            continue

        # 将当前词加入词表
        vocab.append(sub)
        added += 1

        # ── 智能 skip：只跳过「纯冗余」的真子串 ──
        # 如果 freq(sub_child) > α × freq(parent)，
        # 说明 sub_child 在大量其他地方也出现，不应 skip
        # 例如：ing(50000) ≫ doing(2000) → 保留 ing
        #       stonecraft(20) ≈ Wollstonecraft(20) → skip stonecraft
        for i in range(len(sub)):
            for j in range(i + 2, len(sub) + 1):
                if i == 0 and j == len(sub):
                    continue  # 跳过自身
                child = sub[i:j]
                child_freq = freq.get(child, 0)
                if child_freq <= count * SKIP_RATIO:
                    # 子串频率接近父串 → 纯冗余，skip
                    skip.add(child)
                # 否则：子串有独立价值，不加入 skip，留给后续处理

        if added % 1000 == 0:
            print(f"  已添加 {added} 个词条（当前词表 {len(vocab)}，skip集 {len(skip)} 项）")

    print(f"构建完成！最终词表大小: {len(vocab)}（新增 {added} 个子词，skip集 {len(skip)} 项）")
    return vocab


# ── 编码器（基于 Trie 树，O(n) 贪心最长匹配） ──────────────
# ── 编码器（极速重写版：自动继承 encode_fast 的所有优化） ──
def encode(text: str, vocab: list[bytes]) -> list[int]:
    """
    将文本编码为 Token ID 列表。
    内部直接调用极速哈希表 + 形态学优先拆分。
    """
    lookup = build_lookup(vocab)
    return encode_fast(text, lookup, MAX_SUBWORD_LEN)


# ── 解码器（完全不需要修改！） ────────────────────────────
def decode(ids: list[int], vocab: list[bytes]) -> str:
    """将 Token ID 列表还原为原始文本"""
    raw = b""
    for tid in ids:
        if 0 <= tid < len(vocab):
            raw += vocab[tid]
        else:
            # 防御：如果 ID 越界，用 � 占位
            raw += b"\xef\xbf\xbd"
    return raw.decode("utf-8", errors="replace")


# ── 保存词表 ──────────────────────────────────────────────
def save_vocab(vocab: list[bytes], path: str):
    entries = {}
    for tid in range(BYTE_OFFSET, len(vocab)):
        entries[tid] = vocab[tid].hex()

    data = {
        "type": "freq_based_tokenizer",
        "vocab": entries,
        "vocab_size": len(vocab),
        "byte_offset": BYTE_OFFSET,
        "special_tokens": SPECIAL_TOKENS,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"词表已保存至: {path} ({len(vocab)} tokens)")


# ── 新增：构建快速查找表（一次性） ──────────────────────────
def build_lookup(vocab: list[bytes]) -> dict:
    """
    将词表转换成 {字节序列: token_id} 的扁平字典。
    用于 encode_fast 的 O(1) 哈希匹配，比嵌套 Trie 快得多。
    """
    lookup = {}
    for tid in range(BYTE_OFFSET, len(vocab)):
        bs = vocab[tid]
        if bs:  # 跳过空字节
            lookup[bs] = tid
    return lookup


# ── 新增：极速编码函数（无预分词，直接扫描全量 bytes） ──────
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

# ── 主入口 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="训练高频子词分词器（纯标准库）")
    parser.add_argument("text_file", help="训练文本文件路径（TXT）")
    parser.add_argument("--vocab-size", type=int, default=10000, help="目标词表大小")
    parser.add_argument("--output", default="tokenizer.json", help="输出 JSON 路径")
    parser.add_argument("--max-len", type=int, default=16, help="最大子词长度")
    parser.add_argument("--skip-ratio", type=float, default=1.2, help="独立度阈值 α：子串频率需超过 parent 的 α 倍才保留（默认 1.2）")
    args = parser.parse_args()

    # 将最大长度设为全局变量（影响构建函数）
    global MAX_SUBWORD_LEN
    MAX_SUBWORD_LEN = args.max_len

    print(f"读取文件: {args.text_file}")
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"文本字符数: {len(text):,}")

    import time
    t0 = time.time()
    vocab = build_vocab(text, args.vocab_size, skip_ratio=args.skip_ratio)
    t1 = time.time()
    print(f"总耗时: {t1 - t0:.1f} 秒")

    # 展示前 20 个高频子词（验收质量）
    print("\n✅ 高频子词展示（前 20 个）:")
    shown = 0
    for tid in range(BYTE_OFFSET, min(len(vocab), BYTE_OFFSET + 200)):
        try:
            s = vocab[tid].decode("utf-8")
            if len(s) >= 2 and s.strip():
                print(f"  ID {tid}: \"{s}\"")
                shown += 1
                if shown >= 20:
                    break
        except UnicodeDecodeError:
            pass

    # 编码/解码冒烟测试
    test_sent = "Alice was a very good girl, she said hello!"
    ids = encode(test_sent, vocab)
    decoded = decode(ids, vocab)
    print(f"\n🧪 验证:")
    print(f"  原文: {test_sent}")
    print(f"  Token数: {len(ids)}")
    print(f"  解码: {decoded}")
    print(f"  完美还原: {decoded == test_sent}")

    save_vocab(vocab, args.output)


if __name__ == "__main__":
    main()