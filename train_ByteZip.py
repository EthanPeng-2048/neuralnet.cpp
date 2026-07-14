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


# ── 核心：构建高频词表（去嵌套冗余） ──────────────────────
def build_vocab(text: str, max_vocab_size: int) -> list[bytes]:
    """
    1. 统计所有长度 2~16 的子串频率
    2. 按频率降序、长度降序排序
    3. 取 Top-K，但跳过已被更长的词覆盖的子串（去冗余）
    """
    chunks = pre_tokenize_chunks(text)

    print("正在统计子串频率（单次扫描）...")
    freq = Counter()

    for chunk in chunks:
        L = len(chunk)
        if L < 2:
            continue
        # 只取块内所有连续子串，限制最大长度
        for start in range(L):
            max_end = min(start + MAX_SUBWORD_LEN, L)
            if start + 2 > max_end:
                break
            for end in range(start + 2, max_end + 1):
                sub = bytes(chunk[start:end])
                freq[sub] += 1

    print(f"统计完成，共 {len(freq):,} 种不同子串，正在排序...")
    # 排序：频率高的优先，同频时长的优先（让大词先进）
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))

    # 初始化词表：特殊 token(0-3) + 256个单字节(4-259)
    vocab = [b""] * 4 + [bytes([i]) for i in range(256)]
    
    # 核心去冗余集合：记录所有已被收录词条的“真子串”（长度≥2）
    skip = set()

    print(f"开始填充词表（目标 {max_vocab_size}）...")
    added = 0
    for sub, count in sorted_items:
        if len(vocab) >= max_vocab_size:
            break
        if count < MIN_FREQ:
            break  # 后面的频率更低，直接终止
        
        # 跳过单字节（已经存在）以及被长词覆盖的子串
        if len(sub) <= 1 or sub in skip:
            continue

        # 将当前词加入词表
        vocab.append(sub)
        added += 1

        # 把当前词的所有真子串（长度 >=2）加入 skip 集合
        # 例如 b"husband" -> 跳过 b"husban", b"husba", ..., b"us", b"an"
        for i in range(len(sub)):
            for j in range(i + 2, len(sub) + 1):   # 加 1，让 j 能取到 len(sub)
                if i == 0 and j == len(sub):
                    continue  # 跳过自己（不跳过自身会把自己从词表删掉）
                skip.add(sub[i:j])

        # 每 1000 个打印一次进度
        if added % 1000 == 0:
            print(f"  已添加 {added} 个词条（当前词表 {len(vocab)}）")

    print(f"构建完成！最终词表大小: {len(vocab)}（新增 {added} 个子词）")
    return vocab


# ── 编码器（基于 Trie 树，O(n) 贪心最长匹配） ──────────────
def encode(text: str, vocab: list[bytes]) -> list[int]:
    """
    将文本编码为 Token ID 列表。
    贪心策略：每次从当前位置开始，在 Trie 中走最远匹配。
    """
    # 构建嵌套 Trie 树
    trie = {}
    for tid in range(BYTE_OFFSET, len(vocab)):
        bs = vocab[tid]
        if not bs:
            continue
        node = trie
        for b in bs:
            if b not in node:
                node[b] = {}
            node = node[b]
        node["_id"] = tid  # 标记终点 ID

    chunks = pre_tokenize_chunks(text)
    all_ids = []

    for chunk in chunks:
        data = list(chunk)
        pos = 0
        L = len(data)
        while pos < L:
            node = trie
            best_id = data[pos] + BYTE_OFFSET  # 兜底：单字节
            best_len = 1

            offset = 0
            while pos + offset < L and data[pos + offset] in node:
                node = node[data[pos + offset]]
                offset += 1
                if "_id" in node:
                    best_id = node["_id"]
                    best_len = offset

            all_ids.append(best_id)
            pos += best_len

    return all_ids


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
    """
    贪心最长匹配编码（极速版）。
    直接操作 UTF-8 字节流，跳过正则预分词，速度提升 10 倍以上。
    
    参数:
        text: 原始字符串
        lookup: build_lookup 生成的字典
        max_len: 最大匹配长度（必须 >= 训练时的 MAX_SUBWORD_LEN）
    """
    data = text.encode("utf-8")  # 转为 bytes，可索引、可切片
    n = len(data)
    ids = []
    pos = 0

    while pos < n:
        # 兜底：单字节（ID = 字节值 + BYTE_OFFSET）
        best_id = data[pos] + BYTE_OFFSET
        best_len = 1

        # 限制搜索范围，避免无谓的切片
        max_search = min(max_len, n - pos)
        # 从长到短尝试匹配（贪心最长）
        for length in range(max_search, 1, -1):
            sub = data[pos:pos + length]
            tid = lookup.get(sub)
            if tid is not None:
                best_id = tid
                best_len = length
                break  # 找到最长匹配立即退出

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
    args = parser.parse_args()

    # 将最大长度设为全局变量（影响构建函数）
    global MAX_SUBWORD_LEN
    MAX_SUBWORD_LEN = args.max_len

    print(f"读取文件: {args.text_file}")
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"文本字符数: {len(text):,}")

    import time
    t0 = time.time()
    vocab = build_vocab(text, args.vocab_size)
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