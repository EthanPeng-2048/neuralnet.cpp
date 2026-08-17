"""
使用 HuggingFace tokenizers 库训练字节级 BPE 分词器（类似 GPT-2）。

与 train_charbpe.py 的区别：
  - charbpe: 以 Unicode 字符为原子，CJK 字符预分词后按词合并
  - bytebpe: 以 UTF-8 字节为原子，按 GPT-2 风格 regex 预分词后按字节合并
  - bytebpe 对所有语言一视同仁，无需针对 CJK 特殊处理

输出格式与 C++ CharBPETokenizer 完全兼容的 JSON 文件。
ID 布局：
  0-3    : <pad>, <unk>, <bos>, <eos>
  4-259  : 256 个单字节兜底
  260+   : BPE 合并产生的多字节 token

依赖：pip install tokenizers
用法：python train_bytebpe.py <text_file> [选项]
示例：python train_bytebpe.py datasets/llm_corpus.txt --vocab-size 8000 --output bytebpe.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ── 常量（与 C++ CharBPETokenizer 一致） ──────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BYTE_BASE = 4       # 256 个单字节兜底起始 ID
CHAR_BASE = 260     # 合并 token 起始 ID


def make_pretokenizer():
    """
    构造 GPT-2 风格的字节级预分词器。

    两步流水线：先按 regex 把文本切分为"词"，再由 ByteLevel 把每个词
    转为 UTF-8 字节序列。

    关键点 1：**前导空白（\\s*）附加到后面的词**，这样空格字节能像 GPT-2
    一样合并进词 token（如 `Ġthe` → 字节 `0x20 0x74 ...` → 词表里
    `" the"`），而不是变成独立 token。

    ⚠️ 不要用 `[ \t]+` 把空白单独切成一段 —— 那样空格永远无法与词合并，
    会导致：
      - 空格 token 占全部 token 的 ~45%（严重冗余）
      - ~56% 的 token 是单字节（退化成字符级预测）
      - 训练慢、模型容易坍缩到"总是预测空格"
    必须与 C++ CharBPETokenizer::pre_tokenize 的行为一致（它也是把
    前导空白附加到下一个词）。

    关键点 2：**必须经过 ByteLevel**，否则非 ASCII（中文等）无法编码。
    仅用 Split + byte_fallback 时，tokenizers 0.23.x 不会自动补齐 256 个
    字节 token，语料中未出现的字节（如数字 '2'、'+'、中文）会变成 <unk>。

    模式参考 GPT-2 / GPT-4 的字节级预分词：
      1. 前导空白 + 连续中文/日文/韩文字符
      2. 前导空白 + 连续字母数字
      3. 前导空白 + 单个其他字符（标点等，与 C++ 每个标点单独成段一致）
    """
    pattern = (
        r"\s*[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+"
        r"|\s*[a-zA-Z0-9]+"
        r"|\s*[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\sa-zA-Z0-9]"
    )
    return Sequence([
        Split(Regex(pattern), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=True),
    ])


def train_bytebpe(text_file: str, vocab_size: int, min_freq: int) -> Tokenizer:
    """训练字节级 BPE tokenizer。"""
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=False))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = make_pretokenizer()
    tokenizer.post_processor = None
    # ByteLevel decoder 将字节序列还原为文本
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        # 关键：用 ByteLevel.alphabet() 作为初始字母表，确保全部 256 个
        # 字节 token 都在词表里。否则语料中未出现的字节会变成 <unk>。
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train(files=[text_file], trainer=trainer)
    return tokenizer


def convert_to_charbpe_json(tokenizer: Tokenizer, max_vocab: int = 100000) -> dict:
    """
    将 tokenizers 库的字节级 BPE 模型转换为 C++ CharBPETokenizer 兼容的 JSON 格式。

    字节级 BPE 的 token 全部是字节序列，无需区分"字符"和"合并"。
    所有多字节 token 一律作为合并 token 处理。
    """
    # ── 1. 获取原始词表和合并规则 ──
    vocab_str_to_id = tokenizer.get_vocab()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer.save(f.name)
        temp_path = f.name
    with open(temp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    Path(temp_path).unlink(missing_ok=True)

    model = data.get('model', {})
    merges_raw = model.get('merges', [])

    # ── 2. 解码 token 字符串为字节 ──
    # 字节级 BPE 使用 Ġ (U+0120) 表示前导空格等编码
    # tokenizers 的 ByteLevel 编码规则：
    #   字节 0x00-0xFF 映射到 Unicode U+0100-U+01FF (偏移 256)
    BYTE_OFFSET = 256

    def decode_byte_token(tok_str: str) -> bytes:
        """将 tokenizers 的字节级 token 字符串解码为原始字节。"""
        raw = bytearray()
        for ch in tok_str:
            cp = ord(ch)
            if BYTE_OFFSET <= cp < BYTE_OFFSET + 256:
                raw.append(cp - BYTE_OFFSET)
            else:
                # 不在映射范围的字符，按 UTF-8 编码（理论上不应出现）
                raw.extend(ch.encode('utf-8'))
        return bytes(raw)

    # 构建 token_str → bytes 映射
    token_bytes_map = {}
    for tok_str in vocab_str_to_id:
        token_bytes_map[tok_str] = decode_byte_token(tok_str)

    # ── 3. 解析合并规则 ──
    merge_pairs = []  # [(tok1_str, tok2_str)]
    for m in merges_raw:
        if isinstance(m, str):
            parts = m.split()
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))
        elif isinstance(m, list) and len(m) == 2:
            merge_pairs.append((m[0], m[1]))

    # 找出所有由合并产生的 token
    merged_set = set()
    for t1, t2 in merge_pairs:
        merged_set.add(t1 + t2)

    # ── 4. 分配我们的 ID ──
    special_set = set(SPECIAL_TOKENS)
    special_tok = {}    # token_str → 我们的 ID
    byte_tok = {}       # byte_value → 我们的 ID
    merge_tok = {}      # token_str → 我们的 ID（多字节 token）

    # 特殊 token: ID 0-3
    for i, st in enumerate(SPECIAL_TOKENS):
        special_tok[st] = i

    # 256 单字节兜底: ID 4-259
    for b in range(256):
        byte_tok[b] = BYTE_BASE + b

    # 按 tokenizers 内部 ID 排序，保持合并顺序
    sorted_tokens = sorted(vocab_str_to_id.items(), key=lambda x: x[1])

    next_id = CHAR_BASE
    for tok_str, _ in sorted_tokens:
        if tok_str in special_set:
            continue
        bs = token_bytes_map[tok_str]
        if len(bs) == 1:
            # 单字节已在 byte_tok 中
            continue
        # 所有多字节 token 都作为合并 token
        if next_id >= max_vocab:
            break
        merge_tok[tok_str] = next_id
        next_id += 1

    # ── 5. 构建 merges 三元组 ──
    our_merges = []

    def get_our_id(tok: str) -> int:
        """查找 token 在我们 ID 系统中的 ID。"""
        if tok in special_tok:
            return special_tok[tok]
        bs = token_bytes_map.get(tok, b'')
        if len(bs) == 1:
            return byte_tok.get(bs[0], -1)
        if tok in merge_tok:
            return merge_tok[tok]
        return -1

    for t1_str, t2_str in merge_pairs:
        merged_str = t1_str + t2_str
        if merged_str not in vocab_str_to_id:
            continue

        id_a = get_our_id(t1_str)
        id_b = get_our_id(t2_str)
        merged_id = get_our_id(merged_str)

        if id_a < 0 or id_b < 0 or merged_id < 0:
            continue

        our_merges.append([id_a, id_b, merged_id])

    # ── 6. 构建最终词表 ──
    vocab_hex = {}

    # 特殊 token (ID 0-3)
    for st, sid in special_tok.items():
        vocab_hex[str(sid)] = st

    # 单字节兜底 (ID 4-259)
    for b in range(256):
        vocab_hex[str(BYTE_BASE + b)] = f"{b:02x}"

    # 多字节 token (ID 260+)
    all_multi = {**merge_tok}
    for tok_str, our_id in all_multi.items():
        bs = token_bytes_map[tok_str]
        vocab_hex[str(our_id)] = bs.hex()

    # ── 7. 计算最终 vocab_size ──
    all_ids = [int(k) for k in vocab_hex.keys()]
    total_size = max(max(all_ids) + 1 if all_ids else CHAR_BASE, CHAR_BASE)

    return {
        "type": "char_bpe_tokenizer",  # 保持兼容，C++ 只认这个类型
        "vocab": vocab_hex,
        "vocab_size": total_size,
        "merges": our_merges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="训练字节级 BPE 分词器（GPT-2 风格），输出兼容 C++ CharBPETokenizer 的 JSON"
    )
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--vocab-size", type=int, default=8000, help="目标词表大小 (默认: 8000)")
    parser.add_argument("--min-freq", type=int, default=2, help="最小合并频率 (默认: 2)")
    parser.add_argument("--output", default="bytebpe.json", help="输出 JSON 路径")
    args = parser.parse_args()

    if not Path(args.text_file).exists():
        print(f"错误: 文件不存在: {args.text_file}", file=sys.stderr)
        sys.exit(1)

    min_size = CHAR_BASE
    if args.vocab_size < min_size:
        print(f"警告: vocab_size={args.vocab_size} 太小，自动调整为 {min_size}")
        args.vocab_size = min_size

    print(f"训练文本: {args.text_file}")
    print(f"目标词表: {args.vocab_size}, 最小频率: {args.min_freq}")
    print(f"模式: 字节级 BPE (byte-level BPE)")

    # 训练
    tokenizer = train_bytebpe(args.text_file, args.vocab_size, args.min_freq)

    # 转换格式
    print("转换为 CharBPE 兼容格式...")
    result = convert_to_charbpe_json(tokenizer, max_vocab=args.vocab_size * 3)

    # 保存
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    n_merges = len(result['merges'])
    n_total = len(result['vocab'])
    print(f"\n完成! 保存至: {output}")
    print(f"  vocab_size: {result['vocab_size']}")
    print(f"  合并规则: {n_merges}")
    print(f"  词表条目: {n_total}")

    # 验证：尝试编码一小段文本来确认正确性
    print("\n验证编码...")
    test_texts = [
        "Hello, world!",
        "人工智能是未来的方向",
        "The quick brown fox",
        "123 + 456 = 789",
    ]
    for text in test_texts:
        encoding = tokenizer.encode(text)
        ids = encoding.ids
        # 过滤掉特殊 token，用我们的 ID 系统重新映射
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        print(f"  '{text}' → {len(ids)} tokens, 解码: '{decoded}'")


if __name__ == "__main__":
    main()
