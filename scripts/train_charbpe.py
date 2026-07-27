"""
使用 HuggingFace tokenizers 库训练字符级 BPE 分词器，速度远超 C++ 实现。

输出格式与 C++ CharBPETokenizer 完全兼容的 JSON 文件。
ID 布局：
  0-3    : <pad>, <unk>, <bos>, <eos>
  4-259  : 256 个单字节 ASCII 兜底
  260+   : 非 ASCII 字符（中文等）+ BPE 合并 token

依赖：pip install tokenizers
用法：python train_charbpe.py <text_file> [选项]
示例：python train_charbpe.py datasets/llm_corpus.txt --vocab-size 5000 --output charbpe.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.normalizers import Normalizer


# ── 常量（与 C++ CharBPETokenizer 一致） ──────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BYTE_BASE = 4       # 256 个单字节兜底起始 ID
CHAR_BASE = 260     # 非 ASCII 字符 token 起始 ID


def make_pretokenizer():
    """
    构造字符级预分词器：
    - 连续 CJK 字符作为一个词（让 BPE 学习常见词组合并）
    - CJK 标点作为分隔符
    - 连续 ASCII 非空白字符作为一个词
    - 空白作为独立 token
    """
    # 规则：按优先级匹配
    # 1. 连续 CJK 字符（不含标点）→ 一个词
    # 2. 单个 CJK 标点 → 独立
    # 3. 连续非空白非CJK → 一个词（ASCII 单词、数字等）
    # 4. 连续空白 → 一个 token
    pattern = (
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+"    # 连续 CJK 字符
        r"|[\u3000-\u303f\uff00-\uffef]"                   # 单个 CJK 标点
        r"|[^ \t\n\r\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+"  # 连续非空白非CJK
        r"|[ \t\n\r]+"                                      # 空白
    )
    return Split(Regex(pattern), behavior="isolated", invert=False)


def make_normalizer():
    """不做任何 normalization，保留原始文本。"""
    return None  # 不用 normalizer，避免 NFC/NFKC 改变中文


def train_charbpe(text_file: str, vocab_size: int, min_freq: int) -> Tokenizer:
    """训练字符级 BPE tokenizer。"""
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=False))
    tokenizer.normalizer = make_normalizer()
    tokenizer.pre_tokenizer = make_pretokenizer()
    # 禁用 post_processor 和 decoder，避免自动插入特殊符号
    tokenizer.post_processor = None
    tokenizer.decoder = None

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=[],   # 不自动添加字母表，由预分词器控制
    )
    tokenizer.train(files=[text_file], trainer=trainer)
    return tokenizer


def convert_to_charbpe_json(tokenizer: Tokenizer, max_vocab: int = 50000) -> dict:
    """
    将 tokenizers 库的 BPE 模型转换为 C++ CharBPETokenizer 的 JSON 格式。

    关键转换：
    1. tokenizers 库使用 Ġ 表示前导空格 → 替换为实际空格字节
    2. 重新分配 ID：特殊 token → 字节兜底 → 字符 → 合并
    3. merges 转换为 [id_a, id_b, new_id] 三元组
    """
    # ── 1. 获取原始词表和合并规则 ──
    vocab_str_to_id = tokenizer.get_vocab()  # {token_string: id}

    # 读取临时 JSON 获取 merges（tokenizers 库不直接暴露）
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer.save(f.name)
        temp_path = f.name
    with open(temp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    Path(temp_path).unlink(missing_ok=True)

    model = data.get('model', {})
    merges_raw = model.get('merges', [])
    added_tokens = data.get('added_tokens', [])

    # ── 2. 将 token 字符串转为字节 ──
    # tokenizers 库的 BPE token 是字符串，需要正确解码
    # Ġ 是 ByteLevel pre-tokenizer 的编码，我们没用它，
    # 但合并规则可能包含它，需要处理
    def token_to_bytes(tok: str) -> bytes:
        """将 token 字符串转为原始字节。处理 Ġ → 空格。"""
        # 替换 Ġ (U+0120) 为空格
        tok = tok.replace('\u0120', ' ')
        return tok.encode('utf-8')

    # 构建 token_str → bytes 映射
    token_bytes_map = {}
    for tok_str in vocab_str_to_id:
        token_bytes_map[tok_str] = token_to_bytes(tok_str)

    # ── 3. 分类 token ──
    special_set = set(SPECIAL_TOKENS)
    special_tok = {}    # token_str → 我们的 ID
    byte_tok = {}       # byte_value → 我们的 ID
    char_tok = {}       # token_str → 我们的 ID（非 ASCII 字符）
    merge_tok = {}      # token_str → 我们的 ID（BPE 合并结果）

    # 特殊 token 分配 ID 0-3
    for i, st in enumerate(SPECIAL_TOKENS):
        special_tok[st] = i

    # 256 单字节兜底分配 ID 4-259
    for b in range(256):
        byte_tok[b] = BYTE_BASE + b

    # 分类剩余 token
    next_id = CHAR_BASE
    char_list = []   # (token_str, bytes) 非 ASCII 字符
    merge_list = []  # (token_str, bytes, id_a, id_b) 合并 token

    # 解析 merges 获取合并关系
    merge_pairs = []  # [(tok1_str, tok2_str)]
    for m in merges_raw:
        if isinstance(m, str):
            parts = m.split()
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))
        elif isinstance(m, list) and len(m) == 2:
            merge_pairs.append((m[0], m[1]))

    # 找出哪些 token 是合并产生的（出现在 merges 的右侧）
    merged_set = set()
    for t1, t2 in merge_pairs:
        merged_set.add(t1 + t2)

    # 分类
    for tok_str, tid in vocab_str_to_id.items():
        if tok_str in special_set:
            continue
        bs = token_bytes_map[tok_str]
        if len(bs) == 1:
            # 单字节 → 已在 byte_tok 中
            continue
        if tok_str in merged_set:
            # 合并 token，稍后处理
            continue
        # 非 ASCII 字符（多字节但不是合并结果）
        if not all(0x20 <= b < 0x7f for b in bs):
            char_list.append((tok_str, bs))

    # 按首次出现顺序排序字符
    char_list.sort(key=lambda x: vocab_str_to_id[x[0]])

    # 分配字符 ID
    for tok_str, bs in char_list:
        if next_id >= max_vocab:
            break
        char_tok[tok_str] = next_id
        next_id += 1

    # ── 4. 构建 merges 三元组 ──
    our_merges = []
    for t1_str, t2_str in merge_pairs:
        merged_str = t1_str + t2_str
        if merged_str not in vocab_str_to_id:
            continue

        # 查找 t1, t2 在我们 ID 系统中的 ID
        def get_our_id(tok: str) -> int:
            if tok in special_tok:
                return special_tok[tok]
            bs = token_bytes_map[tok]
            if len(bs) == 1:
                return byte_tok[bs[0]]
            if tok in char_tok:
                return char_tok[tok]
            if tok in merge_tok:
                return merge_tok[tok]
            # 如果找不到，分配一个新 ID
            nonlocal next_id
            if next_id < max_vocab:
                merge_tok[tok] = next_id
                next_id += 1
                return merge_tok[tok]
            return -1

        id_a = get_our_id(t1_str)
        id_b = get_our_id(t2_str)
        if id_a < 0 or id_b < 0:
            continue

        # 合并结果的 ID
        merged_id = get_our_id(merged_str)
        if merged_id < 0:
            continue

        our_merges.append([id_a, id_b, merged_id])

    # ── 5. 构建最终词表 ──
    vocab_hex = {}

    # 特殊 token (ID 0-3)：纯文本
    for st, sid in special_tok.items():
        vocab_hex[str(sid)] = st

    # 单字节兜底 (ID 4-259)：hex 编码
    for b in range(256):
        vocab_hex[str(BYTE_BASE + b)] = f"{b:02x}"

    # 非 ASCII 字符：hex 编码 UTF-8
    all_chars = {**char_tok}
    for tok_str, our_id in all_chars.items():
        bs = token_bytes_map[tok_str]
        vocab_hex[str(our_id)] = bs.hex()

    # 合并 token：hex 编码
    all_merges = {**merge_tok}
    for tok_str, our_id in all_merges.items():
        bs = token_bytes_map[tok_str]
        vocab_hex[str(our_id)] = bs.hex()

    # ── 6. 计算最终 vocab_size ──
    all_ids = [int(k) for k in vocab_hex.keys()]
    total_size = max(max(all_ids) + 1 if all_ids else CHAR_BASE, CHAR_BASE)

    return {
        "type": "char_bpe_tokenizer",
        "vocab": vocab_hex,
        "vocab_size": total_size,
        "merges": our_merges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="使用 tokenizers 库训练字符级 BPE，输出兼容 C++ 格式的 JSON"
    )
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--vocab-size", type=int, default=5000, help="目标词表大小 (默认: 5000)")
    parser.add_argument("--min-freq", type=int, default=2, help="最小合并频率 (默认: 2)")
    parser.add_argument("--output", default="charbpe_fast.json", help="输出 JSON 路径")
    args = parser.parse_args()

    if not Path(args.text_file).exists():
        print(f"错误: 文件不存在: {args.text_file}", file=sys.stderr)
        sys.exit(1)

    # 限制 vocab_size 范围
    min_size = CHAR_BASE  # 至少要能容纳特殊 token + 字节兜底 + 一些字符
    if args.vocab_size < min_size:
        print(f"警告: vocab_size={args.vocab_size} 太小，自动调整为 {min_size}")
        args.vocab_size = min_size

    print(f"训练文本: {args.text_file}")
    print(f"目标词表: {args.vocab_size}, 最小频率: {args.min_freq}")

    # 训练
    tokenizer = train_charbpe(args.text_file, args.vocab_size, args.min_freq)

    # 转换格式
    print("转换为 CharBPE 兼容格式...")
    result = convert_to_charbpe_json(tokenizer, max_vocab=args.vocab_size * 3)

    # 保存
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    n_chars = sum(1 for k in result['vocab'] if int(k) >= CHAR_BASE and int(k) < CHAR_BASE + 10000)
    n_merges = len(result['merges'])
    print(f"\n完成! 保存至: {output}")
    print(f"  vocab_size: {result['vocab_size']}")
    print(f"  字符 token: ~{n_chars}")
    print(f"  合并规则: {n_merges}")


if __name__ == "__main__":
    main()
