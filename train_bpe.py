"""
训练 BPE 分词器，并输出与 token_bench 兼容的 JSON 格式。
依赖：pip install tokenizers
用法：python train_bpe.py dataset.txt --vocab-size 1000 --output bpe.json
"""

import argparse
import json
import tempfile
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 特殊 token 定义（与 ByteZip 保持一致）
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BYTE_OFFSET = 4   # 仅用于输出时的 ID 偏移，但 BPE 自己的 ID 是连续分配的，不需要偏移


def train_bpe(text_file: str, vocab_size: int) -> Tokenizer:
    """训练 BPE 并返回 tokenizer 对象"""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train(files=[text_file], trainer=trainer)
    return tokenizer


def extract_vocab_hex(tokenizer: Tokenizer) -> dict:
    """
    从 tokenizer 提取词表，转换为 {id: hex_string} 格式。
    注意：BPE 的 ID 从 0 开始，我们直接使用这些 ID。
    """
    vocab_str_to_id = tokenizer.get_vocab()  # {token: id}
    # 按 ID 排序得到列表
    max_id = max(vocab_str_to_id.values())
    vocab = [b""] * (max_id + 1)
    for token_str, tid in vocab_str_to_id.items():
        # 将 token 字符串转回字节（ByteLevel 的 token 已经是可打印字符串，但需要正确编码）
        token_bytes = token_str.encode("utf-8")
        vocab[tid] = token_bytes
    # 构建 {id: hex}
    hex_dict = {}
    for tid, bs in enumerate(vocab):
        if bs:
            hex_dict[str(tid)] = bs.hex()
    return hex_dict, len(vocab)


def extract_merges(tokenizer: Tokenizer) -> list[list[int]]:
    """
    提取合并规则，转换为三元组 [id1, id2, new_id]。
    采用保存临时文件的方式以确保兼容性。
    """
    # 保存 tokenizer 到临时 JSON 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer.save(f.name)
        temp_path = f.name

    # 读取临时文件
    with open(temp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 merges 原始数据
    merges_raw = data.get('model', {}).get('merges', [])
    
    # 转换为统一的二元组列表 (token1, token2)
    merges = []
    for item in merges_raw:
        if isinstance(item, str):
            # 如果格式是 "Ġ t" 这样的字符串，按空格拆分为两个 token
            parts = item.split()
            if len(parts) == 2:
                merges.append(tuple(parts))
        elif isinstance(item, list) and len(item) == 2:
            # 如果格式是 ["Ġ", "t"] 这样的列表
            merges.append(tuple(item))
        else:
            # 未知格式，跳过
            continue

    # 删除临时文件
    Path(temp_path).unlink(missing_ok=True)

    # 构建 token 字符串到 ID 的映射
    vocab_str_to_id = tokenizer.get_vocab()
    result = []
    for t1, t2 in merges:
        # 合并后的 token 字符串是 t1 + t2（ByteLevel 下如此）
        merged_str = t1 + t2
        if merged_str in vocab_str_to_id:
            new_id = vocab_str_to_id[merged_str]
            id1 = vocab_str_to_id[t1]
            id2 = vocab_str_to_id[t2]
            result.append([id1, id2, new_id])
        else:
            # 如果合并后的 token 不在词表中（理论上不应发生），跳过
            continue
    return result


def save_compatible_json(tokenizer: Tokenizer, path: str):
    vocab_hex, vocab_size = extract_vocab_hex(tokenizer)
    merges = extract_merges(tokenizer)

    data = {
        "type": "bpe_tokenizer",   # 标识类型
        "vocab": vocab_hex,
        "vocab_size": vocab_size,
        "merges": merges,          # 三元组列表，顺序即优先级
        "special_tokens": {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)},
        "byte_offset": 0,          # BPE 不使用偏移
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"BPE 词表已保存至: {path} (vocab_size={vocab_size}, merges={len(merges)})")


def main():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器并输出兼容格式")
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--vocab-size", type=int, default=1000, help="目标词表大小")
    parser.add_argument("--output", default="bpe_compatible.json", help="输出 JSON 路径")
    args = parser.parse_args()

    print(f"读取文件: {args.text_file}")
    tokenizer = train_bpe(args.text_file, args.vocab_size)
    save_compatible_json(tokenizer, args.output)


if __name__ == "__main__":
    main()