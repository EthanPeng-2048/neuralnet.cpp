"""
预处理脚本：分析训练文本，生成固定词表 JSON。

词表构成:
  ID 0-2:    <unk> <pad> <num>
  ID 3-130:  128 个 ASCII 字符
  ID 131+:   高频词（按词频降序）

用法:
  python train_vocab.py dataset.txt --vocab-size 10000
  python train_vocab.py dataset.txt --vocab-size 5000 --output my_vocab.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="生成固定词表 JSON")
    parser.add_argument("text_file", help="训练文本文件路径")
    parser.add_argument("--output", default="gpt_vocab.json", help="输出 JSON 路径")
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"文本: {len(text):,} 字符")

    # 统计词频（按空白分词）
    words = text.split()
    freq = Counter(words)
    print(f"不同词数: {len(freq):,}")

    # 构建词表
    # ID 布局: 0-2 特殊, 3+ 词(含单字母词), 词之后 ASCII补缺
    # 单字母词 "a","I" 等作为词 token，不与 ASCII 字符共享 ID
    vocab = {"<unk>": 0, "<pad>": 1, "<num>": 2}
    for word, _ in freq.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    ascii_start = len(vocab)  # ASCII 字符的起始 ID
    for i in range(128):
        if chr(i) not in vocab:
            vocab[chr(i)] = len(vocab)

    word_count = len(freq)
    ascii_count = len(vocab) - 3 - word_count
    print(f"词表大小: {len(vocab):,} (特殊3 + 词{word_count:,} + ASCII补缺{ascii_count})")
    print(f"ASCII 起始 ID: {ascii_start}")
    print(f"覆盖语料中 100% 的词，不会产生 <unk>")

    # 保存
    output = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "ascii_start": ascii_start,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
