"""合并所有中文语料来源，去重，输出最终训练集。

输入来源：
  1. datasets/llm_corpus.txt        — 原始 LLM 生成语料
  2. datasets/qwen_corpus.txt       — Qwen v1 生成语料
  3. datasets/qwen_corpus_v2.txt    — Qwen v2 扩展语料
  4. datasets/chinese_datasets_corpus.txt — HuggingFace/新闻/精选语料
  5. datasets/charbpe_corpus.json    — charbpe 语料
  6. datasets/charbpe_llm.json       — charbpe LLM 语料
  7. datasets/charbpe_mini.json      — charbpe 小语料

输出：
  datasets/training_corpus_final.txt — 最终训练语料（每行一段/一篇，长度不限）

滑动窗口说明：
  训练端 (src/text_train.cpp) 会把每行当作一个文档，编码为
  [BOS]+tokens+[EOS] 后拼接成连续 token 流，再按 seq_len 滑动切窗。
  因此：
    - 每行应保留完整段落/文档，不要按句号切碎（切碎会丢失跨句上下文）
    - 超长文档无需截断，训练端会自动切成多个窗口
    - 仅过滤过短噪声（< --min-len）和极端异常超长（> --max-len）

用法：
  python scripts/merge_corpus.py [--min-len 8] [--max-len 300]
"""

import json
import os
import re
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")

# 所有语料源文件
CORPUS_FILES = [
    "llm_corpus.txt",
    "qwen_corpus.txt",
    "qwen_corpus_v2.txt",
    "chinese_datasets_corpus.txt",
    "chinese_wiki_corpus.txt",
    "expand_chinese_corpus.txt",
    "llm_corpus_expanded.txt",
]

# JSON 格式的语料源
JSON_CORPUS_FILES = [
    "charbpe_corpus.json",
    "charbpe_llm.json",
    "charbpe_mini.json",
]


def load_txt_corpus(path: str) -> list[str]:
    """加载 txt 格式语料（每行一段/一篇）。"""
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception as e:
        print(f"  ⚠ 读取 {path} 失败: {e}")
    return lines


def load_json_corpus(path: str) -> list[str]:
    """加载 JSON 格式语料。"""
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    lines.append(item.strip())
                elif isinstance(item, dict):
                    # 尝试多种字段名
                    for key in ["text", "content", "sentence", "corpus", "segment"]:
                        if key in item and isinstance(item[key], str):
                            lines.append(item[key].strip())
                            break
    except Exception as e:
        print(f"  ⚠ 读取 {path} 失败: {e}")
    return lines


def is_chinese_dominant(text: str) -> bool:
    """检查文本是否以中文为主。"""
    if not text:
        return False
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese >= len(text) * 0.3


def main():
    parser = argparse.ArgumentParser(description="合并所有中文语料来源")
    parser.add_argument("--min-len", type=int, default=8, help="最短段落长度（过短视为噪声丢弃）")
    parser.add_argument("--max-len", type=int, default=4096,
                        help="最长段落长度（防御性上限，超过视为异常丢弃；滑动窗口会自动切分长文）")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 中文语料合并工具")
    print("=" * 60)

    all_lines = []

    # 加载 txt 语料
    print("\n📄 加载 txt 语料文件:")
    for fname in CORPUS_FILES:
        fpath = os.path.join(DATASETS_DIR, fname)
        if os.path.exists(fpath):
            lines = load_txt_corpus(fpath)
            print(f"  ✓ {fname}: {len(lines)} 行")
            all_lines.extend(lines)
        else:
            print(f"  - {fname}: 不存在，跳过")

    # 加载 json 语料
    print("\n📄 加载 json 语料文件:")
    for fname in JSON_CORPUS_FILES:
        fpath = os.path.join(DATASETS_DIR, fname)
        if os.path.exists(fpath):
            lines = load_json_corpus(fpath)
            print(f"  ✓ {fname}: {len(lines)} 行")
            all_lines.extend(lines)
        else:
            print(f"  - {fname}: 不存在，跳过")

    print(f"\n📊 原始总量: {len(all_lines)} 行")

    # 过滤 + 去重（保留完整段落，长度过滤仅用于剔除噪声/异常）
    seen = set()
    filtered = []
    for line in all_lines:
        line = line.strip()
        if not line:
            continue

        # 长度过滤：过短视为噪声，过长视为异常（滑动窗口会自动切分长文）
        if len(line) < args.min_len:
            continue
        if len(line) > args.max_len:
            continue

        # 中文比例检查
        if not is_chinese_dominant(line):
            continue

        # 去重（基于前 40 字符）
        key = line[:40]
        if key in seen:
            continue
        seen.add(key)
        filtered.append(line)

    print(f"📊 去重+过滤后: {len(filtered)} 行")

    # 统计
    total_chars = sum(len(line) for line in filtered)
    avg_len = total_chars / len(filtered) if filtered else 0
    print(f"📊 总字符数: {total_chars:,}")
    print(f"📊 平均长度: {avg_len:.0f} 字符")

    # 写入最终文件
    out_path = os.path.join(DATASETS_DIR, "training_corpus_final.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in filtered:
            f.write(line + "\n")

    print(f"\n{'='*60}")
    print(f"✅ 最终训练语料: {out_path}")
    print(f"   总计 {len(filtered)} 句, {total_chars:,} 字符")
    print(f"{'='*60}")

    # 也更新 llm_corpus.txt（合并回去）
    update_path = os.path.join(DATASETS_DIR, "llm_corpus.txt")
    existing = set()
    if os.path.exists(update_path):
        with open(update_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(line[:40])

    new_count = 0
    with open(update_path, "a", encoding="utf-8") as f:
        for line in filtered:
            key = line[:40]
            if key not in existing:
                f.write(line + "\n")
                existing.add(key)
                new_count += 1

    print(f"\n📝 已追加 {new_count} 条新句子到 {update_path}")


if __name__ == "__main__":
    main()
