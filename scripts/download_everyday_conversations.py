#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 everyday-conversations 对话数据集并转为 SFT 训练格式。

数据集: HuggingFaceTB/everyday-conversations-llama3.1-2k
  - 2.2k 条由 Llama-3.1-70B-Instruct 生成的多轮对话（日常话题 + 初等科学）
  - messages 列: [{role, content}, ...]，role ∈ {system, user, assistant}

转换格式（每行 = 一篇对话文档，使用新保留标记，与 tokenizer 词表一致）:
  <|system|>...</|end_of_system|><|user|>...</|end_of_user|><|assistant|>...</|end_of_assistant|>

输出:
  datasets/everyday_conversations_train_sft.txt
  datasets/everyday_conversations_test_sft.txt

用法:
  python scripts/download_everyday_conversations.py
依赖: pip install datasets
"""
import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")
_WS_RE = re.compile(r"\s+")


def render_content(content) -> str:
    """content 可能是 str，也可能是 [{type,text}] 结构，统一展平为字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("text", "")))
            else:
                parts.append(str(c))
        return " ".join(parts)
    return str(content)


def render_conversation(messages) -> str:
    """把一条对话（messages 列表）渲染成带保留标记的单行文本。"""
    parts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = render_content(m.get("content", ""))
        if role == "system":
            parts.append(f"<|system|>{content}<|end_of_system|>")
        elif role == "user":
            parts.append(f"<|user|>{content}<|end_of_user|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|end_of_assistant|>")
        else:
            parts.append(content)
    line = " ".join(parts).replace("\r", " ")
    return _WS_RE.sub(" ", line).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=OUT_DIR, help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 需要 datasets 库: pip install datasets", file=sys.stderr)
        return 1

    print("[i] 下载 everyday-conversations-llama3.1-2k ...")
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")

    for split in ("train_sft", "test_sft"):
        rows = ds[split]
        out_path = os.path.join(args.output_dir, f"everyday_conversations_{split}.txt")
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                line = render_conversation(row.get("messages"))
                if line:
                    f.write(line + "\n")
                    n += 1
        print(f"[+] {split}: {n} 条对话 → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
