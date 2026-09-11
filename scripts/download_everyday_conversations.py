#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 everyday-conversations 对话数据集并转为 SFT 训练格式。

数据集: HuggingFaceTB/everyday-conversations-llama3.1-2k
  - 2.2k 条由 Llama-3.1-70B-Instruct 生成的多轮对话（日常话题 + 初等科学）
  - messages 列: [{role, content}, ...]，role ∈ {system, user, assistant}

转换格式（每行 = 一篇对话文档，使用新保留标记，与 tokenizer 词表一致）:
  <|system|>...</|end_of_system|><|user|>...</|end_of_user|><|assistant|>...</|end_of_assistant|>

语言过滤: 仅保留主体为英文的对话（ASCII 字母占比 >= 60%，
与 download_fineweb.py / download_openwebtext.py 同一兜底启发式），
中文/日文/韩文/阿拉伯文/西里尔文等非英文对话直接跳过。

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
from tqdm import tqdm   # 新增导入

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")
_WS_RE = re.compile(r"\s+")
_SFT_TAG_RE = re.compile(r"<\|[^|]*\|>")


def is_english_text(text: str) -> bool:
    """简单判断文本主体是否为英文（与 download_fineweb.py 同一兜底启发式）。

    要求 ASCII 字母占比 >= 60%，可排除 CJK/阿拉伯/西里尔等非英文对话；
    注意这是兜底而非严格检测器，法/德/西等拉丁字母语言无法区分。
    """
    if not text:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return False
    ascii_letters = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    return ascii_letters / letters >= 0.6


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
    ap.add_argument("--max-size-mb", type=float, default=None, help="限制输出文件大小（MB），达到后停止下载")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 需要 datasets 库: pip install datasets", file=sys.stderr)
        return 1

    max_bytes = int(args.max_size_mb * 1024 * 1024) if args.max_size_mb else None

    print("[i] 开始流式下载 Nemotron-Cascade-2-SFT-Data ...")
    ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "chat", streaming=True)

    for split in ("train",):
        out_path = os.path.join(args.output_dir, f"everyday_conversations_{split}_sft.txt")
        current_size = 0
        n = 0
        skipped_lang = 0

        # 创建进度条
        # 如果设置了大小上限，则 total 为 max_bytes，显示百分比；否则不设 total，只显示计数和速度
        pbar = tqdm(
            total=max_bytes if max_bytes else None,
            unit='B',
            unit_scale=True,
            desc=f"下载 {split}",
            ncols=100,
            # 若 total=None，则默认不显示百分比，只显示已处理条数（由手动更新控制）
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                line = render_conversation(row.get("messages"))
                if not line:
                    continue

                # 语言过滤：跳过非英文对话。先剔除 <|...|> 保留标记再检测，
                # 否则标记自身的 ASCII 字母会让很短的非英文对话误判为英文
                if not is_english_text(_SFT_TAG_RE.sub(" ", line)):
                    skipped_lang += 1
                    continue

                line_bytes = len((line + "\n").encode("utf-8"))

                # 检查是否超过大小限制
                if max_bytes and current_size + line_bytes > max_bytes:
                    pbar.update(max_bytes - current_size)  # 将进度条补满
                    pbar.set_description("已达到大小限制，停止下载")
                    break

                f.write(line + "\n")
                current_size += line_bytes
                n += 1

                # 更新进度条（增加已下载字节数）
                pbar.update(line_bytes)
                # 同时可以附加显示已处理的条数（后处理描述）
                pbar.set_postfix(条数=n)

        pbar.close()
        print(f"[+] {split}: 共写入 {n} 条对话（跳过非英文 {skipped_lang:,} 条），"
              f"文件大小约 {current_size / (1024*1024):.2f} MB → {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
