#!/usr/bin/env python3
"""
将 GPT_teacher-3.37M-cn 数据集转换为 neuralnet.cpp 对话格式。

注意：训练端 (src/text_train.cpp) 当前为纯文本滑动窗口模式，
已移除对话 loss mask 训练。本脚本产出的 <user>/<assistant> 对话
格式数据暂时不用于训练，仅保留转换工具备用。

源数据格式 (JSONL):
  {"prompt": "...", "completion": "..."}
  {"prompt": "...", "completion": "..."}

目标格式 (每行一段对话):
  <user>prompt</user><assistant>completion</assistant>
"""

import json
import os
import sys
import urllib.request

REPO_BASE = "https://raw.githubusercontent.com/helloworldtang/GPT_teacher-3.37M-cn/main/data"
FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def download_file(url: str, timeout: float = 60.0) -> str:
    """下载文件内容并返回字符串（带超时，避免网络异常时无限阻塞）。"""
    print(f"  下载: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def convert_jsonl_to_dialogue(jsonl_text: str) -> list[str]:
    """将 JSONL 文本转换为 <user>...</user><assistant>...</assistant> 格式的行列表。"""
    lines = []
    for raw_line in jsonl_text.strip().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as e:
            print(f"  跳过无效 JSON 行: {e}")
            continue
        prompt = obj.get("prompt", "").strip()
        completion = obj.get("completion", "").strip()
        if not prompt or not completion:
            continue
        lines.append(f"<user>{prompt}</user><assistant>{completion}</assistant>")
    return lines


def main():
    # 输出目录
    out_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    for split, filename in FILES.items():
        url = f"{REPO_BASE}/{filename}"
        try:
            jsonl_text = download_file(url)
        except Exception as e:
            print(f"  下载失败: {e}")
            sys.exit(1)

        dialogue_lines = convert_jsonl_to_dialogue(jsonl_text)
        out_path = os.path.join(out_dir, f"gpt_teacher_{split}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in dialogue_lines:
                f.write(line + "\n")
        print(f"  保存: {out_path}  ({len(dialogue_lines)} 条)")

    print("\n转换完成！使用示例:")
    print(f"  text_train.exe datasets/gpt_teacher_train.txt"
          f" --test-file datasets/gpt_teacher_test.txt"
          f" --vocab wordzip.json --epochs 20 --gpu")


if __name__ == "__main__":
    main()
