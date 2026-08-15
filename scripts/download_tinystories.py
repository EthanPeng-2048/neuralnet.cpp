#!/usr/bin/env python3
"""下载 TinyStories（英文小模型故事数据集）并转为项目预训练格式。

TinyStories (roneneldan/TinyStories) 专为 10M 级小模型设计：
  - 语法规范、语义连贯、主题丰富的英文短篇故事
  - 每条 story 一个 text 字段（内部含段落换行）
  - 保留大小写，非常适合"续写文本、语法连贯、上下文相关"的演示

项目训练格式（src/text_train.cpp 滑动窗口）：
  - 每行 = 一个完整故事（故事内换行转为空格）
  - 纯文本模式（无对话标记），所有位置参与 loss
  - 训练端每行 [BOS]+tokens+[EOS] 后拼成连续 token 流，再按 seq_len 滑动切窗

用法:
  python scripts/download_tinystories.py
  python scripts/download_tinystories.py --target_mb 60 --max_stories 80000

依赖: pip install datasets
"""
import argparse
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

DEFAULT_TARGET_MB = 60          # 目标语料体积（足够 10M 模型快速演示）
MIN_LEN = 60                    # 过短故事丢弃（噪声）
MAX_LEN = 4000                  # 过长故事截断（防止单行异常长）
PROGRESS_INTERVAL = 5 * 1024 * 1024   # 每 5MB 打印一次进度

_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """故事内换行/制表符 → 空格，压缩连续空白，去首尾空白。"""
    text = text.replace("\r", " ")
    return _WS_RE.sub(" ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_mb", type=int, default=DEFAULT_TARGET_MB,
                    help="目标语料体积 (MB)，达到后停止")
    ap.add_argument("--max_stories", type=int, default=0,
                    help="最大故事数（0 = 不限，按体积停止）")
    ap.add_argument("--output", default=None, help="输出文件路径")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.output or os.path.join(OUT_DIR, "tinystories.txt")
    target_bytes = args.target_mb * 1024 * 1024

    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 需要 datasets 库: pip install datasets", file=sys.stderr)
        return 1

    # streaming 模式：不整包下载，按需拉取 parquet 分片，达到目标体积即停
    print(f"[加载] roneneldan/TinyStories (train, streaming) ...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    written = 0
    count = 0
    skipped = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        it = iter(ds)
        while True:
            if written >= target_bytes:
                break
            if args.max_stories and count >= args.max_stories:
                break

            # 网络不稳时单条拉取可能抛异常：容错重试
            try:
                item = next(it)
            except StopIteration:
                break
            except Exception as e:
                skipped += 1
                if skipped % 20 == 0:
                    print(f"  [重试] 拉取下一条失败 {e.__class__.__name__}: {e}")
                time.sleep(0.5)
                continue

            try:
                text = clean(item["text"])
            except Exception:
                skipped += 1
                continue

            if len(text) < MIN_LEN:
                skipped += 1
                continue
            if len(text) > MAX_LEN:
                text = text[:MAX_LEN].rstrip()

            f.write(text + "\n")
            written += len(text.encode("utf-8")) + 1
            count += 1

            if written // PROGRESS_INTERVAL != (written - len(text.encode("utf-8")) - 1) // PROGRESS_INTERVAL:
                el = time.time() - t0
                print(f"  [进度] {written/1024/1024:.1f} MB / {count} 条故事 ({el:.0f}s)")

    el = time.time() - t0
    print(f"\n完成: {out_path}")
    print(f"  故事数: {count}   跳过: {skipped}")
    print(f"  体积: {written/1024/1024:.2f} MB   耗时: {el:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
