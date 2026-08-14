"""下载 OpenWebText 数据集并转为项目预训练格式。

OpenWebText 是 Reddit 高赞链接抓取的网页文本，英文为主，适合作为
GPT base 模型的预训练数据。

项目训练格式（参考 src/text_train.cpp，滑动窗口）：
  - 每行 = 一个完整文档（长度不限），行内不能含换行符
  - 纯文本模式（无对话标记），所有位置参与 loss
  - 训练端把每行编码为 [BOS]+tokens+[EOS] 后拼成连续 token 流，
    再按 seq_len 滑动切窗——因此超长文档无需截断，会被自动切成
    多个窗口，短文档也会被窗口切分/拼接利用

本脚本以 streaming 模式遍历 OpenWebText，逐条抽取 text 字段，清洗后写为
单行，累计到目标字节数后停止。streaming 模式不会把整个数据集下载
到本地，只会按需拉取分片。

用法:
  python scripts/download_openwebtext.py
  python scripts/download_openwebtext.py --target_mb 200
  python scripts/download_openwebtext.py --min_len 256

依赖: pip install datasets
"""
import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# OpenWebText 配置
# 常用版本: "openwebtext" 或 "Skylion007/openwebtext"
DEFAULT_DATASET = "Skylion007/openwebtext"
DEFAULT_TARGET_MB = 100

# 最短文档长度（字符数）：过短视为噪声丢弃
# 超长文档不设截断——训练端滑动窗口会自动切分；max_len 仅作防御性上限
DEFAULT_MIN_LEN = 256
DEFAULT_MAX_LEN = 100000

# 触发进度打印的间隔（字节数）
PROGRESS_INTERVAL_BYTES = 5 * 1024 * 1024


# 简单的清洗：去 HTML 残留、合并空白、去掉换行（项目要求每行一个文档）
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")


def clean_doc(text: str) -> str:
    """清洗一个 OpenWebText 文档：去 HTML 标签、合并空白、去换行。"""
    if not text:
        return ""
    # 移除 HTML 标签
    text = _HTML_TAG_RE.sub(" ", text)
    # 把所有空白（含换行）合并为单个空格
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def is_english_doc(text: str) -> bool:
    """简单判断文档主体是否为英文。

    OpenWebText 主要是英文，这里做兜底检查：要求 ASCII 字母占比 >= 60%。
    """
    if not text:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return False
    ascii_letters = sum(
        1 for c in text if ("a" <= c.lower() <= "z")
    )
    return ascii_letters / letters >= 0.6


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(dataset_name: str, target_bytes: int, min_len: int, max_len: int, out_path: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset        : {dataset_name}")
    print(f"Target size    : {fmt_size(target_bytes)}")
    print(f"Length filter  : {min_len} - {max_len} chars")
    print(f"Output file    : {out_path}")
    print(f"Streaming ...  : 启动（首次拉取分片可能稍慢）")

    # 尝试加载数据集
    # OpenWebText 通常只有 train split
    try:
        ds = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"加载数据集失败: {e}", file=sys.stderr)
        print("尝试使用 'openwebtext' 作为备选...", file=sys.stderr)
        try:
            ds = load_dataset(
                "openwebtext",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            print(f"备选数据集也失败: {e2}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    written = 0
    skipped_short = 0
    skipped_long = 0
    skipped_lang = 0
    bytes_written = 0
    last_progress = 0

    # newline="\n" 强制 LF，避免在 Windows 上写成 CRLF
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i, sample in enumerate(ds):
            text = sample.get("text", "") or ""
            cleaned = clean_doc(text)

            # 长度过滤：过短视为噪声；超长文档保留整篇（训练端滑动窗口切分）
            if len(cleaned) < min_len:
                skipped_short += 1
                continue
            if len(cleaned) > max_len:
                # 仅防御极端异常（默认上限很大，正常文档不会触发）
                skipped_long += 1
                continue

            # 语言过滤：仅保留英文
            if not is_english_doc(cleaned):
                skipped_lang += 1
                continue

            line = cleaned + "\n"
            f.write(line)
            written += 1
            bytes_written += len(line.encode("utf-8"))

            # 进度打印
            if bytes_written - last_progress >= PROGRESS_INTERVAL_BYTES:
                print(
                    f"  docs={written:,}  "
                    f"size={fmt_size(bytes_written)}  "
                    f"skipped(short={skipped_short},long={skipped_long},lang={skipped_lang})"
                )
                last_progress = bytes_written

            # 达到目标大小就停
            if bytes_written >= target_bytes:
                break

    print()
    print(f"完成:")
    print(f"  写入文档数  : {written:,}")
    print(f"  跳过(过短)  : {skipped_short:,}")
    print(f"  跳过(过长)  : {skipped_long:,}")
    print(f"  跳过(非英文): {skipped_lang:,}")
    print(f"  实际大小    : {fmt_size(os.path.getsize(out_path))}")
    print(f"  输出文件    : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 OpenWebText 预训练语料（项目格式：每行一个文档）"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"数据集名称 (默认: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--target_mb",
        type=int,
        default=DEFAULT_TARGET_MB,
        help=f"目标体积(MB), 达到即停止 (默认: {DEFAULT_TARGET_MB})",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=DEFAULT_MIN_LEN,
        help=f"最小字符数 (默认: {DEFAULT_MIN_LEN})",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"最大字符数 (默认: {DEFAULT_MAX_LEN})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/openwebtext_en_<N>mb.txt)",
    )
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            OUT_DIR, f"openwebtext_en_{args.target_mb}mb.txt"
        )

    download(args.dataset, target_bytes, args.min_len, args.max_len, out_path)


if __name__ == "__main__":
    main()
