"""下载 HuggingFace FineWeb-Edu 数据集（全英文）并转为项目预训练格式。

FineWeb-Edu 是 HuggingFaceFW 在 FineWeb 基础上用教育模型打分得到的
教育类高质量语料：每条文档带 edu_score（0-5 分，越高教育内容越多）。
整个数据集全英文（language=en），无需再做语言过滤。

项目训练格式（参考 src/text_train.cpp，滑动窗口）：
  - 每行 = 一个完整文档（长度不限），行内不能含换行符
  - 纯文本模式（无对话标记），所有位置参与 loss
  - 训练端把每行编码为 [BOS]+tokens+[EOS] 后拼成连续 token 流，
    再按 seq_len 滑动切窗——因此超长文档无需截断，会被自动切成
    多个窗口，短文档也会被窗口切分/拼接利用

本脚本以 streaming 模式遍历 FineWeb-Edu，逐条抽取 text 字段，按
edu_score 与长度过滤后清洗写为单行，累计到目标字节数后停止；
--target_mb 0 表示不限体积，遍历整个子集。streaming 模式不会把
整个数据集下载到本地，只会按需拉取 parquet 分片。

用法:
  python scripts/download_fineweb_edu.py
  python scripts/download_fineweb_edu.py --target_mb 200
  python scripts/download_fineweb_edu.py --subset sample-100BT --min_edu_score 3.5
  python scripts/download_fineweb_edu.py --min_edu_score 0   # 不做教育分数过滤
  python scripts/download_fineweb_edu.py --min_len 512 --max_len 50000
  python scripts/download_fineweb_edu.py --target_mb 0       # 不限体积, 遍历整个子集

依赖: pip install datasets
"""
import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# FineWeb-Edu 配置项（子集）：
#   sample-10BT  : 10B token 抽样集（推荐起点，体积小、覆盖广）
#   sample-100BT : 100B token 抽样集
#   sample-350BT : 350B token 抽样集
#   CC-MAIN-*    : 单次 CC 抓取（2013-2025 各年份）
DEFAULT_SUBSET = "sample-10BT"
DEFAULT_TARGET_MB = 100

# edu_score 阈值（0-5 分，数据集中的 score 字段）。
# 论文推荐的默认用法是取 edu_score >= 3；设为 0 表示不过滤。
DEFAULT_MIN_EDU_SCORE = 3.0

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
    """清洗一个 FineWeb-Edu 文档：去 HTML 标签、合并空白、去换行。"""
    if not text:
        return ""
    # 移除 HTML 标签（FineWeb 系已基本清理过，保险起见再过一遍）
    text = _HTML_TAG_RE.sub(" ", text)
    # 把所有空白（含换行）合并为单个空格
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(
    subset: str,
    target_bytes: int,
    min_edu_score: float,
    min_len: int,
    max_len: int,
    out_path: str,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset        : HuggingFaceFW/fineweb-edu")
    print(f"Subset         : {subset}")
    print(f"edu_score 过滤 : >= {min_edu_score}" if min_edu_score > 0 else "edu_score 过滤 : 关闭")
    print(f"Length filter  : {min_len} - {max_len} chars")
    # target_bytes == 0 表示不限体积，遍历整个子集
    print(f"Target size    : {'不限（整个子集）' if target_bytes == 0 else fmt_size(target_bytes)}")
    print(f"Output file    : {out_path}")
    print(f"Streaming ...  : 启动（首次拉取分片可能稍慢）")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        subset,
        split="train",
        streaming=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    written = 0
    skipped_short = 0
    skipped_low_score = 0
    score_sum = 0.0
    bytes_written = 0
    last_progress = 0

    # newline="\n" 强制 LF，避免在 Windows 上写成 CRLF
    # （text_train.cpp 的 getline 已 trim \r，但保持 LF 更通用）
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for sample in ds:
            text = sample.get("text", "") or ""
            cleaned = clean_doc(text)

            # 长度过滤：过短视为噪声；超长文档保留整篇（训练端滑动窗口切分）
            if len(cleaned) < MIN_DOC_LEN:
                skipped_short += 1
                continue

            # 教育分数过滤：score 字段为 edu_score（0-5）
            edu_score = sample.get("score")
            if (
                min_edu_score > 0
                and (edu_score is None or edu_score < min_edu_score)
            ):
                skipped_low_score += 1
                continue

            line = cleaned + "\n"
            f.write(line)
            written += 1
            score_sum += edu_score if edu_score is not None else 0.0
            bytes_written += len(line.encode("utf-8"))

            # 进度打印
            if bytes_written - last_progress >= PROGRESS_INTERVAL_BYTES:
                avg = score_sum / written if written else 0.0
                print(
                    f"  docs={written:,}  "
                    f"size={fmt_size(bytes_written)}  "
                    f"skipped(short={skipped_short},low_score={skipped_low_score})  "
                    f"avg_edu={avg:.2f}"
                )
                last_progress = bytes_written

            # 达到目标大小就停
            if bytes_written >= target_bytes:
                break

    avg = score_sum / written if written else 0.0
    print()
    print(f"完成:")
    print(f"  写入文档数    : {written:,}")
    print(f"  跳过(过短)    : {skipped_short:,}")
    print(f"  跳过(分数过低): {skipped_low_score:,}")
    print(f"  平均 edu_score: {avg:.2f}")
    print(f"  实际大小      : {fmt_size(os.path.getsize(out_path))}")
    print(f"  输出文件      : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 FineWeb-Edu 教育类英文预训练语料（项目格式：每行一个文档）"
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help=f"子集名 (默认: {DEFAULT_SUBSET}；可选 sample-10BT / sample-100BT / sample-350BT / CC-MAIN-*)",
    )
    parser.add_argument(
        "--target_mb",
        type=int,
        default=DEFAULT_TARGET_MB,
        help=f"目标体积(MB), 达到即停止 (默认: {DEFAULT_TARGET_MB})",
    )
    parser.add_argument(
        "--min_edu_score",
        type=float,
        default=DEFAULT_MIN_EDU_SCORE,
        help=f"edu_score 下限 0-5 (默认: {DEFAULT_MIN_EDU_SCORE}；设 0 关闭过滤)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/fineweb_edu_en_<N>mb.txt)",
    )
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            OUT_DIR, f"fineweb_edu_en_{args.target_mb}mb.txt"
        )

    download(args.subset, target_bytes, args.min_edu_score, out_path)


if __name__ == "__main__":
    main()
