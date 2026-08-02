"""下载 HuggingFace FineWeb 数据集（英文）并转为项目预训练格式。

FineWeb 是 HuggingFaceFW 整理的英文网页预训练语料，本身就是英文
（创建时已通过语言过滤），适合作为 GPT base 模型的预训练数据。

项目训练格式（参考 src/text_train.cpp）：
  - 每行一个独立样本
  - 纯文本模式（无 <assistant> 等对话标记），所有位置参与 loss
  - 行内不能含换行符，否则会被当作多个样本
  - 训练时每行自动加 [BOS] 头 / [EOS] 尾

本脚本以 streaming 模式遍历 FineWeb，逐条抽取 text 字段，清洗后写为
单行，累计到目标字节数后停止。streaming 模式不会把整个数据集下载
到本地，只会按需拉取 parquet 分片。

用法:
  python scripts/download_fineweb.py
  python scripts/download_fineweb.py --target_mb 200
  python scripts/download_fineweb.py --subset sample-10BT --target_mb 50

依赖: pip install datasets
"""
import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# FineWeb 配置项：
#   sample-10BT  : 10B token 抽样集（推荐起点，体积小、覆盖广）
#   sample-100BT : 100B token 抽样集
#   CC-MAIN-*    : 单次 CC 抓取
DEFAULT_SUBSET = "sample-10BT"
DEFAULT_TARGET_MB = 100

# 单条样本长度上下限（字符数）
# 256-384 字符范围，适合 seq_len=256 的训练
MIN_DOC_LEN = 256
MAX_DOC_LEN = 384

# 触发进度打印的间隔（字节数）
PROGRESS_INTERVAL_BYTES = 5 * 1024 * 1024


# 简单的清洗：去 HTML 残留、合并空白、去掉换行（项目要求每行一个样本）
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")


def clean_doc(text: str) -> str:
    """清洗一个 FineWeb 文档：去 HTML 标签、合并空白、去换行。"""
    if not text:
        return ""
    # 移除 HTML 标签（FineWeb 已基本清理过，保险起见再过一遍）
    text = _HTML_TAG_RE.sub(" ", text)
    # 把所有空白（含换行）合并为单个空格
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def truncate_at_sentence(text: str, max_len: int) -> str:
    """在句子边界截断文档，避免从单词中间切断。
    
    优先在最后一个句号（.）处截断，保证语义完整性。
    如果找不到句号，回退到直接截断。
    """
    if len(text) <= max_len:
        return text
    
    # 在 max_len 范围内找最后一个句号
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    
    if last_period > 0:
        # 在句号处截断（保留句号）
        return text[:last_period + 1]
    else:
        # 找不到句号，回退到直接截断
        return truncated


def is_english_doc(text: str) -> bool:
    """简单判断文档主体是否为英文。

    FineWeb 在创建时已经按 language_score 过滤过英文样本，这里仅做
    兜底：要求 ASCII 字母占比 >= 60%，避免偶发的非英文混入。
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


def download(subset: str, target_bytes: int, out_path: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"FineWeb subset : {subset}")
    print(f"Target size    : {fmt_size(target_bytes)}")
    print(f"Output file    : {out_path}")
    print(f"Streaming ...  : 启动（首次拉取分片可能稍慢）")

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        subset,
        split="train",
        streaming=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    written = 0
    skipped_short = 0
    skipped_lang = 0
    bytes_written = 0
    last_progress = 0

    # newline="\n" 强制 LF，避免在 Windows 上写成 CRLF
    # （text_train.cpp 的 getline 已 trim \r，但保持 LF 更通用）
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i, sample in enumerate(ds):
            text = sample.get("text", "") or ""
            cleaned = clean_doc(text)

            # 长度过滤：过短的文档噪声多，过长的可能反序列化异常
            if len(cleaned) < MIN_DOC_LEN:
                skipped_short += 1
                continue
            if len(cleaned) > MAX_DOC_LEN:
                # 超长文档在句子边界截断，避免从单词中间切断
                cleaned = truncate_at_sentence(cleaned, MAX_DOC_LEN)
                # 截断后可能变得太短，再检查一次
                if len(cleaned) < MIN_DOC_LEN:
                    skipped_short += 1
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
                    f"skipped(short={skipped_short},lang={skipped_lang})"
                )
                last_progress = bytes_written

            # 达到目标大小就停
            if bytes_written >= target_bytes:
                break

    print()
    print(f"完成:")
    print(f"  写入文档数  : {written:,}")
    print(f"  跳过(过短)  : {skipped_short:,}")
    print(f"  跳过(非英文): {skipped_lang:,}")
    print(f"  实际大小    : {fmt_size(os.path.getsize(out_path))}")
    print(f"  输出文件    : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 FineWeb 英文预训练语料（项目格式：每行一个文档）"
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help=f"FineWeb 子集名 (默认: {DEFAULT_SUBSET})",
    )
    parser.add_argument(
        "--target_mb",
        type=int,
        default=DEFAULT_TARGET_MB,
        help=f"目标体积(MB), 达到即停止 (默认: {DEFAULT_TARGET_MB})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/fineweb_en_<N>mb.txt)",
    )
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            OUT_DIR, f"fineweb_en_{args.target_mb}mb.txt"
        )

    download(args.subset, target_bytes, out_path)


if __name__ == "__main__":
    main()
