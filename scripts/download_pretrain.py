"""下载 HuggingFace 英文预训练语料并转为项目预训练格式（统一入口）。

合并原 download_openwebtext.py / download_fineweb.py / download_fineweb_edu.py
三个独立脚本：

  openwebtext  : Reddit 高赞链接抓取的网页文本（英文为主，Skylion007/openwebtext）
  fineweb      : HF 英文网页预训练语料（HuggingFaceFW/fineweb，
                 子集 sample-10BT / sample-100BT / CC-MAIN-*）
  fineweb_edu  : FineWeb 教育模型打分的教育类子语料（HuggingFaceFW/fineweb-edu，
                 每条带 edu_score 0-5 分，整个数据集全英文，
                 子集 sample-10BT / sample-100BT / sample-350BT / CC-MAIN-*）

项目训练格式（参考 src/text_train.cpp，滑动窗口）：
  - 每行 = 一个完整文档（长度不限），行内不能含换行符
  - 纯文本模式（无对话标记），所有位置参与 loss
  - 训练端把每行编码为 [BOS]+tokens+[EOS] 后拼成连续 token 流，
    再按 seq_len 滑动切窗——因此超长文档无需截断，会被自动切成
    多个窗口，短文档也会被窗口切分/拼接利用

本脚本以 streaming 模式遍历数据集，逐条抽取 text 字段，按长度 /
edu_score 过滤并清洗后写为单行，累计到目标字节数后停止；
--target_mb 0 表示不限体积，遍历整个子集。streaming 模式不会把
整个数据集下载到本地，只会按需拉取 parquet 分片。

用法:
  python scripts/download_pretrain.py fineweb_edu
  python scripts/download_pretrain.py fineweb --subset sample-100BT --target_mb 200
  python scripts/download_pretrain.py openwebtext --min_len 512 --max_len 50000
  python scripts/download_pretrain.py fineweb_edu --min_edu_score 3.5
  python scripts/download_pretrain.py fineweb_edu --min_edu_score 0   # 不做教育分数过滤
  python scripts/download_pretrain.py fineweb --target_mb 0           # 不限体积, 遍历整个子集

依赖: pip install datasets
"""
import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# 各数据集配置：
#   hf_id          : HuggingFace 数据集 ID
#   fallback_hf_id : 备选数据集 ID（无子集数据集加载失败时尝试）
#   default_subset : 默认子集（None 表示无子集）
#   english_filter : 是否做英文兜底过滤（全英文数据集关掉）
#   has_edu_score  : 是否带 edu_score（score 字段，0-5 分）
DATASETS = {
    "openwebtext": {
        "hf_id": "Skylion007/openwebtext",
        "fallback_hf_id": "openwebtext",
        "default_subset": None,
        "english_filter": True,
        "has_edu_score": False,
    },
    "fineweb": {
        "hf_id": "HuggingFaceFW/fineweb",
        "default_subset": "sample-10BT",
        "english_filter": True,
        "has_edu_score": False,
    },
    "fineweb_edu": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "default_subset": "sample-10BT",
        "english_filter": False,  # 全英文数据集（language=en），无需语言过滤
        "has_edu_score": True,
    },
}

# 默认目标体积（MB）；0 表示不限，遍历整个子集
DEFAULT_TARGET_MB = 100

# 文档长度限制（字符数）：过短视为噪声丢弃
# 超长文档不设截断——训练端滑动窗口会自动切分；max_len 仅作防御性上限
DEFAULT_MIN_LEN = 256
DEFAULT_MAX_LEN = 100000

# fineweb_edu 的 edu_score 默认阈值（0-5 分，论文推荐用法是 >= 3）；0 关闭过滤
DEFAULT_MIN_EDU_SCORE = 3.0

# 触发进度打印的间隔（字节数）
PROGRESS_INTERVAL_BYTES = 5 * 1024 * 1024

# 简单的清洗：去 HTML 残留、合并空白、去掉换行（项目要求每行一个文档）
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")


def clean_doc(text: str) -> str:
    """清洗一个文档：去 HTML 标签、合并空白、去换行。"""
    if not text:
        return ""
    # 移除 HTML 标签（各数据集已基本清理过，保险起见再过一遍）
    text = _HTML_TAG_RE.sub(" ", text)
    # 把所有空白（含换行）合并为单个空格
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def is_english_doc(text: str) -> bool:
    """简单判断文档主体是否为英文。

    仅作"英文为主"数据集（openwebtext / fineweb）的兜底：要求 ASCII
    字母占比 >= 60%，避免偶发的非英文混入；fineweb_edu 全英文，不用。
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


def load_streaming(dataset_key: str, subset) -> object:
    """以 streaming 模式加载数据集；无子集的数据集带备选 ID 回退。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets", file=sys.stderr)
        sys.exit(1)

    cfg = DATASETS[dataset_key]

    if subset is not None:
        return load_dataset(cfg["hf_id"], subset, split="train", streaming=True)

    # 无子集数据集（openwebtext）：先试官方 ID，失败再试备选 ID
    try:
        return load_dataset(cfg["hf_id"], split="train", streaming=True)
    except Exception as e:
        print(f"加载 {cfg['hf_id']} 失败: {e}", file=sys.stderr)
    print("尝试使用备选 ID 'openwebtext'...", file=sys.stderr)
    try:
        return load_dataset(cfg["fallback_hf_id"], split="train", streaming=True)
    except Exception as e2:
        print(f"备选数据集也失败: {e2}", file=sys.stderr)
        sys.exit(1)


def download(
    dataset_key: str,
    subset,
    target_bytes: int,
    min_edu_score: float,
    min_len: int,
    max_len: int,
    out_path: str,
) -> None:
    cfg = DATASETS[dataset_key]
    unlimited = target_bytes == 0

    print(f"Dataset        : {cfg['hf_id']}" + (f"  subset: {subset}" if subset else ""))
    if cfg["has_edu_score"]:
        print(f"edu_score 过滤 : >= {min_edu_score}" if min_edu_score > 0 else "edu_score 过滤 : 关闭")
    print(f"Length filter  : {min_len} - {max_len} chars")
    print(f"Target size    : {'不限（整个子集）' if unlimited else fmt_size(target_bytes)}")
    print(f"Output file    : {out_path}")
    print(f"Streaming ...  : 启动（首次拉取分片可能稍慢）")

    ds = load_streaming(dataset_key, subset)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    written = 0
    skipped_short = 0
    skipped_long = 0
    skipped_lang = 0
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

            # 长度过滤：过短视为噪声；过长仅防御极端异常
            if len(cleaned) < min_len:
                skipped_short += 1
                continue
            if len(cleaned) > max_len:
                skipped_long += 1
                continue

            # 语言过滤：仅"英文为主"的数据集做兜底
            if cfg["english_filter"] and not is_english_doc(cleaned):
                skipped_lang += 1
                continue

            # 教育分数过滤：score 字段为 edu_score（0-5）
            edu_score = sample.get("score") if cfg["has_edu_score"] else None
            if cfg["has_edu_score"] and min_edu_score > 0:
                if edu_score is None or edu_score < min_edu_score:
                    skipped_low_score += 1
                    continue

            line = cleaned + "\n"
            f.write(line)
            written += 1
            if edu_score is not None:
                score_sum += edu_score
            bytes_written += len(line.encode("utf-8"))

            # 进度打印
            if bytes_written - last_progress >= PROGRESS_INTERVAL_BYTES:
                skips = f"skipped(short={skipped_short},long={skipped_long}"
                if cfg["english_filter"]:
                    skips += f",lang={skipped_lang}"
                if cfg["has_edu_score"]:
                    skips += f",low_score={skipped_low_score}"
                skips += ")"
                extra = ""
                if cfg["has_edu_score"] and written:
                    extra = f"  avg_edu={score_sum / written:.2f}"
                print(
                    f"  docs={written:,}  "
                    f"size={fmt_size(bytes_written)}  "
                    f"{skips}{extra}"
                )
                last_progress = bytes_written

            # 达到目标大小就停（不限体积则遍历到数据集末尾）
            if not unlimited and bytes_written >= target_bytes:
                break

    print()
    print(f"完成:")
    print(f"  写入文档数    : {written:,}")
    print(f"  跳过(过短)    : {skipped_short:,}")
    print(f"  跳过(过长)    : {skipped_long:,}")
    if cfg["english_filter"]:
        print(f"  跳过(非英文)  : {skipped_lang:,}")
    if cfg["has_edu_score"]:
        print(f"  跳过(分数过低) : {skipped_low_score:,}")
        if written:
            print(f"  平均 edu_score: {score_sum / written:.2f}")
    print(f"  实际大小      : {fmt_size(os.path.getsize(out_path))}")
    print(f"  输出文件      : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 英文预训练语料（项目格式：每行一个文档）"
    )
    parser.add_argument(
        "dataset",
        choices=list(DATASETS),
        help="数据集: openwebtext / fineweb / fineweb_edu",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="子集名 (fineweb / fineweb_edu 默认: sample-10BT；openwebtext 无子集)",
    )
    parser.add_argument(
        "--target_mb",
        type=int,
        default=DEFAULT_TARGET_MB,
        help=f"目标体积(MB), 达到即停止 (默认: {DEFAULT_TARGET_MB}; 0 = 不限, 遍历整个子集)",
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
        "--min_edu_score",
        type=float,
        default=None,
        help=f"fineweb_edu 的 edu_score 下限 0-5 (默认: {DEFAULT_MIN_EDU_SCORE}; "
        "设 0 关闭过滤; 其他数据集无此字段)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/<dataset>_en_<N>mb.txt, 不限时为 _full.txt)",
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    # 子集：无子集数据集忽略 --subset；有子集的默认 sample-10BT
    if cfg["default_subset"] is None:
        if args.subset is not None:
            print(f"警告: {args.dataset} 无子集, 忽略 --subset", file=sys.stderr)
        subset = None
    else:
        subset = args.subset if args.subset is not None else cfg["default_subset"]

    # edu_score 阈值：fineweb_edu 默认 3.0，其他数据集默认关闭
    if args.min_edu_score is None:
        min_edu_score = DEFAULT_MIN_EDU_SCORE if cfg["has_edu_score"] else 0.0
    elif not cfg["has_edu_score"]:
        print(f"警告: {args.dataset} 无 edu_score 字段, 忽略 --min_edu_score", file=sys.stderr)
        min_edu_score = 0.0
    else:
        min_edu_score = args.min_edu_score

    target_bytes = args.target_mb * 1024 * 1024  # 0 = 不限

    if args.output:
        out_path = args.output
    else:
        suffix = f"{args.target_mb}mb" if target_bytes > 0 else "full"
        out_path = os.path.join(OUT_DIR, f"{args.dataset}_en_{suffix}.txt")

    download(
        args.dataset,
        subset,
        target_bytes,
        min_edu_score,
        args.min_len,
        args.max_len,
        out_path,
    )


if __name__ == "__main__":
    main()
