"""下载 HuggingFace 数学预训练语料并转为项目预训练格式。

数据集（回退链：按序尝试，取第一个「streaming 可用且带 text 字段」的源，
全部失败则以非 0 退出）:

  1. HuggingFaceFW/OpenWebMath（split=train，正文字段 text）
  2. eleutherai/open-web-math（split=train，正文字段 text）
  3. open-web-math/open-web-math（split=train，631 万条，正文字段 text）
     ——2026-10 实测：前两个 ID 在 HF 均返回 401（不存在/不可访问），
     OpenWebMath 的现行官方仓库是 open-web-math/open-web-math，故追加为
     最终兜底，保证回退链实际可用；前两者保留（若日后恢复/改名则优先命中）。

转换格式（项目预训练格式，同 download_pretrain.py；参考 src/text_train.cpp
的滑动窗口训练端）:
  - 每行 = 一篇数学网页文档，行内不含换行符
  - 清洗：去控制字符 + 合并空白（含换行/Tab）为单空格——与
    download_pretrain.py 的 clean_doc 同构，但省去 HTML 标签剥离：
    OpenWebMath 已做过 boilerplate 清理，且数学正文里 < > 常见于比较式
    （a < b），download_pretrain 的 <[^>]+> 会把这类内容误删
  - 训练端把每行编码为 [BOS]+tokens+[EOS] 拼成连续 token 流后按 seq_len
    滑动切窗，超长文档会被自动切成多个窗口

过滤:
  - 长度过滤 --min_len / --max_len（行 = 输出行 = 一个文档的字符数，0 = 关闭）
  - 英文兜底：不套用 download_everyday_conversations.py 的 is_english_text
    （按 ASCII 字母占比、阈值 0.6 的口径），改用更宽松的 **ASCII 占比
    >= 50%**（--min_ascii_ratio，0 = 关闭）。理由：OpenWebMath 主体是英文
    数学网页（官方已用 fastText 语言识别过滤），但正文大量出现 LaTeX 与
    数学符号（× ≤ ∑、希腊字母等，均为非 ASCII），按「字母」口径或高阈值
    会误杀正常英文数学文本；而整段 CJK / 西里尔 / 阿拉伯文页面的 ASCII
    占比通常远低于 50%，用全字符 ASCII 占比 >= 50% 可以只剔除这类页面。
    与 is_english_text 同为启发式：法/德/西等拉丁字母语言无法区分（保留）。

用法:
  python scripts/download_math_dataset.py
  python scripts/download_math_dataset.py --target_mb 200
  python scripts/download_math_dataset.py --target_mb 0            # 不限体积, 遍历整个数据集
  python scripts/download_math_dataset.py --min_ascii_ratio 0      # 关闭 ASCII 占比过滤
  python scripts/download_math_dataset.py --min_len 0 --max_len 0  # 关闭行长过滤
  python scripts/download_math_dataset.py --output datasets/my_math.txt

依赖: pip install datasets（进度条: from tqdm import tqdm）
"""
import argparse
import os
import re
import sys

try:
    from tqdm import tqdm   # 进度条（与 download_everyday_conversations.py 一致）
except ImportError:
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# 回退链（按序尝试；每个源先试首选 split，再退到不指定 split）：
#   前两个 ID 2026-10 实测为 401（不存在），第三个是现行官方仓库（见文件头）
MATH_SOURCES = (
    {"hf_id": "HuggingFaceFW/OpenWebMath", "split": "train"},
    {"hf_id": "eleutherai/open-web-math", "split": "train"},
    {"hf_id": "open-web-math/open-web-math", "split": "train"},
)

# 正文字段（三源一致为 text；缺失即落到下一源）
TEXT_FIELD_CANDIDATES = ("text",)

# 默认目标体积（MB）；0 表示不限，遍历整个数据集
DEFAULT_TARGET_MB = 100

# 行长过滤（字符数）：过短视为噪声丢弃；超长不截断——训练端滑动窗口会
# 自动切分，max_len 仅作防御性上限；两项都可设 0 关闭
DEFAULT_MIN_LEN = 256
DEFAULT_MAX_LEN = 100000

# ASCII 占比下限（英文兜底，理由见文件头）；0 = 关闭过滤
DEFAULT_MIN_ASCII_RATIO = 0.5

# 清洗：去控制字符（\t \n \r \f \v 保留给下一步空白合并；NUL/ESC 等直接剔除）
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 合并所有空白（含换行/Tab）为单个空格——保证输出行内不含换行符
_WS_RE = re.compile(r"\s+")


def clean_doc(text: str) -> str:
    """清洗一个文档：去控制字符、合并空白（含换行）为单空格。

    与 download_pretrain.py 的 clean_doc 同构，省去 HTML 标签剥离
    （数学比较式 a < b 会被 <[^>]+> 误删，理由见文件头）。
    """
    if not text:
        return ""
    text = _CTRL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def ascii_ratio(text: str) -> float:
    """ASCII 字符占比（含空格；is_english_text 思路的宽松变体，理由见文件头）。"""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isascii()) / len(text)


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_streaming(load_dataset, source):
    """尝试加载单个源（内部按 split 回退）。

    成功返回 (ds, split, text_field)，失败返回 None。
    加载后会拉第一条样本验证 text 字段——streaming 是惰性的，
    字段问题只有真正取数时才暴露。
    """
    errors = []
    for split in (source["split"], None):
        try:
            if split is None:
                # 不指定 split → DatasetDict；优先 train，否则取第一个
                ddict = load_dataset(source["hf_id"], streaming=True)
                used_split = next(
                    (s for s in ("train",) if s in ddict), next(iter(ddict))
                )
                ds = ddict[used_split]
            else:
                ds = load_dataset(source["hf_id"], split=split, streaming=True)
                used_split = split
            first = next(iter(ds))
            text_field = next(
                (k for k in TEXT_FIELD_CANDIDATES
                 if isinstance(first.get(k), str)),
                None,
            )
            if text_field is None:
                raise ValueError(
                    f"无正文字段（候选 {TEXT_FIELD_CANDIDATES}），"
                    f"实际字段: {sorted(first.keys())}"
                )
            return ds, used_split, text_field
        except Exception as e:
            errors.append(f"split={split!r}: {e}")
    for msg in errors:
        print(f"  [失败] {source['hf_id']} {msg}", file=sys.stderr)
    return None


def download(
    target_bytes: int,
    min_len: int,
    max_len: int,
    min_ascii_ratio: float,
    out_path: str,
) -> int:
    if tqdm is None:
        print("[错误] 需要 tqdm 库: pip install tqdm", file=sys.stderr)
        return 1
    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 需要 datasets 库: pip install datasets", file=sys.stderr)
        return 1

    unlimited = target_bytes == 0
    print("回退链        : " + " → ".join(s["hf_id"] for s in MATH_SOURCES))
    print(f"Length filter : {min_len} - {max_len if max_len > 0 else '∞'} chars (0 = 关闭)")
    if min_ascii_ratio > 0:
        print(f"ASCII filter  : >= {min_ascii_ratio:.0%}（宽松英文兜底, 0 = 关闭）")
    else:
        print("ASCII filter  : 关闭")
    print(f"Target size   : {'不限（整个数据集）' if unlimited else fmt_size(target_bytes)}")
    print(f"Output file   : {out_path}")
    print("Streaming ... : 启动（首次拉取分片可能稍慢）")

    loaded = None
    chosen = None
    for source in MATH_SOURCES:
        print(f"尝试 {source['hf_id']} ...")
        loaded = load_streaming(load_dataset, source)
        if loaded is not None:
            chosen = source
            break
    if loaded is None:
        print("[错误] 回退链所有数据集均加载失败", file=sys.stderr)
        return 1
    ds, split, text_field = loaded
    print(f"已选择        : {chosen['hf_id']}  split: {split}")
    print(f"字段          : 正文='{text_field}'")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n = 0
    skipped_short = 0
    skipped_long = 0
    skipped_ascii = 0
    bytes_written = 0

    # total=None 时 tqdm 只显示已处理量（无百分比），与
    # download_everyday_conversations.py 的进度条用法一致
    pbar = tqdm(
        total=None if unlimited else target_bytes,
        unit="B",
        unit_scale=True,
        desc="下载 math",
        ncols=100,
    )

    # newline="\n" 强制 LF，避免在 Windows 上写成 CRLF
    # （text_train.cpp 的 getline 已 trim \r，但保持 LF 更通用）
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for sample in ds:
            text = sample.get(text_field, "") or ""
            cleaned = clean_doc(text)

            # 行长过滤：过短视为噪声；过长仅防御极端异常（0 = 关闭）
            if not cleaned or len(cleaned) < min_len:
                skipped_short += 1
                continue
            if max_len > 0 and len(cleaned) > max_len:
                skipped_long += 1
                continue

            # 英文兜底：宽松 ASCII 占比（理由见文件头）
            if min_ascii_ratio > 0 and ascii_ratio(cleaned) < min_ascii_ratio:
                skipped_ascii += 1
                continue

            line = cleaned + "\n"
            f.write(line)
            n += 1
            line_bytes = len(line.encode("utf-8"))
            bytes_written += line_bytes

            # 更新进度条（增加已写出字节数）
            pbar.update(line_bytes)
            pbar.set_postfix(条数=n)

            # 达到目标大小就停（不限体积则遍历到数据集末尾）
            if not unlimited and bytes_written >= target_bytes:
                pbar.set_description("已达到大小限制，停止下载")
                break

    pbar.close()

    print()
    print("[+] 完成:")
    print(f"    写入文档数        : {n:,}")
    print(f"    跳过(过短/空)     : {skipped_short:,}")
    print(f"    跳过(过长)        : {skipped_long:,}")
    if min_ascii_ratio > 0:
        print(f"    跳过(ASCII占比过低): {skipped_ascii:,}")
    print(f"    写出字节数        : {fmt_size(bytes_written)}（文件 {fmt_size(os.path.getsize(out_path))}）")
    print(f"    输出文件          : {out_path}")
    if n == 0:
        print("[错误] 未写出任何文档（过滤条件可能过严）", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 数学预训练语料（项目格式：每行一个文档）"
    )
    parser.add_argument(
        "--target_mb",
        type=int,
        default=DEFAULT_TARGET_MB,
        help=f"目标体积(MB), 达到即停止 (默认: {DEFAULT_TARGET_MB}; 0 = 不限, 遍历整个数据集)",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=DEFAULT_MIN_LEN,
        help=f"最小行长(字符数), 0 = 关闭 (默认: {DEFAULT_MIN_LEN})",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"最大行长(字符数), 0 = 不限 (默认: {DEFAULT_MAX_LEN})",
    )
    parser.add_argument(
        "--min_ascii_ratio",
        type=float,
        default=DEFAULT_MIN_ASCII_RATIO,
        help=f"ASCII 字符占比下限, 0 = 关闭 (默认: {DEFAULT_MIN_ASCII_RATIO}; "
        "宽松英文兜底, 详见文件头说明)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/math_pretrain.txt)",
    )
    args = parser.parse_args()

    if args.target_mb < 0 or args.min_len < 0 or args.max_len < 0:
        parser.error("--target_mb / --min_len / --max_len 不能为负数")
    if not 0.0 <= args.min_ascii_ratio <= 1.0:
        parser.error("--min_ascii_ratio 需在 0~1 之间（0 = 关闭过滤）")

    target_bytes = args.target_mb * 1024 * 1024  # 0 = 不限
    out_path = args.output or os.path.join(OUT_DIR, "math_pretrain.txt")

    return download(
        target_bytes, args.min_len, args.max_len, args.min_ascii_ratio, out_path
    )


if __name__ == "__main__":
    sys.exit(main())
