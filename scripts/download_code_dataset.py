"""下载 HuggingFace 代码预训练语料并转为项目预训练格式。

数据集（回退链：按序尝试，取第一个「streaming 可用且带正文字段」的源，
全部失败则以非 0 退出）:

  1. bigcode/the-stack-dedup（先按 config=default、split=train 加载，
     config 失败再不带配置名重试一次）
     - The Stack 的近重复去重版，海量开源多语言代码（gated=auto 仓库）；
     - 正文字段自动识别 text → content（the-stack 系列实际为 content；
       两者都缺失才落到下一源）；
     - 若样本带 language/lang 字段 → 按白名单过滤，只保留
       Python/JavaScript/Java/C/C++/Go/Rust/TypeScript；
       无该字段 → 跳过语言过滤（本回退链中 codeparrot 源即如此）；
     - 2026-10 实测：该仓库 gated，未登录（无 HF token）拉流会 401，
       此时自动落入下一源。
  2. codeparrot/codeparrot-clean-valid（split=train，6.1 万行）
     - 正文字段 content；无语言字段 → 跳过语言过滤。

转换格式（项目预训练格式，同 download_pretrain.py；参考 src/text_train.cpp
的滑动窗口训练端）:
  - 每行 = 一个代码文件（文档），行内不含换行符
  - 清洗：去控制字符 + 合并空白（换行/Tab/缩进 → 单空格）——与
    download_pretrain.py 的 clean_doc 同构，但**不做 HTML 标签剥离**
    （代码里的 #include <vector>、模板参数、比较运算符会被 <[^>]+> 误删）
  - 训练端把每行编码为 [BOS]+tokens+[EOS] 拼成连续 token 流后按 seq_len
    滑动切窗，超长文档会被自动切成多个窗口，短文档也被窗口拼接利用

过滤:
  - 代码**不做**英文兜底过滤：download_everyday_conversations.py 的
    is_english_text 依赖「ASCII 字母占比」，代码符号/数字密度高会大量
    误杀正常源码，故改为可选的行长度过滤 --min_len / --max_len
    （行 = 输出行 = 一个文档的字符数；设 0 即关闭该项）
  - 语言白名单仅在源带 language/lang 字段时生效（见上）

用法:
  python scripts/download_code_dataset.py
  python scripts/download_code_dataset.py --target_mb 200
  python scripts/download_code_dataset.py --target_mb 0           # 不限体积, 遍历整个数据集
  python scripts/download_code_dataset.py --min_len 0 --max_len 0 # 关闭行长过滤
  python scripts/download_code_dataset.py --output datasets/my_code.txt

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

# 回退链（按序尝试；每个源先试 (config, split) 组合，再退到不指定 split）：
#   configs : 依次尝试的配置名（None = 不传配置名）
#   split   : 首选 split（找不到时退到 DatasetDict 的 train / 第一个 split）
CODE_SOURCES = (
    {
        "hf_id": "bigcode/the-stack-dedup",
        "configs": ("default", None),
        "split": "train",
    },
    {
        "hf_id": "codeparrot/codeparrot-clean-valid",
        "configs": (None,),
        "split": "train",
    },
)

# 正文字段候选（按序识别）：the-stack / codeparrot 系列为 content，部分数据集为 text
TEXT_FIELD_CANDIDATES = ("text", "content")
# 语言字段候选（按序识别）：the-stack 为 lang，个别数据集为 language；
# 都不存在 → 跳过语言过滤
LANGUAGE_FIELD_CANDIDATES = ("language", "lang")

# 语言白名单（小写比较；the-stack 的 lang 取值 = GitHub Linguist 语言名）
CODE_LANG_WHITELIST = {
    "python", "javascript", "java", "c", "c++", "go", "rust", "typescript",
}

# 默认目标体积（MB）；0 表示不限，遍历整个数据集
DEFAULT_TARGET_MB = 100

# 行长过滤（字符数）：过短（空文件/只剩 import 头的碎片）视为噪声丢弃；
# 超长不截断——训练端滑动窗口会自动切分，max_len 仅作防御性上限；
# 两项都可设 0 关闭（代码语料不做过滤时的退路）
DEFAULT_MIN_LEN = 40
DEFAULT_MAX_LEN = 100000

# 清洗：去控制字符（\t \n \r \f \v 保留给下一步空白合并；NUL/ESC 等直接剔除）
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 合并所有空白（含换行/Tab/缩进）为单个空格——保证输出行内不含换行符
_WS_RE = re.compile(r"\s+")


def clean_doc(text: str) -> str:
    """清洗一个代码文档：去控制字符、合并空白（含换行）为单空格。

    与 download_pretrain.py 的 clean_doc 同构，但省去 HTML 标签剥离——
    代码里的 <...> 是模板参数 / include 路径 / 比较运算符，不是标签。
    """
    if not text:
        return ""
    text = _CTRL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_streaming(load_dataset, source):
    """尝试加载单个源（内部按 (config, split) 组合回退）。

    成功返回 (ds, split, config, text_field, lang_field)，失败返回 None。
    加载后会拉第一条样本验证正文字段是否存在——streaming 是惰性的，
    字段问题只有真正取数时才暴露。
    """
    errors = []
    for config in source["configs"]:
        for split in (source["split"], None):
            try:
                if split is None:
                    # 不指定 split → DatasetDict；优先 train，否则取第一个
                    ddict = load_dataset(
                        source["hf_id"], config, streaming=True
                    )
                    used_split = next(
                        (s for s in ("train",) if s in ddict), next(iter(ddict))
                    )
                    ds = ddict[used_split]
                else:
                    ds = load_dataset(
                        source["hf_id"], config, split=split, streaming=True
                    )
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
                lang_field = next(
                    (k for k in LANGUAGE_FIELD_CANDIDATES
                     if isinstance(first.get(k), str)),
                    None,
                )
                return ds, used_split, config, text_field, lang_field
            except Exception as e:
                errors.append(f"config={config!r} split={split!r}: {e}")
    for msg in errors:
        print(f"  [失败] {source['hf_id']} {msg}", file=sys.stderr)
    return None


def download(
    target_bytes: int,
    min_len: int,
    max_len: int,
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
    print("回退链        : " + " → ".join(s["hf_id"] for s in CODE_SOURCES))
    print(f"Length filter : {min_len} - {max_len if max_len > 0 else '∞'} chars (0 = 关闭)")
    print(f"Target size   : {'不限（整个数据集）' if unlimited else fmt_size(target_bytes)}")
    print(f"Output file   : {out_path}")
    print("Streaming ... : 启动（首次拉取分片可能稍慢）")

    loaded = None
    chosen = None
    for source in CODE_SOURCES:
        print(f"尝试 {source['hf_id']} ...")
        loaded = load_streaming(load_dataset, source)
        if loaded is not None:
            chosen = source
            break
    if loaded is None:
        print("[错误] 回退链所有数据集均加载失败", file=sys.stderr)
        return 1
    ds, split, config, text_field, lang_field = loaded

    lang_note = f"language='{lang_field}'" if lang_field else "无语言字段（跳过语言过滤）"
    print(
        f"已选择        : {chosen['hf_id']}"
        + (f"  config: {config}" if config else "")
        + f"  split: {split}"
    )
    print(f"字段          : 正文='{text_field}'  {lang_note}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n = 0
    skipped_short = 0
    skipped_long = 0
    skipped_lang = 0
    bytes_written = 0

    # total=None 时 tqdm 只显示已处理量（无百分比），与
    # download_everyday_conversations.py 的进度条用法一致
    pbar = tqdm(
        total=None if unlimited else target_bytes,
        unit="B",
        unit_scale=True,
        desc="下载 code",
        ncols=100,
    )

    # newline="\n" 强制 LF，避免在 Windows 上写成 CRLF
    # （text_train.cpp 的 getline 已 trim \r，但保持 LF 更通用）
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for sample in ds:
            text = sample.get(text_field, "") or ""
            cleaned = clean_doc(text)

            # 行长过滤（可选）：过短视为噪声；过长仅防御极端异常（0 = 关闭）
            if not cleaned or len(cleaned) < min_len:
                skipped_short += 1
                continue
            if max_len > 0 and len(cleaned) > max_len:
                skipped_long += 1
                continue

            # 语言白名单：仅源带 language/lang 字段时生效
            if lang_field:
                lang = str(sample.get(lang_field) or "").strip().lower()
                if lang and lang not in CODE_LANG_WHITELIST:
                    skipped_lang += 1
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
    if lang_field:
        print(f"    跳过(语言白名单外) : {skipped_lang:,}")
    print(f"    写出字节数        : {fmt_size(bytes_written)}（文件 {fmt_size(os.path.getsize(out_path))}）")
    print(f"    输出文件          : {out_path}")
    if n == 0:
        print("[错误] 未写出任何文档（过滤条件可能过严）", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 代码预训练语料（项目格式：每行一个文档）"
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
        "--output",
        default=None,
        help="输出文件路径 (默认: scripts/datasets/code_pretrain.txt)",
    )
    args = parser.parse_args()

    if args.target_mb < 0 or args.min_len < 0 or args.max_len < 0:
        parser.error("--target_mb / --min_len / --max_len 不能为负数")

    target_bytes = args.target_mb * 1024 * 1024  # 0 = 不限
    out_path = args.output or os.path.join(OUT_DIR, "code_pretrain.txt")

    return download(target_bytes, args.min_len, args.max_len, out_path)


if __name__ == "__main__":
    sys.exit(main())
