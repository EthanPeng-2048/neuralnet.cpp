"""将 HuggingFace 对话数据集转换为带对话标记的纯文本语料。

输出格式（一行一个 step）：
  <system>系统提示</system><user>用户问题</user><assistant>模型回答</assistant>

换行符在内容中会被移除。分词器读到对话标记会映射为特殊 token ID，
普通文本正常分词。

用法:
  python hf_to_txt.py <dataset_name> [--subset <subset>] [--split <split>]
                      [--output <path>] [--max_rows <n>]

示例:
  python hf_to_txt.py Mxode/Chinese-Instruct --subset dpsk-r1-distil --max_rows 50000
  python hf_to_txt.py Mxode/Chinese-Instruct-Lite --output datasets/chat_corpus.txt
"""
import argparse
import json
import os
import re
import sys

import markdown as md

# 对话标记（与分词器中定义一致）
SYSTEM_TAG    = "<system>"
USER_TAG      = "<user>"
ASSISTANT_TAG = "<assistant>"
END_PREFIX    = "</"


_HTML_TAG_RE = re.compile(r"<[^>]+>")

# 在块级 Markdown 标记前插入换行，让 markdown 库正确识别
_BLOCK_RE = re.compile(
    r"(?<=\S)"           # 前面有非空白字符（不是行首）
    r"(?=---(?!\w))"     # --- 分隔线
    r"|"
    r"(?<=\S)"
    r"(?=#{1,6}\s)"      # # 标题
    r"|"
    r"(?<=\s)"
    r"(?=-\s)"           # - 列表
    r"|"
    r"(?<=\s)"
    r"(?=\d+\.\s)"       # 1. 有序列表
)


def _protect_latex(text: str) -> tuple[str, list[tuple[str, str]]]:
    """提取 LaTeX 数学片段并用占位符替换，防止被清理过程破坏。

    处理: \\boxed{...} \\text{...} \\[...\\] \\(...\\) $...$
    返回 (处理后文本, [(占位符, 原文)] )。
    """
    blocks: list[tuple[str, str]] = []
    counter = [0]

    def _save(s: str) -> str:
        tag = f"\x00L{counter[0]}\x00"
        counter[0] += 1
        blocks.append((tag, s))
        return tag

    # 1) 简化 \boxed{...} \text{...} → 保留花括号内容
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)

    # 2) 字符扫描提取 \[...\] 和 \(...\)，只保留内容，丢弃定界符
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in ("[", "("):
            open_ch = text[i + 1]
            close_ch = "]" if open_ch == "[" else ")"
            j = i + 2
            while j < len(text):
                if text[j] == "\\" and j + 1 < len(text) and text[j + 1] == close_ch:
                    break
                j += 1
            # 提取 \[ 和 \] 之间的内容（不含定界符）
            inner = text[i + 2 : j]
            result.append(_save(inner))
            i = j + 2  # 跳过 \]
        else:
            result.append(text[i])
            i += 1
    text = "".join(result)

    # 3) 行内 $...$（排除 $$ 块级）
    text = re.sub(r"(?<!\$)\$(?!\$)[^$]+?\$(?!\$)",
                  lambda m: _save(m.group(0)), text)

    # 4) 块级 $$...$$
    text = re.sub(r"\$\$.+?\$\$",
                  lambda m: _save(m.group(0)), text)

    return text, blocks


def strip_markdown(text: str) -> str:
    """清理 Markdown 格式标记，保留纯文本内容。

    使用 markdown 库将 Markdown 转为 HTML，再剥离 HTML 标签。
    额外处理 LaTeX 定界符和 \\boxed / \\text 等 LaTeX 命令。
    """
    if not text:
        return ""

    # --- 预处理 LaTeX：提取并保护 LaTeX 片段 ---
    text, latex_blocks = _protect_latex(text)

    # --- 恢复块级结构，让 markdown 库正确解析 ---
    text = _BLOCK_RE.sub("\n", text)

    # --- markdown → HTML → 纯文本 ---
    html = md.markdown(text, extensions=["extra"])
    clean = _HTML_TAG_RE.sub(" ", html)

    # 还原 LaTeX 片段
    for tag, original in latex_blocks:
        clean = clean.replace(tag, original)

    return clean


def clean_text(text: str) -> str:
    """清洗文本：移除 Markdown 格式、换行、多余空白。"""
    if not text:
        return ""
    # 先清理 Markdown 格式
    text = strip_markdown(text)
    # 移除残留的水平分隔线标记
    text = re.sub(r"(?:---|\*\*\*|___)+", " ", text)
    # 移除所有换行和回车
    text = text.replace("\r", "").replace("\n", "")
    # 合并连续空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_dialogue(system: str, prompt: str, response: str) -> str | None:
    """将一条对话组装为一行带标记的文本。"""
    prompt = clean_text(prompt)
    response = clean_text(response)

    if not prompt or not response:
        return None
    # 跳过过短或过长的样本
    if len(prompt) < 4 or len(response) < 8:
        return None
    if len(prompt) > 2000 or len(response) > 2000:
        return None

    parts = []
    if system:
        system = clean_text(system)
        parts.append(f"{SYSTEM_TAG}{system}{END_PREFIX}system>")
    parts.append(f"{USER_TAG}{prompt}{END_PREFIX}user>")
    parts.append(f"{ASSISTANT_TAG}{response}{END_PREFIX}assistant>")
    return "".join(parts)


def detect_columns(sample: dict) -> tuple[str, str, str]:
    """自动检测 prompt/response/system 字段名。"""
    prompt_col = response_col = system_col = ""

    # prompt 字段候选
    for col in ("prompt", "instruction", "input", "question", "query",
                "human", "user_input", "question_text"):
        if col in sample and isinstance(sample[col], str):
            prompt_col = col
            break
    # response 字段候选
    for col in ("response", "output", "answer", "assistant", "completion",
                "gpt", "target", "answer_text"):
        if col in sample and isinstance(sample[col], str):
            response_col = col
            break
    # system 字段候选
    for col in ("system", "system_prompt", "context"):
        if col in sample and isinstance(sample[col], str):
            system_col = col
            break

    # 兼容 messages 格式（OpenAI 风格）
    if not prompt_col and "messages" in sample:
        messages = sample["messages"]
        if isinstance(messages, list):
            system_msgs = []
            user_msgs = []
            assistant_msgs = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_msgs.append(content)
                elif role == "user":
                    user_msgs.append(content)
                elif role == "assistant":
                    assistant_msgs.append(content)
            if user_msgs and assistant_msgs:
                # 返回一个临时的组装结果
                return "__messages__", "", ""  # 标记为 messages 格式

    return prompt_col, response_col, system_col


def process_messages(sample: dict) -> str | None:
    """处理 OpenAI messages 格式。"""
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None

    system_parts = []
    user_parts = []
    assistant_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # 多模态格式，只取文本部分
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    system = " ".join(system_parts)
    prompt = " ".join(user_parts)
    response = " ".join(assistant_parts)

    return format_dialogue(system, prompt, response)


def convert_dataset(
    dataset_name: str,
    subset: str | None,
    split: str,
    output_path: str,
    max_rows: int | None,
):
    """主转换函数。"""
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("需要安装 datasets 库: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # 获取子集列表
    if subset is None:
        try:
            configs = get_dataset_config_names(dataset_name)
            if configs:
                subset = configs[0]
                print(f"自动选择子集: {subset}")
        except Exception:
            pass

    print(f"加载数据集: {dataset_name} (subset={subset}, split={split}) ...")
    ds = load_dataset(dataset_name, subset, split=split)

    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    print(f"总样本数: {len(ds)}")

    # 自动检测字段（用第一条样本）
    first = ds[0]
    prompt_col, response_col, system_col = detect_columns(first)
    is_messages = (prompt_col == "__messages__")

    if is_messages:
        print("检测到 messages 格式")
    else:
        print(f"检测到字段: prompt={prompt_col!r}, response={response_col!r}, system={system_col!r}")
        if not prompt_col or not response_col:
            print(f"字段: {list(first.keys())}", file=sys.stderr)
            print("无法自动识别 prompt/response 字段，请检查数据集格式", file=sys.stderr)
            sys.exit(1)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            if is_messages:
                line = process_messages(sample)
            else:
                system = sample.get(system_col, "") if system_col else ""
                prompt = sample.get(prompt_col, "")
                response = sample.get(response_col, "")
                line = format_dialogue(system, prompt, response)

            if line:
                f.write(line + "\n")
                written += 1
            else:
                skipped += 1

            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i + 1} 条, 写入 {written}, 跳过 {skipped}")

    print(f"\n完成! 写入 {written} 条, 跳过 {skipped} 条")
    print(f"输出: {output_path}")

    # 显示文件大小
    size = os.path.getsize(output_path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            print(f"文件大小: {size:.1f} {unit}")
            break
        size /= 1024


def main():
    parser = argparse.ArgumentParser(description="HF 对话数据集 → 带标记的纯文本语料")
    parser.add_argument("dataset", help="HuggingFace 数据集名称，如 Mxode/Chinese-Instruct")
    parser.add_argument("--subset", default=None, help="子集名称 (默认自动选择第一个)")
    parser.add_argument("--split", default="train", help="数据集 split (默认: train)")
    parser.add_argument("--output", default=None, help="输出文件路径 (默认: datasets/<dataset_name>.txt)")
    parser.add_argument("--max_rows", type=int, default=None, help="最大处理行数")
    args = parser.parse_args()

    output = args.output
    if output is None:
        # 从数据集名生成输出路径
        name = args.dataset.split("/")[-1].lower().replace("-", "_")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output = os.path.join(script_dir, "datasets", f"{name}.txt")

    convert_dataset(args.dataset, args.subset, args.split, output, args.max_rows)


if __name__ == "__main__":
    main()
