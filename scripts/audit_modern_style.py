#!/usr/bin/env python3
"""audit_modern_style.py — neuralnet.cpp 现代 C++ 风格审查器（指针/new-delete/malloc 重点）

用途：
  1. 日常开发：`python scripts/audit_modern_style.py` —— 有 ERROR 时退出码 1，可挂 CI。
  2. 基线核对：输出 E（错误）/ W（裸指针候选，需人工裁决）/ A（nn-allow 豁免）统计。

检查项：
  E1 C 风格 new          E2 C 风格 delete        E3 malloc/free 族
  E4 reinterpret_cast     E5 const_cast           E6 C 风格指针转换 (T*)
  E7 void* 裸空指针
  W1 裸指针声明候选（Type *name / Type* name / 模板类型 *name）—— 需人工裁决：
     所有权应改 unique_ptr；非拥有应改 observer_ptr<T>/引用；连续数据视图应改 std::span。
  W2 函数参数 C 数组（T arr[]，隐式退化为裸指针）
  W3 函数指针声明（void (*fp)(...)，C 兼容风格，建议改 std::function / 概念约束）
  I1-I4 信息统计：unique_ptr / make_unique / observer_ptr / std::span 使用量

豁免机制：
  合法的 C 互操作 / 第三方边界行尾加 `// nn-allow: <原因>`，该行不计入 E/W，只计入 A。

退出码：0 = 无 ERROR；1 = 存在 ERROR。W 不阻塞（--fail-on-warn 时阻塞）。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 注释/字符串剥离（保留行号与列位置：用空格替换，不删字符） ──────────────
def strip_comments(src: str) -> str:
    out = []
    i, n = 0, len(src)
    state = "code"  # code | line | block | str | chr
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                out.append("  ")
                i += 2
                continue
            if c == "R" and nxt == '"':
                # raw string R"delim(...)delim"：整段置空（可跨行，含 GLSL 片段）
                k = i + 2
                while k < n and src[k] != "(":
                    k += 1
                delim = src[i + 2:k] if k < n else ""
                term = ")" + delim + '"'
                e = src.find(term, k)
                e = e + len(term) if e != -1 else n
                out.append(" " * (e - i))
                i = e
                continue
            if c == '"':
                state = "str"
            elif c == "'":
                state = "chr"
            out.append(c)
            i += 1
        elif state == "line":
            if c == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
        else:  # str / chr：内容全部置空（避免字符串里的 new/malloc 等误报）
            if c == "\n":  # 非 raw string 不可能跨行；未闭合时重置状态防止失步
                state = "code"
                out.append("\n")
                i += 1
                continue
            if c == "\\":
                out.append("  ")
                i += 2
                continue
            if (state == "str" and c == '"') or (state == "chr" and c == "'"):
                state = "code"
                out.append(c)
            else:
                out.append(" ")
            i += 1
    return "".join(out)


@dataclass
class Finding:
    code: str      # E1..E7 / W1..W3 / A
    level: str     # ERROR / WARN / ALLOW
    path: str
    line: int
    text: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def add(self, f: Finding) -> None:
        self.findings.append(f)
        self.stats[f.code] = self.stats.get(f.code, 0) + 1


# ── 检查规则 ────────────────────────────────────────────────────────────────
RE_NEW = re.compile(r"\bnew\s+(?=[\w:(])")
# delete 操作符：delete 后跟表达式；`= delete;`（删除特殊成员）不匹配
RE_DELETE = re.compile(r"\bdelete\s+(?=[\w[(])")
# malloc 族：排除成员调用（pool.free() / p->free()）
RE_MALLOC = re.compile(r"(?<![\.\w>])\b(malloc|calloc|realloc|aligned_alloc|_aligned_malloc|aligned_free|posix_memalign|free)\s*\(")
RE_REINTER = re.compile(r"\breinterpret_cast\b")
RE_CONSTCAST = re.compile(r"\bconst_cast\b")
RE_C_CAST = re.compile(r"\(\s*[\w:<>,&]+\s*\*+\s*\)")
RE_VOIDPTR = re.compile(r"\bvoid\s*\*")
# W1 裸指针声明候选：Type *name（用项目命名规范区分类型/变量：类 CamelCase、变量 snake_case）
RE_PTR_DECL = re.compile(
    r"(?<![\w.>])"
    r"((?:const\s+|volatile\s+|signed\s+|unsigned\s+)*"
    r"[A-Za-z_][\w:]*(?:<[^<>]{0,80}>)?)"  # 类型（允许一层模板）
    r"\s*\*{1,2}\s*"
    r"([A-Za-z_]\w*)"                        # 变量名
)
# 尾随返回类型 -> T *（无变量名，单独检测）
RE_TRAIL_RET = re.compile(r"->\s*(?:const\s+)?[A-Za-z_][\w:]*(?:<[^<>]{0,80}>)?\s*\*")
# 内置/项目级类型名（小写标识符中算"类型"的白名单）
_BUILTIN_TYPES = {
    "int", "float", "double", "char", "bool", "void", "auto", "long", "short",
    "size_t", "ssize_t", "ptrdiff_t", "nullptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "Scalar",
}

# C++ 关键字——不应被识别为类型名或变量名（借鉴 CODE_CHECK）
_CPP_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "class", "struct",
    "enum", "typedef", "using", "namespace", "template", "operator",
    "sizeof", "alignof", "new", "delete", "throw", "try", "do", "else",
    "case", "default", "goto", "break", "continue", "const", "constexpr",
    "volatile", "static", "extern", "inline", "virtual", "friend",
    "public", "private", "protected", "final", "override", "void",
    "int", "short", "long", "char", "float", "double", "bool",
    "signed", "unsigned", "auto", "decltype", "typename", "this",
    "nullptr", "true", "false", "and", "or", "not", "xor", "bitand",
    "bitor", "compl", "typeid", "static_assert",
}


def _is_type_token(tok: str) -> bool:
    """按项目命名规范判断 * 前的 token 是否为类型：
    CamelCase 类名 / 含 :: / 内置类型 / 以 > 结尾的模板类型。
    排除：关键字、单字母大写（R/C/M/B/T/K 等常量）和全大写（BLOCK_SIZE 等宏常量）。
    小写 snake_case 标识符视为变量 → 大概率是乘法表达式，跳过。"""
    t = re.sub(r"^(?:const|volatile|signed|unsigned)\s+", "", tok).strip()
    if not t:
        return False
    core = t.split("<")[0].strip()
    if core in _CPP_KEYWORDS:
        return False
    if core in _BUILTIN_TYPES:
        return True
    if "::" in t or t.endswith(">"):
        return True
    # CamelCase 类名：首字母大写 + 第二字母小写（排除 R/M/B 等单字母和 BLOCK_SIZE 等全大写）
    if re.match(r"[A-Z][a-z]", core) and not core.isupper():
        return True
    return False


# ── 模板安全的参数分割（借鉴 CODE_CHECK）──────────────────────────────────
def _split_params(params_text: str) -> list[str]:
    """按顶层逗号分割参数列表，正确处理嵌套 ()[]{}<>。"""
    items: list[str] = []
    buf: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    close_map = {")": "(", "]": "[", "}": "{", ">": "<"}
    for ch in params_text:
        if ch in depths:
            depths[ch] += 1
        elif ch in close_map:
            depths[close_map[ch]] = max(0, depths[close_map[ch]] - 1)
        if ch == "," and all(v == 0 for v in depths.values()):
            token = "".join(buf).strip()
            if token:
                items.append(token)
            buf = []
        else:
            buf.append(ch)
    token = "".join(buf).strip()
    if token:
        items.append(token)
    return items


def _extract_param_pointer_names(src: str) -> set[str]:
    """提取整个源码中函数参数里的指针参数名——参数里的 * 必然是指针。"""
    ptr_names: set[str] = set()
    for m in re.finditer(r"\(([^)]{1,2000})\)", src):
        params_text = m.group(1)
        if "*" not in params_text:
            continue
        for param in _split_params(params_text):
            p = param.split("=")[0].strip()
            if not p or p == "void" or "..." in p:
                continue
            if "*" not in p:
                continue
            name_m = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", p)
            if name_m:
                name = name_m.group(1)
                if name not in _CPP_KEYWORDS:
                    ptr_names.add(name)
    return ptr_names


RE_ARR_PARAM = re.compile(r"\b[\w:]+\s*\w+\s*\[\s*\]\s*[,)=]")
RE_FUNCPTR = re.compile(r"\(\s*\*\s*[\w]+\s*\)\s*\([^;{]*\)")
RE_ALLOW = re.compile(r"//\s*nn-allow\b")
RE_SMART = re.compile(r"\bstd::(?:unique_ptr|shared_ptr|weak_ptr)\b")
RE_MAKE_UNIQUE = re.compile(r"\bstd::make_unique\b")
RE_OBSERVER = re.compile(r"\bobserver_ptr\b|\bmake_observer\b")
RE_SPAN = re.compile(r"\bstd::span\b")

CODE_LEVEL = {
    "E1": "ERROR", "E2": "ERROR", "E3": "ERROR", "E4": "ERROR",
    "E5": "ERROR", "E6": "ERROR", "E7": "ERROR",
    "W1": "WARN", "W2": "WARN", "W3": "WARN",
    "A": "ALLOW",
}


def scan_file(path: Path, rel: str, rep: Report) -> None:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  ! 无法读取 {path}: {e}", file=sys.stderr)
        return
    clean = strip_comments(src)
    lines = clean.split("\n")
    raw_lines = src.split("\n")
    # 预提取函数参数中的指针参数名（参数里的 * 必然是指针声明，不是乘法）
    _known_ptr_params = _extract_param_pointer_names(clean)
    for idx, line in enumerate(lines, start=1):
        raw = raw_lines[idx - 1].strip() if idx - 1 < len(raw_lines) else ""
        allowed = bool(RE_ALLOW.search(raw)) or bool(RE_ALLOW.search(line))
        if allowed:
            rep.add(Finding("A", "ALLOW", rel, idx, raw))
            continue

        def hit(code: str, text: str) -> None:
            rep.add(Finding(code, CODE_LEVEL[code], rel, idx, text))

        for m in RE_NEW.finditer(line):
            # 排除 `new` 作为普通单词误报（如 newline）——finditer 已用 \b
            t = line[m.start():].strip()
            if not re.match(r"new\s*[\w(:]", t):
                continue
            hit("E1", raw)
        if RE_DELETE.search(line):
            hit("E2", raw)
        for m in RE_MALLOC.finditer(line):
            # 排除方法/函数定义（如 MemoryPool 的成员 `void free(...)`）
            if re.search(r"\b(void|auto|static|int|bool)\s*$", line[:m.start()].rstrip()):
                continue
            hit("E3", raw)
            break
        if RE_REINTER.search(line):
            hit("E4", raw)
        if RE_CONSTCAST.search(line):
            hit("E5", raw)
        for m in RE_C_CAST.finditer(line):
            seg = line[m.start():m.end()]
            # 排除 sizeof(T*) / decltype 上下文误报
            if re.search(r"sizeof|decltype", line[:m.start()][-30:]):
                continue
            hit("E6", raw)
        if RE_VOIDPTR.search(line):
            hit("E7", raw)
        for m in RE_PTR_DECL.finditer(line):
            ttype, name = m.group(1), m.group(2)
            # ── 排除 1：变量名已是函数参数中的指针 → 不是新声明 ──
            if name in _known_ptr_params:
                continue
            # ── 排除 2：乘法 / 解引用误报 ──
            _star_pos = m.end(1)
            _left_ctx  = line[:_star_pos].rstrip()
            _right_ctx = line[m.start(2):m.start(2) + 20].lstrip()
            # 如果 * 前没有声明上下文关键字 → 是表达式（乘法/解引用），不是声明
            _has_decl_ctx = bool(re.search(
                r"(?:^|[{;,=(]|\b(?:const|auto|static|volatile|virtual|override|"
                r"noexcept|constexpr|inline|using|typedef|typename|class|struct)\b)\s*$",
                _left_ctx
            ))
            if not _has_decl_ctx:
                continue
            # 排除解引用：return *ptr / = *ptr / ( *ptr ) 中的 * 是解引用运算符
            if re.search(r"(?:return|=|[(,])\s*$", _left_ctx):
                continue
            # 排除函数指针 (*fp) —— 交给 W3
            if re.search(r"[\(&]\s*$", _left_ctx):
                if re.match(r"[\w]+\s*\)", _right_ctx):
                    continue
            hit("W1", raw)
        if RE_ARR_PARAM.search(line):
            hit("W2", raw)
        if RE_FUNCPTR.search(line):
            hit("W3", raw)

    # 信息统计（对全文件计次）
    full = clean
    rep.stats["I1"] = rep.stats.get("I1", 0) + len(RE_SMART.findall(full))
    rep.stats["I2"] = rep.stats.get("I2", 0) + len(RE_MAKE_UNIQUE.findall(full))
    rep.stats["I3"] = rep.stats.get("I3", 0) + len(RE_OBSERVER.findall(full))
    rep.stats["I4"] = rep.stats.get("I4", 0) + len(RE_SPAN.findall(full))


def main() -> int:
    ap = argparse.ArgumentParser(description="neuralnet.cpp 现代 C++ 风格审查器")
    ap.add_argument("roots", nargs="*", default=["include", "src", "tools"],
                    help="扫描根目录（默认 include src tools）")
    ap.add_argument("--fail-on-warn", action="store_true", help="W 级也视为失败")
    ap.add_argument("--by-file", action="store_true", help="按文件列出 E/W 分布（裁决用）")
    ap.add_argument("--quiet", action="store_true", help="只打印 ERROR 与汇总")
    args = ap.parse_args()

    rep = Report()
    root = Path(__file__).resolve().parent.parent
    seen: set[Path] = set()
    for r in args.roots:
        p = (root / r) if not Path(r).is_absolute() else Path(r)
        if p.is_file():
            files = [p]
        else:
            files = sorted(
                x for x in p.rglob("*")
                if x.suffix in (".hpp", ".cpp", ".h", ".cc") and x.is_file()
            ) if p.exists() else []
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            rel = str(f.relative_to(root)).replace("\\", "/")
            scan_file(f, rel, rep)

    errors = [f for f in rep.findings if f.level == "ERROR"]
    warns = [f for f in rep.findings if f.level == "WARN"]
    allows = [f for f in rep.findings if f.level == "ALLOW"]

    # 汇总
    order = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "W1", "W2", "W3", "A",
             "I1", "I2", "I3", "I4"]
    labels = {
        "E1": "new 操作符", "E2": "delete 操作符", "E3": "malloc/free 族",
        "E4": "reinterpret_cast", "E5": "const_cast", "E6": "C 风格指针转换",
        "E7": "void* 裸空指针", "W1": "裸指针声明候选(待裁决)",
        "W2": "C 数组参数", "W3": "C 风格函数指针", "A": "nn-allow 豁免",
        "I1": "smart ptr 使用", "I2": "make_unique 使用",
        "I3": "observer_ptr 使用", "I4": "std::span 使用",
    }
    print("=" * 78)
    print("neuralnet.cpp 现代 C++ 风格审查报告")
    print("=" * 78)
    for code in order:
        if code in rep.stats:
            tag = "ERROR" if code.startswith("E") else ("WARN " if code.startswith("W") else "info ")
            print(f"  [{tag}] {labels[code]:<24} {rep.stats[code]:>5}")
    print("-" * 78)
    if args.by_file:
        per: dict[str, dict[str, int]] = {}
        for f in rep.findings:
            if f.level in ("ERROR", "WARN"):
                per.setdefault(f.path, {}).setdefault(f.level, 0)
                per[f.path][f.level] += 1
        for path in sorted(per):
            d = per[path]
            print(f"  {path:<58} E:{d.get('ERROR', 0):>3}  W:{d.get('WARN', 0):>3}")
    if errors or not args.quiet:
        for f in errors + ([] if args.quiet else warns):
            print(f"  {f.level[0]} {f.code} {f.path}:{f.line}: {f.text[:110]}")
    print("-" * 78)
    print(f"  文件数: {len(seen)}   ERROR: {len(errors)}   WARN: {len(warns)}   豁免: {len(allows)}")
    verdict = "FAIL" if errors else ("FAIL(warn)" if (args.fail_on_warn and warns) else "PASS")
    print(f"  结果: {verdict}")
    return 1 if (errors or (args.fail_on_warn and warns)) else 0


if __name__ == "__main__":
    sys.exit(main())
