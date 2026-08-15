#!/usr/bin/env python3
"""
扫描代码库中所有 .cpp / .hpp 文件，
找出包含以下内容的文件和行号：
  - new / new[]            （手动分配）
  - delete / delete[]      （手动释放）
  - malloc / calloc / realloc / free  （C 风格内存管理）
  - 裸指针声明（非智能指针）
  - void* / reinterpret_cast / const_cast（危险类型转换）
  - C 风格强制转换 (type*)expr

输出格式：按文件分组，每个命中显示行号和对应代码。
"""

import os
import re
from collections import defaultdict

# ---------- 扫描目标目录 ----------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIRS = [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "include")]

# ---------- 匹配规则 ----------
# 每条规则: (category, description, compiled_regex)
RULES = [
    # ---- 手动内存管理 ----
    ("alloc",   "new",
        re.compile(r'\bnew\s+[A-Za-z_]\w*(\s*\[|\s*\()')),
    ("free",    "delete",
        re.compile(r'\bdelete\b')),
    ("alloc",   "malloc/calloc/realloc",
        re.compile(r'\b(malloc|calloc|realloc)\s*\(')),
    ("free",    "free()",
        re.compile(r'\bfree\s*\(')),

    # ---- 裸指针声明（排除智能指针和注释） ----
    ("ptr",     "raw pointer decl",
        re.compile(
            r'(?<!\w)'                       # 非标识符字符开头
            r'(?:const\s+)?'                 # 可选 const
            r'[A-Za-z_:<>]+'                 # 类型名
            r'(?:\s*<[^>]+>)?'               # 可选模板参数
            r'\s*\*'                          # 指针星号
            r'\s+[a-zA-Z_]\w*'               # 变量名
            r'(?:\s*[=\[;(,])'              # 后跟赋值/数组/分号/逗号/括号
            # 排除行注释和智能指针
        )),

    # ---- void* ----
    ("cast",   "void*",
        re.compile(r'\bvoid\s*\*')),

    # ---- reinterpret_cast / const_cast ----
    ("cast",   "reinterpret_cast",
        re.compile(r'\breinterpret_cast\s*<')),
    ("cast",   "const_cast",
        re.compile(r'\bconst_cast\s*<')),

    # ---- C 风格强制转换 (type*)expr ----
    ("cast",   "C-style cast",
        re.compile(
            r'\((?:const\s+)?[A-Za-z_]\w*(?:\s*\*+)+\s*\)\s*[A-Za-z_(]')),

    # ---- static_cast<...*> ----  (指针类型的 static_cast)
    ("cast",   "static_cast<ptr>",
        re.compile(r'\bstatic_cast\s*<[^>]*\*[^>]*>')),
]

# 行注释 / 字符串字面量 的简单过滤（避免误报）
SKIP_COMMENT = re.compile(r'^\s*//')
SKIP_STRING  = re.compile(r'^\s*"(?:[^"\\]|\\.)*"$')  # 整行就是字符串

# 智能指针名（排除这些 "指针声明" 误报）
SMART_PTR = re.compile(r'\b(unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b')


def scan_file(filepath):
    """扫描单个文件，返回 [(line_no, category, desc, line_text), ...]"""
    hits = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            # 跳过纯注释行 / 预处理包含
            if SKIP_COMMENT.match(line) or line.strip().startswith("#"):
                continue
            # 跳过智能指针所在行（降低 ptr 类误报）
            is_smart = bool(SMART_PTR.search(line))
            for cat, desc, rx in RULES:
                if rx.search(line):
                    # ptr 类规则：如果该行出现了智能指针则跳过
                    if cat == "ptr" and is_smart:
                        continue
                    hits.append((lineno, cat, desc, line.rstrip()))
                    break  # 一行只报一次（取最高优先级）
    return hits


def main():
    all_files = []
    for d in SRC_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in files:
                if name.endswith((".cpp", ".hpp")):
                    all_files.append(os.path.join(root, name))

    all_files.sort()

    results = {}        # filepath -> [hits]
    summary = defaultdict(list)  # category -> [filepath]

    for fp in all_files:
        hits = scan_file(fp)
        if hits:
            results[fp] = hits
            cats = {h[1] for h in hits}
            for c in cats:
                summary[c].append(fp)

    # ---- 输出报告 ----
    total_files = len(all_files)
    flagged = len(results)
    total_hits = sum(len(v) for v in results.values())

    print(f"扫描完成：共 {total_files} 个文件，{flagged} 个文件包含手动内存管理/裸指针，共 {total_hits} 处命中\n")

    if not results:
        print("✅ 未发现手动内存管理或裸指针，代码库非常干净！")
        return

    for fp in results:
        rel = os.path.relpath(fp, REPO_ROOT)
        print(f"{'=' * 80}")
        print(f"📄 {rel}  ({len(results[fp])} 处)")
        print(f"{'-' * 80}")
        for lineno, cat, desc, line_text in results[fp]:
            # 截断过长行
            display = line_text[:120] + (" …" if len(line_text) > 120 else "")
            print(f"  L{lineno:>4d}  [{cat:<6s}] {desc:<20s} | {display}")
        print()

    # ---- 汇总 ----
    print(f"{'=' * 80}")
    print("📊 汇总统计")
    print(f"{'-' * 80}")
    for cat in ["alloc", "free", "ptr", "cast"]:
        files_in_cat = summary.get(cat, [])
        if files_in_cat:
            print(f"  {cat:<8s}: {len(files_in_cat)} 个文件")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
