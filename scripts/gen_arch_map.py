#!/usr/bin/env python3
"""gen_arch_map.py — 从代码本身生成架构地图 docs/development/architecture-map.md

目的：解决"AI 辅助开发后，主导者本人不能完全看懂项目"的问题。
原则：地图必须由脚本从源码生成（永不腐烂），而不是手写（写的人不更新，读的人不敢信）。

用法：python3 scripts/gen_arch_map.py
输出：docs/development/architecture-map.md
建议：接入 pre-commit / CI（重新生成后 git diff --exit-code 校验一致性）。
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "include" / "neuralnet.cpp"          # header-only 库
SRC = ROOT / "src"                                 # 可执行入口 + 测试
OUT = ROOT / "docs" / "development" / "architecture-map.md"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)
COMMENT_BLOCK_RE = re.compile(r'/\*.*?\*/', re.S)
LINE_COMMENT_RE = re.compile(r'//[^\n]*')
# 顶层类型定义：class/struct + 名字 + 紧跟 { （排除前置声明），允许前导 template<>
TYPE_DEF_RE = re.compile(
    r'^(?:template\s*<[^{;]*?>[\s\n]*)?(?:export\s+)?(class|struct)\s+([A-Za-z_]\w*)\s*(?:final\s*)?[{]',
    re.M)
NNNS_RE = re.compile(r'^\s*namespace\s+nn\b', re.M)
# 文件头 purpose 注释：匹配 `// ── xxx.hpp — 说明`（前缀允许任意个划线字符）
PURPOSE_RE = re.compile(
    r'^//[─—–\s]*[\w./-]+\.(?:hpp|h|cpp)\s*[—\-–]+\s*(.+?)\s*[─—–]*\s*$', re.M)

LAYER_PREFIX = "compute_layer_"


def strip_comments(text: str) -> str:
    text = COMMENT_BLOCK_RE.sub(' ', text)
    return LINE_COMMENT_RE.sub('', text)


def lib_files():
    return sorted(LIB.rglob("*.hpp")) + sorted(LIB.rglob("*.h"))


def short(p: Path) -> str:
    """相对仓库根的展示路径"""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def includes_of(p: Path) -> set[str]:
    """文件内 include 的、属于本仓库内的头（解析为绝对路径集合）"""
    text = p.read_text(encoding="utf-8", errors="replace")
    deps = set()
    for m in INCLUDE_RE.finditer(text):
        inc = m.group(1)
        # 只关心库内依赖：neuralnet.cpp/xxx.hpp 或同目录相对 include
        cand = None
        if inc.startswith("neuralnet.cpp/"):
            cand = LIB / inc[len("neuralnet.cpp/"):]
        else:
            cand = p.parent / inc
        if cand.is_file():
            deps.add(short(cand.resolve()))
    return deps


def purpose_of(p: Path) -> str:
    """从文件头注释提取一行用途说明。

    支持两种既有惯例：
      A. `// ── xxx.hpp — 说明`（同行）
      B. `// ── xxx.hpp ───…` + 下一行 `// 说明`（划线收尾，说明在第二行）
    两者都取不到时返回空串（地图会如实暴露缺失，倒逼补齐文件头）。
    """
    text = p.read_text(encoding="utf-8", errors="replace")[:4000]
    m = PURPOSE_RE.search(text)
    if m and m.group(1):
        s = m.group(1).strip()
        return (s[:60] + "…") if len(s) > 60 else s
    # 惯例 B：标题行以划线收尾，取其后第一条非空注释正文行
    head = re.search(r'^//[─—–\s]*[\w./-]+\.(?:hpp|h|cpp)\s*[─—–]+\s*$', text, re.M)
    if head:
        rest = text[head.end():]
        for line in rest.splitlines()[:3]:
            t = line.strip()
            if not t.startswith("//"):
                break
            t = t.lstrip("/").strip()
            if t and not t.startswith("─") and not t.startswith("-"):
                return (t[:60] + "…") if len(t) > 60 else t
    return ""


def public_types(p: Path) -> list[str]:
    raw = p.read_text(encoding="utf-8", errors="replace")
    if not NNNS_RE.search(raw):
        return []
    text = strip_comments(raw)
    names = [m.group(2) for m in TYPE_DEF_RE.finditer(text)]
    # 去重保序，过滤 Impl/detail
    seen, out = set(), []
    for n in names:
        if n.endswith("Impl") or n == "detail" or n.startswith("_"):
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:12]  # 每文件最多列 12 个代表类型


def layer_groups(files):
    """把 compute_layer_* / backend/* / cli/* 归组，其余按前缀聚类"""
    groups = defaultdict(list)
    for f in files:
        name = f.name
        rel = short(f)
        if "backend/" in rel:
            g = "GPU/Vulkan 后端"
        elif "cli/" in rel:
            g = "CLI 应用层"
        elif name.startswith(LAYER_PREFIX):
            g = "网络层（compute_layer_*）"
        elif name.startswith("expr_"):
            g = "表达式 DSL / 算子融合"
        elif name.startswith("compute_"):
            g = "计算引擎核心"
        elif name.startswith("model_"):
            g = "模型规格与序列化"
        elif name.startswith("domain_"):
            g = "领域组件（GPT/tokenizer/dataset）"
        elif name.startswith("core_") or name.startswith("precision"):
            g = "基础设施（config/errors/tensor）"
        else:
            g = "其他"
        groups[g].append(f)
    return groups


def main():
    files = [f for f in lib_files() if f.is_file()]
    lines = []
    w = lines.append

    w("# 架构地图（自动生成）")
    w("")
    w("> ⚠️ 本文件由 `scripts/gen_arch_map.py` 从源码生成，**请勿手工编辑**。")
    w("> 重新生成：`python3 scripts/gen_arch_map.py`。")
    w("> 用途：人类主导者与 AI 建立共享心智模型的**第一入口**——先读本页，再按需下钻单文件。")
    w("")

    # ---- 1. 规模总览 ----
    total_loc = sum(len(f.read_text(encoding='utf-8', errors='replace').splitlines()) for f in files)
    w("## 1. 规模总览")
    w("")
    w(f"- 库头文件：**{len(files)} 个**，共 **{total_loc:,} 行**（header-only）")
    apps = sorted(SRC.glob("*.cpp"))
    w(f"- 可执行入口/测试：`src/` 下 {len(apps)} 个 .cpp")
    w("- 阅读顺序建议：`nn.hpp`（聚合入口）→ §2 分层 → §3 依赖图 → §4 逐文件清单")
    w("")

    # ---- 2. 分层视图 ----
    w("## 2. 分层视图（按文件名前缀自动归类）")
    w("")
    order = ["基础设施（config/errors/tensor）", "计算引擎核心", "表达式 DSL / 算子融合",
             "网络层（compute_layer_*）", "GPU/Vulkan 后端", "模型规格与序列化",
             "领域组件（GPT/tokenizer/dataset）", "CLI 应用层", "其他"]
    groups = layer_groups(files)
    w("| 层 | 文件数 | 行数 | 代表文件 |")
    w("|---|---|---|---|")
    for g in order:
        fs = groups.get(g, [])
        if not fs:
            continue
        loc = sum(len(f.read_text(encoding='utf-8', errors='replace').splitlines()) for f in fs)
        rep = ", ".join(sorted(fs, key=lambda x: -len(open(x, errors='replace').readlines()))[0].stem for _ in [0]) if False else ""
        tops = sorted(fs, key=lambda f: -len(f.read_text(encoding='utf-8', errors='replace').splitlines()))[:3]
        w(f"| {g} | {len(fs)} | {loc:,} | {', '.join(t.name for t in tops)} |")
    w("")

    # ---- 3. 内部依赖图（Mermaid）----
    w("## 3. 内部依赖图（#include 关系，仅库内边）")
    w("")
    w("```mermaid")
    w("graph LR")
    dep_edges = set()
    node_deg = defaultdict(int)
    for f in files:
        src_node = f.name
        for d in includes_of(f):
            dst = Path(d).name
            if dst != src_node:
                dep_edges.add((src_node, dst))
    for a, b in sorted(dep_edges):
        node_deg[a] += 1
        node_deg[b] += 1
    # 只画扇入+扇出 >= 2 的核心节点，避免 63 节点毛球；叶子注明省略数
    core = {n for n, d in node_deg.items() if d >= 2}
    shown = 0
    for a, b in sorted(dep_edges):
        if a in core and b in core:
            w(f"    {a.replace('.hpp','')} --> {b.replace('.hpp','')}")
            shown += 1
    hidden = len(dep_edges) - shown
    w(f"    %% 另有 {hidden} 条叶子/外围边未画出（阈值：度数≥2）")
    w("```")
    w("")
    w("完整边清单（`源 依赖 目标`，即 src include dst）：")
    w("")
    w("```text")
    for a, b in sorted(dep_edges):
        w(f"{a}  -->  {b}")
    w("```")
    w("")

    # ---- 4. 逐文件清单 ----
    w("## 4. 逐文件清单（用途 + 公开类型 + 直接依赖）")
    w("")
    w("| 文件 | 行数 | 用途（取自文件头注释） | 公开类型（前12） | 直接依赖（库内） |")
    w("|---|---|---|---|---|")
    for f in files:
        loc = len(f.read_text(encoding='utf-8', errors='replace').splitlines())
        purp = purpose_of(f) or "—"
        types = ", ".join(public_types(f)) or "—"
        deps = ", ".join(sorted(Path(d).name for d in includes_of(f))) or "—"
        w(f"| `{short(f)}` | {loc} | {purp} | {types} | {deps} |")
    w("")

    # ---- 5. src 入口清单 ----
    w("## 5. `src/` 入口与测试")
    w("")
    tests, progs = [], []
    for f in apps:
        (tests if "_test" in f.name or "gradcheck" in f.name else progs).append(f)
    w(f"- 测试/验证程序：{len(tests)} 个；正式入口：{len(progs)} 个")
    w("")
    w("| 文件 | 行数 | 说明（库内 include 首项推断用途域） |")
    w("|---|---|---|")
    for f in sorted(apps):
        loc = len(f.read_text(encoding='utf-8', errors='replace').splitlines())
        deps = includes_of(f)
        hint = Path(next(iter(sorted(deps)))).name if deps else "—"
        w(f"| `{short(f)}` | {loc} | 依赖 {hint} 等 {len(deps)} 项 |")
    w("")

    w("---")
    w("*生成时间戳见 git commit；如与源码不一致，重跑 `python3 scripts/gen_arch_map.py`。*")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else None
    new = "\n".join(lines) + "\n"
    if old != new:
        OUT.write_text(new, encoding="utf-8")
        print(f"[gen_arch_map] 已写入 {short(OUT)}（{len(new.splitlines())} 行）")
    else:
        print(f"[gen_arch_map] {short(OUT)} 无变化")


if __name__ == "__main__":
    sys.exit(main())
