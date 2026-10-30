#!/usr/bin/env python3
"""doc_audit.py — 文档↔代码对齐审计 + 文档规范检查（只读，跨平台）

规范正文：docs/contributors/documentation-standard.md §5——本文档审计的**唯一实现**。
八类检查：A 文件/链接、B 已删符号残留、C 数字断言披露、D CLI 参数、E 头部模板/状态词/
索引登记、F nn:: 符号与测试名、G 代码示例清单（--report 生成人工复核报告）、
H 历史性记载禁令。

用法：
    python scripts/doc_audit.py             # 全量检查；有 FAIL 时 exit 1
    python scripts/doc_audit.py --report    # 另生成 build/doc_audit/code_examples_report.md
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 白名单（每项带来历注释，新增须说明）──────────────────────────────────

# [A] 历史文件：已被有意删除，历史文档提及属正常（勿报警）
HISTORICAL_FILES = [
    'compute_cuda_engine.hpp', 'compute_cuda_backend.hpp', 'cuda_kernels.h', 'cuda_kernels.cu',
    'expr_graph.hpp', 'cpu_emitter.hpp', 'algebra_expr.hpp', 'algebra_compute.hpp',
    'perf_smoke.cpp', 'mnist_common.hpp', 'train_bytebpe.py',
    'expr_fuse_test.cpp', 'expr_spec_test.cpp', 'matmul_fusion_test.cpp',
    'offload_primitive_test.cpp', 'offload_test.cpp', 'broadcast.comp',
    'compute_precision_engine.hpp',          # P-1 随 PrecisionEngine 下沉删除
    'compute_layer_zipt.hpp', 'domain_zipt.hpp',  # ZiPT 移除（分支 legacy/zipt）
    'ternary_deploy.hpp',                    # NNQT 独立部署格式，随库级清理（internal/23）删除
    '08-object-lifetimes.md',                # 2026-11 文档重构删除的转发页（真源 = 07-engine-ownership.md）
    'doc_align_audit.ps1',                   # 文档审计由 scripts/doc_audit.py 承担（口径见文档规范 §5）
]

# [B] 已删除符号（命中行若带"不存在/已删除"等否定语境则跳过）
DEAD_SYMBOLS = [
    'axpy_inplace', 'broadcast_row_inplace', 'broadcast_col_inplace', 'elementwise_unary',
    'elementwise_binary_scalar', 'elementwise_select_scalar_cond', 'offload_store', 'offload_load',
    'multiply_transposed_add_to', 'compute::apply', 'algebra_expr', 'algebra_compute',
    'set_offload_enabled', 'UnaryOp', 'CompareOp',
]
NEGATION_PAT = re.compile(r'不存在|已删除|已移除|没有|无此|已归档|归档于|历史')

# [D] 外部工具参数白名单（非本仓 CLI：ctest/cmake/glslc/torch 脚本等）
EXTERNAL_FLAGS = {
    '--test-dir', '--parallel', '--target-env', '--dtype', '--adam-eps', '--device',
    '--task', '--compress', '--list', '--image', '--list-backends', '--summary', '--include',
    '--preset', '--workflow', '--list-presets', '--output-on-failure',
    # git 自带选项（文档命令用）：git clang-format --diff / clone --recurse-submodules / tag --tags
    '--diff', '--recurse-submodules', '--tags',
}
# [D] 已在文档中明确标注"已移除"的历史参数
REMOVED_FLAGS = {
    '--cuda', '--tdr-retry', '--max-tdr-retries', '--memory-tokens', '--window',
    '--osc-guard', '--osc-window', '--osc-threshold',
    '--no-cache', '--test-file', '--loss-scope', '--export-deploy',
}
# [D] 文档已标注"计划中、尚未实现"的参数（实现后移出本表）
PLANNED_FLAGS = {'--stochastic-rounding', '--unit-scaling'}

# [F] 历史/已并入测试名（当时目标名，勿再报警）
HISTORICAL_TESTS = {
    'expr_fuse_test', 'expr_graph_test', 'expr_spec_test', 'matmul_fusion_test',
    'offload_primitive_test', 'zipt_test', 'zipt_gradcheck', 'zipt_smoke_test',
    'zipt_consistency_test', 'zipt_doc_test', 'ternary_deploy_test',
}

# [E] 头部模板豁免（文档规范 §2.2）：历史归档 / 随 tag 冻结 / 采集时快照
HEADER_EXEMPT = [
    re.compile(r'^research/'),              # research/ 有自己的四段式口径（research/README.md）
    re.compile(r'^docs/history/history\.md$'),
    re.compile(r'^docs/release-notes/v[\d.]+(-rc[\d.]+)?\.md$'),
    re.compile(r'^docs/benchmarks/.+\.md$'),
]
# [E] 状态词表（封闭集合）
VALID_STATUS = {'现行', '冻结', 'TODO', '转发页', '已归档'}


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, '/')


def walk(exts, skip_dirs={'build', '.git', 'extern', 'node_modules', '.venv', '__pycache__'}):
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for fn in files:
            if os.path.splitext(fn)[1] in exts:
                yield os.path.join(root, fn)


HISTORY_MD = None  # 在 main 里赋值


def load_docs():
    """审计对象：AGENTS.md / README.md / CONTRIBUTING.md + docs/**.md + research/**.md"""
    docs = [os.path.join(ROOT, x) for x in ('AGENTS.md', 'README.md', 'CONTRIBUTING.md')]
    for p in walk({'.md'}):
        r = rel(p)
        if r.startswith(('docs/', 'research/')):
            docs.append(p)
    return sorted(docs)


def strip_inline_code(line):
    return re.sub(r'`[^`]*`', '', line)


def read_lines(p):
    with open(p, encoding='utf-8') as f:
        return f.read().split('\n')


def load_code_text():
    parts = []
    for p in walk({'.hpp', '.cpp'}, skip_dirs={'build', '.git', 'node_modules', '.venv', '__pycache__'}):
        if rel(p).startswith(('include/', 'src/', 'tools/', 'extern/')):
            with open(p, encoding='utf-8', errors='replace') as f:
                parts.append(f.read())
    for p in walk({'.py'}):
        with open(p, encoding='utf-8', errors='replace') as f:
            parts.append(f.read())
    with open(os.path.join(ROOT, 'CMakeLists.txt'), encoding='utf-8') as f:
        parts.append(f.read())
    return '\n'.join(parts)


# ═══ [A] 文档引用的文件/链接目标存在 ═════════════════════════════════════

def check_A(docs):
    miss, hist, link_miss, link_checked = 0, 0, 0, 0
    print('=== [A] 文档引用但仓库中不存在的文件 / 链接目标 ===')
    # 全仓真实文件名索引（递归，含 extern/ 子仓头文件，与 ps1 口径一致）
    real_names = set()
    for p in walk({'.hpp', '.cpp', '.comp', '.py'},
                  skip_dirs={'build', '.git', 'node_modules', '.venv', '__pycache__'}):
        real_names.add(os.path.basename(p))
    for fn in os.listdir(ROOT):
        if os.path.isfile(os.path.join(ROOT, fn)):
            real_names.add(fn)
    # 构建生成物（AOT 融合管线产物，ps1 扫 build/generated；未构建机器上不存在，勿报）
    GENERATED_FILES = {'fused_registry.hpp'}
    # 目录/仓库名被反引号包裹（非文件引用）与外部项目文件引用
    NON_FILE_NAMES = {'neuralnet.cpp', 'go_train_cli.cpp'}
    for d in docs:
        if d == HISTORY_MD:
            continue  # 历史归档按设计记录已删符号/已删文件（ps1 同口径，整体豁免 A）
        hits = []
        lines = read_lines(d)
        for i, line in enumerate(lines):
            for m in re.finditer(r'`([A-Za-z0-9_]+\.(hpp|cpp|comp|py|md))`', line):
                name = m.group(1)
                if name.endswith('.md'):
                    continue  # md 引用走链接检查（裸文件名跨目录歧义大）
                if name not in real_names:
                    if name in GENERATED_FILES or name in NON_FILE_NAMES:
                        continue
                    if name in HISTORICAL_FILES:
                        hist += 1
                    else:
                        hits.append(f'{i+1}: `{name}`')
            # markdown 链接目标（剥行内代码防误报；外部/锚点跳过）
            text = strip_inline_code(line)
            for m in re.finditer(r'\]\(([^)\s]+)\)', text):
                t = m.group(1)
                if re.match(r'^(https?:|mailto:|codex:|#)', t):
                    continue
                target = t.split('#')[0]
                if not target:
                    continue
                link_checked += 1
                full = os.path.normpath(os.path.join(os.path.dirname(d), target))
                if not os.path.exists(full):
                    link_miss += 1
                    hits.append(f'{i+1}: {t}')
        if hits:
            miss += len(hits)
            shown = hits[:12]
            print(f'  -- {rel(d)}（{len(hits)} 处，示前 12）')
            for h in shown:
                print(f'    {h}')
    print(f'  [A] 可行动 {miss} 处（检查了 {link_checked} 个链接目标）；历史白名单跳过 {hist} 处\n')
    return miss


# ═══ [B] 已删除符号的文档残留 ═══════════════════════════════════════════

def check_B(docs):
    print('=== [B] 已删除符号的文档残留 ===')
    total = 0
    cache = {}
    for sym in DEAD_SYMBOLS:
        remains, neg = [], 0
        for d in docs:
            if d == HISTORY_MD:
                continue
            if d not in cache:
                cache[d] = read_lines(d)
            lines = cache[d]
            for i, line in enumerate(lines):
                if sym not in line:
                    continue
                win = ' '.join(lines[i:i + 3])
                if NEGATION_PAT.search(win):
                    neg += 1
                else:
                    remains.append(f'{rel(d)}:{i+1}: {line.strip()[:78]}')
        if remains:
            total += len(remains)
            print(f'  -- {sym}: {len(remains)} 处')
            for r in remains[:6]:
                print(f'     {r}')
        if neg:
            print(f'     （另有 {neg} 处为「该符号不存在/已删除」陈述，跳过）')
    print(f'  [B] 合计 {total} 处残留\n')
    return total


# ═══ [C] 数字型断言披露（WARN）═══════════════════════════════════════════

def check_C(docs):
    print('=== [C] 数字型断言（需人工与实测核对，WARN）===')
    n = 0
    pat = re.compile(r'\d+ 个 virtual|\d+ 个目标|ctest \d+|\d+/\d+ *(绿|全绿|通过|测试)|目标（\d+ 个）')
    for d in docs:
        if d == HISTORY_MD:
            continue
        for i, line in enumerate(read_lines(d)):
            if pat.search(line):
                n += 1
                if n <= 15:
                    print(f'  {rel(d)}:{i+1}: {line.strip()[:88]}')
    print(f'  [C] 共 {n} 处披露\n')
    return 0  # WARN 不阻塞


# ═══ [D] CLI 参数存在性 ══════════════════════════════════════════════════

def check_D(docs, code_text):
    print('=== [D] 文档引用但代码中不存在的 CLI 参数 ===')
    flags = {}
    for d in docs:
        if d == HISTORY_MD:
            continue
        for i, line in enumerate(read_lines(d)):
            for m in re.finditer(r'(?<![\w-])--[a-z][a-z0-9-]{2,}', line):
                flags.setdefault(m.group(0), []).append(f'{rel(d)}:{i+1}')
    miss = skipped = 0
    for f in sorted(flags):
        if f in code_text:
            continue
        if f in EXTERNAL_FLAGS or f in REMOVED_FLAGS or f in PLANNED_FLAGS:
            skipped += 1
            continue
        miss += 1
        print(f'  -- {f}   出现于: {", ".join(flags[f][:4])}')
    print(f'  [D] 找不到 {miss} 个；白名单跳过 {skipped} 个\n')
    return miss


# ═══ [E] 头部模板 + 状态词 + 索引登记 ════════════════════════════════════

def check_E(docs):
    print('=== [E] 头部模板 / 状态词 / 索引登记 ===')
    problems = 0
    indexed = set()   # 被索引提到的 docs/ 文件名
    index_files = [os.path.join(ROOT, 'docs/README.md')]
    for sub in ('development', 'internal', 'contributors', 'usage', 'introduction', 'release-notes'):
        p = os.path.join(ROOT, 'docs', sub, 'README.md')
        if os.path.isfile(p):
            index_files.append(p)
    for p in index_files:
        with open(p, encoding='utf-8') as f:
            text = f.read()
        for m in re.finditer(r'\(([A-Za-z0-9._/-]+\.md)\)', text):
            indexed.add(os.path.basename(m.group(1)))
        for m in re.finditer(r'[`"]([A-Za-z0-9._/-]+\.md)[`"]', text):
            indexed.add(os.path.basename(m.group(1)))

    for d in docs:
        r = rel(d)
        if any(pat.match(r) for pat in HEADER_EXEMPT):
            continue
        if not r.startswith('docs/'):
            continue  # 头部模板适用 docs/**（规范 §2）；根级入口文档豁免
        head = '\n'.join(read_lines(d)[:15])
        ok_reader = '**读者**' in head or '**本分册写给' in head
        ok_status = '**状态**' in head
        if not (ok_reader and ok_status):
            problems += 1
            print(f'  -- {r}: 头部模板缺要素（读者={"✓" if ok_reader else "✗"} 状态={"✓" if ok_status else "✗"}）')
        if ok_status:
            m = re.search(r'\*\*状态\*\*：\s*([^\s。（(]+)', head)
            if m and m.group(1) not in VALID_STATUS:
                problems += 1
                print(f'  -- {r}: 非法状态词「{m.group(1)}」（合法：{"/".join(sorted(VALID_STATUS))}）')
    # 索引登记：docs/**.md（除 README 自身）必须出现在分册 README 或总索引
    for d in docs:
        r = rel(d)
        if not r.startswith('docs/') or r == 'docs/README.md' or r.endswith('/README.md'):
            continue
        fn = os.path.basename(d)
        if fn not in indexed:
            problems += 1
            print(f'  -- {r}: 未登记进任何索引（docs/README.md 或分册 README）')
    print(f'  [E] 问题 {problems} 处\n')
    return problems


# ═══ [F] nn:: 符号 / 测试名存在性 ════════════════════════════════════════

def check_F(docs, code_text):
    print('=== [F] 文档引用但代码中不存在的 nn:: 符号 / 测试名 ===')
    syms, tests = {}, {}
    test_pat = re.compile(
        r'\b([a-z][a-z0-9_]{2,}_test|maxpool_gradcheck|conv2d_gradcheck|rmsnorm_gradcheck|'
        r'softmax_gradcheck|swiglu_gradcheck|attn_gradcheck|gpt_gradcheck|rapt_gradcheck|'
        r'f16_cpu_probe|f16_writeback_probe|mem_probe|layer_bench)\b')
    for d in docs:
        if d == HISTORY_MD:
            continue
        in_research = rel(d).startswith('research/')  # 研究报告引用其自有脚本符号，非 ctest 目标
        for i, line in enumerate(read_lines(d)):
            for m in re.finditer(r'nn::([A-Za-z_][A-Za-z0-9_]*)', line):
                syms.setdefault(m.group(1), []).append(f'{rel(d)}:{i+1}')
            if in_research:
                continue
            for m in test_pat.finditer(line):
                tests.setdefault(m.group(1), []).append(f'{rel(d)}:{i+1}')
    miss_s = 0
    for s in sorted(syms):
        if not re.search(r'\b' + re.escape(s) + r'\b', code_text):
            miss_s += 1
            print(f'  -- nn::{s}   出现于: {", ".join(syms[s][:3])}')
    with open(os.path.join(ROOT, 'CMakeLists.txt'), encoding='utf-8') as f:
        cm = f.read()
    real_src = {os.path.basename(p) for p in walk({'.cpp'})}
    extern_tests = set()
    for p in walk({'.cpp'}, skip_dirs={'build', '.git', 'node_modules', '.venv', '__pycache__'}):
        if '/tests/' in rel(p):
            extern_tests.add(os.path.basename(p)[:-4])
    miss_t = hist_t = 0
    for t in sorted(tests):
        if t + '.cpp' in real_src or t in extern_tests or t in cm:
            continue
        if t in HISTORICAL_TESTS:
            hist_t += 1
            continue
        miss_t += 1
        print(f'  -- 测试名 {t}   出现于: {", ".join(tests[t][:3])}')
    print(f'  [F] nn:: 符号找不到 {miss_s} 个；测试名可行动 {miss_t} 个（历史白名单跳过 {hist_t} 个）\n')
    return miss_s + miss_t


# ═══ [G] 代码示例清单（复核报告）═════════════════════════════════════════

FENCE_PAT = re.compile(r'^(`{3,})(\w+)[ \t]*(.*)$')


def extract_snippets(docs):
    """提取全部围栏代码块：(文件, 行号, 语言, 角色, 片段原文)。角色来自 info string。"""
    out = []
    for d in docs:
        lines = read_lines(d)
        i = 0
        while i < len(lines):
            m = FENCE_PAT.match(lines[i])
            if not m:
                i += 1
                continue
            fence, lang, rest = m.group(1), m.group(2), m.group(3)
            role = 'snippet'
            for token in rest.split():
                if token in ('example', 'snippet', 'cli'):
                    role = token
                elif token.startswith('api='):
                    role = 'example api'
            start = i + 1
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(fence):
                body.append(lines[i])
                i += 1
            out.append((rel(d), start, lang, role, '\n'.join(body)))
            i += 1
    return out


# ═══ [H] 历史性记载禁令（文档规范 §3.4）══════════════════════════════════

def check_H(docs):
    """history/ 之外不得出现历史性记载：commit hash 引用、"已在…修复"、日期+变更动作。"""
    print('=== [H] 历史性记载（history/ 之外应为当前事实叙事）===')
    pats = [
        (re.compile(r'commit [0-9a-f]{7,}'), 'commit hash 引用'),
        (re.compile(r'(已|刚)(经)?在.{0,12}修复'), '"已在…修复"叙事'),
        (re.compile(r'20[0-9][0-9]-[01][0-9]-[0-3][0-9] .{0,20}(修复|迁移到|迁入|迁至|重排|并入|改名|删除)'), '日期+变更动作叙事'),
    ]
    # 时点记录文档豁免（本身就是记录，按版本/采集时点冻结，文档规范 §3.4）
    exempt = re.compile(r'^docs/(history/|release-notes/v|benchmarks/)|^docs/contributors/release-requirements\.md$|^research/')
    total = 0
    for d in docs:
        r = rel(d)
        if exempt.match(r):
            continue
        for i, line in enumerate(read_lines(d)):
            if 'doc-audit: allow H' in line:
                continue
            for pat, label in pats:
                m = pat.search(line)
                if m:
                    total += 1
                    if total <= 20:
                        print(f'  -- {r}:{i+1} [{label}] {line.strip()[:78]}')
                    break
    if total > 20:
        print(f'  …（仅示前 20）')
    print(f'  [H] 合计 {total} 处（改写为当前事实或移入 history/history.md）\n')
    return total


def check_G(docs, report_path):
    print('=== [G] 代码示例清单 ===')
    snippets = extract_snippets(docs)
    # 违规判定：C++ 块声明了 api 角色 → 必须写成 "example api"；未声明 → 默认 snippet（允许）
    bad = 0
    report = ['# 代码示例复核报告（自动生成，勿手改）', '',
              f'共 {len(snippets)} 个代码块。**机器只核对符号存在性；逻辑正确性由人工复核**——',
              '改了 API/CLI 后跑 `python scripts/doc_audit.py --report` 并通读本文件。', '']
    by_file = {}
    for f, ln, lang, role, body in snippets:
        by_file.setdefault(f, []).append((ln, lang, role, body))
    for f in sorted(by_file):
        report.append(f'## {f}')
        report.append('')
        for ln, lang, role, body in by_file[f]:
            report.append(f'### {f}:{ln}  语言={lang}  角色={role}')
            # 抽取片段中的符号供人工对照
            api_refs = sorted(set(re.findall(r'\bnn::[A-Za-z_][A-Za-z0-9_]*', body)))
            cli_refs = sorted(set(re.findall(r'(?<![\w-])--[a-z][a-z0-9-]{2,}', body)))
            if api_refs:
                report.append(f'- 引用符号: {", ".join(api_refs)}')
            if cli_refs:
                report.append(f'- CLI 参数: {", ".join(cli_refs)}')
            report.append('')
            report.append('```' + lang)
            report.append(body)
            report.append('```')
            report.append('')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'  代码块 {len(snippets)} 个 → 复核报告 {rel(report_path)}（{len(by_file)} 个文件）')
    print(f'  [G] 违规 {bad} 处\n')
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true', help='生成代码示例复核报告')
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(ROOT, 'extern', 'base', 'include')):
        print('⚠ extern/ 子仓未初始化——nn:: 符号与子仓测试名检查不完整。先跑：git submodule update --init --recursive\n')
    docs = load_docs()
    global HISTORY_MD
    HISTORY_MD = os.path.join(ROOT, 'docs', 'history', 'history.md')
    code_text = load_code_text()
    print(f'审计对象：{len(docs)} 个文档（AGENTS/README/CONTRIBUTING + docs/** + research/**）\n')

    fails = 0
    fails += check_A(docs)
    fails += check_B(docs)
    check_C(docs)                      # WARN，不计入
    fails += check_D(docs, code_text)
    fails += check_E(docs)
    fails += check_F(docs, code_text)
    fails += check_G(docs, os.path.join(ROOT, 'build', 'doc_audit', 'code_examples_report.md'))
    fails += check_H(docs)

    if fails:
        print(f'❌ FAIL：{fails} 处需处理')
        return 1
    print('✅ 全绿（[C] 为披露项，不阻塞）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
