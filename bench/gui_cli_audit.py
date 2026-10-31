# -*- coding: utf-8 -*-
"""GUI / CLI 参数一致性审计（可复现门禁）。

复现：`python bench/gui_cli_audit.py`（需先构建 build/ 下的可执行文件）

四节：
  [1] GUI → CLI flag：把每个 Tab 可能发出的完整 kwargs 喂给对应 Controller，
      提取生成的 `--flag`，与该可执行文件 `--help` 的 flag 集合求差集。
      → 抓"GUI 会发出的、CLI 不认识的参数"（多余项 / flag 名写错）。
  [2] 幽灵选项：`--help` 声明的 flag 是否真有解析分支（源码里非帮助行出现）。
      → 抓"帮助里有、实际没解析"（用户按帮助用即报未知参数）。
  [3] gui.py 静态属性：每个类里 `self.X` 的读取是否都有赋值（含模块内祖先类）。
      → 抓"控件删了、引用还在"这类只在运行时才炸的错误（无需起 GUI / 装 customtkinter）。
  [4] 优化器名单一致性（RV-04）：CLI `--help` 声明的名单（唯一真源 =
      `compute_optimizer.hpp` 的 `kOptimizerTable`）与 gui.py 的兜底名单逐项比对。
      → 抓"改了表忘了改 GUI"（GUI 的选项运行时从 CLI 派生，兜底只在未构建时生效）。
  [5] 声明式字段表（P2-6）：`gui.py` 各 Tab 的 `fields()` 里声明的**每个**
      kwargs 键，喂给对应 Controller 后必须真的产生影响（生成 flag 或位置参数）。
      → 抓"表里加了一行、控制器不认"（表单是死的）；与 [1]（手写场景 → flag 存在性）
      互补：[1] 验"发出去的 flag 都存在"，[5] 验"表里的每一行都发得出去"。
  [6] 断点续训（P1-2）：`.nnckpt` 的 A/B 类参数禁列（GUI 集合 ↔ CLI 表 ↔ 收集行为）。
  [7] GUI 多轮对话闭环（P0-5 尾项）：`ChatSession` 每轮把**全部历史**编成
      `--turn <role>:<text>`（第 N 轮的参数以第 N-1 轮的参数为前缀 ⇒ 上下文不丢），
      且 `GptInferTab._on_send` 确实把它交给 CLI（静态查"_on_send 里没写历史"）。

退出码非 0 = 存在可行动问题。历史/有意为之的差异在此显式登记为白名单。
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cli_controllers import (  # noqa: E402
    MnistTrainController, MnistInferController,
    TokenizerTrainController, TokenizerInferController,
    GptTrainController, GptInferController,
    ChatSession, DEFAULT_SYSTEM_PROMPT,
)

# flag 词法：允许 `_`（唯一的带下划线 flag 是精度预设 `--t1_58`，见
# include/neuralnet.cpp/precision.hpp profile_t1_58()）。不加 `_` 时
# `--t1_58` 会被截成 `--t1`，在 [1]/[2] 两节里误报。
FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9_-]*")
# 统一帮助排版器（include/neuralnet.cpp/cli/cli_help.hpp）的调用行形态：
#   help.opt("--epochs <n>", "...")
# 帮助内容只出现在这种行里（而非 `<<` 字符串链），审计据此把帮助行排除在
# "解析分支"之外——否则帮助里声明的 flag 会被误当成已解析，幽灵选项检测失效。
# 见 cli_help.hpp 文件头的审计约定：每条 help.<方法>() 必须落在同一物理行。
HELP_CALL_RE = re.compile(r"\bhelp\s*\.\s*(usage|text|section|note|item|opt|flush)\s*\(")
EXES = ["mnist_train", "mnist_infer", "text_train", "text_infer",
        "tokenizer_train", "tokenizer_infer"]

# ── 有意不暴露在 GUI 的 CLI 参数 ──
#   --interactive : stdin 交互外壳；GUI 的多轮走**无状态**的 --turn 协议
#                   （每轮把整段历史交回 CLI，见 ChatSession / [7]），
#                   不再需要套一个 stdin 循环（进程内的历史 GUI 也拿不到）
#   --t1_58       : 三值预设的**别名**（等价 --precision-param t1_58）；GUI 的
#                   "精度预设"下拉填的是 precision_param 字段，三条命令在
#                   docs/internal/21 §4.10 里已声明等价，故此处不算缺口
INTENTIONAL_GAPS = {"--interactive", "--t1_58"}


def exe_path(exe: str) -> Path:
    """定位 build/ 下的可执行文件（跨平台：Windows 带 .exe，Linux/macOS 不带）。

    历史上这里写死 `build/<exe>.exe`，在 Linux 上直接 FileNotFoundError ——
    本门禁因此只能在 Windows 跑。按"先 .exe 后无后缀"的顺序探测。
    `CMakePresets.json` 的构建把产物放在 `build/<preset>/` 下，故再遍历一层 preset 目录
    （`--help` 文案与构建类型无关，取哪个 preset 都不影响本门禁的判定）。
    """
    names = (f"{exe}.exe", exe)
    cands = [ROOT / "build" / n for n in names]
    for sub in sorted((ROOT / "build").glob("*")):
        if sub.is_dir():
            cands += [sub / n for n in names]
    for cand in cands:
        if cand.exists():
            return cand
    return ROOT / "build" / f"{exe}.exe"


def cli_help_flags(exe: str) -> set[str]:
    out = subprocess.run([str(exe_path(exe)), "--help"],
                         capture_output=True, text=True, errors="replace")
    flags = set(FLAG_RE.findall(out.stdout + out.stderr))
    return flags - {"--help"}


# ══════════════════════════════════════════════════════════════════════
# [1] GUI → CLI flag
# ══════════════════════════════════════════════════════════════════════
BASE_PREC = dict(precision_param="f32", precision_compute="f32",
                 precision_stable="f32", precision_optimizer="f32")

CASES: list = []

# MNIST 训练：三个架构各覆盖自己的专属参数
CASES.append(("mnist_train", MnistTrainController(), dict(
    arch="transformer", dataset="datasets/mnist_data", save="m.bin", resume="r.bin",
    epochs=10, lr=0.001, batch_size=64, optimizer="adam", weight_decay=0.01,
    gpu=True, max_samples=0, shuffle_steps="true", lr_schedule="cosine",
    min_lr=1e-6, warmup_epochs=1, lr_per_epoch="0.01,0.001",
    norm="layernorm", norm_place="final",
    d_model=64, num_heads=4, num_layers=2, d_ff=128, patch_size=7,
    eval_samples=200, **BASE_PREC)))
CASES.append(("mnist_train", MnistTrainController(), dict(
    arch="mlp", layer_dims="784,512,10", norm="layernorm", shuffle_steps="false")))
CASES.append(("mnist_train", MnistTrainController(), dict(
    arch="cnn", cnn_channels="6,16", cnn_kernels="5,5", cnn_pool=2, cnn_fc="120,10",
    norm="rmsnorm", norm_place="both")))
CASES.append(("mnist_train", MnistTrainController(), dict(arch="mlp", f16=True)))
CASES.append(("mnist_train", MnistTrainController(), dict(arch="mlp", t1_58=True)))

CASES.append(("mnist_infer", MnistInferController(), dict(
    model="m.bin", input="x.csv", topk=3, show_pixels=True, gpu=True)))

CASES.append(("tokenizer_train", TokenizerTrainController(), dict(
    text_file="t.txt", tokenizer="charbpe", output="v.json", vocab_size=5000,
    min_freq=2, threads=0)))

CASES.append(("tokenizer_infer", TokenizerInferController(), dict(
    vocab="v.json", encode="hi", decode="1,2", encode_file="f.txt",
    text_file="t.txt", top=10, threads=0, show_bytes=True)))

# GPT 训练（词表/loss 掩码/test 子集随 .nndataset，19 号设计后不再有独立选项）
GPT_COMMON = dict(
    save="g.bin", epochs=10, batch_size=32,
    accum_steps=1, seq_len=256, stride=0, optimizer="adam", weight_decay=0.01,
    gpu=True, lr=0.001, lr_schedule="cosine", min_lr=1e-6, warmup_epochs=1,
    d_model=128, num_heads=4, num_layers=4, d_ff=512,
    flush_interval=1, checkpoint_every=1, activation_offload=True,
    log_interval=50, save_interval=100, grad_log=True, **BASE_PREC)
CASES.append(("text_train", GptTrainController(), dict(
    model="gpt", positional_encoding="alibi", activation="swiglu", norm="rmsnorm",
    **GPT_COMMON)))
CASES.append(("text_train", GptTrainController(), dict(
    **{**GPT_COMMON,
       "model": "rapt", "resume": "r.bin", "resume_epoch": 1, "resume_step": 5,
       "lr_schedule": "step_cosine", "warmup_steps": 10, "max_norm": 1.0,
       "beta1": 0.9, "beta2": 0.95,
       "lr_per_epoch": "0.001"})))
CASES.append(("text_train", GptTrainController(), dict(
    model="gpt", text_file="t.txt", f16=True, max_steps=5)))
CASES.append(("text_infer", GptInferController(), dict(
    model="g.bin", vocab="v.json", prompt="hello", max_tokens=200,
    temperature=1.0, top_k=40, top_p=0.9, repeat_penalty=1.2,
    repeat_window=64, seed=7, gpu=True, show_tokens=True)))
# 多轮对话（P0-5 尾项）：turns = 历史逐轮 + 本轮输入，回复走 --reply-out。
# 与 gui.py 的 ChatSession.ask() 同形（CLI 侧校验：user 起头、交替、末条 user）。
CASES.append(("text_infer", GptInferController(), dict(
    model="g.bin", vocab="v.json", max_tokens=50, system=DEFAULT_SYSTEM_PROMPT,
    turns=["user:第一问", "assistant:第一答", "user:第二问"],
    reply_out="/tmp/nn_chat_reply.txt")))


def audit_gui_to_cli() -> int:
    print("=== [1] GUI → CLI flag（多余项 / 名字写错）===")
    help_cache: dict[str, set[str]] = {}
    emitted: dict[str, set[str]] = {}
    bad = 0
    for exe, ctrl, kw in CASES:
        help_cache.setdefault(exe, cli_help_flags(exe))
        cmd = ctrl._build_command(**kw)
        flags = [a for a in cmd[1:] if a.startswith("--")]
        emitted.setdefault(exe, set()).update(flags)
        unknown = [f for f in flags if f not in help_cache[exe]]
        print(f"  [{'BAD' if unknown else 'OK '}] {exe:17s} flags={len(flags)} unknown={unknown or '无'}")
        if unknown:
            print("        cmd:", " ".join(cmd))
            bad += 1

    print("\n  ── 反向：CLI 有而 GUI 未暴露（缺失项；--interactive 为有意不暴露）──")
    for exe in sorted(emitted):
        missing = sorted(help_cache[exe] - emitted[exe])
        unexpected = [m for m in missing if m not in INTENTIONAL_GAPS]
        tag = "BAD" if unexpected else "OK "
        print(f"  [{tag}] {exe:17s} 缺失={missing or '无'}")
        bad += len(unexpected)
    return bad


# ══════════════════════════════════════════════════════════════════════
# [2] 幽灵选项：--help 声明 vs 解析分支
# ══════════════════════════════════════════════════════════════════════
def audit_phantom_options() -> int:
    print("\n=== [2] 幽灵选项（帮助声明了、解析分支不存在）===")
    cli_dir = ROOT / "include" / "neuralnet.cpp" / "cli"
    shared: list[tuple[Path, str]] = []
    for p in cli_dir.glob("*.hpp"):
        shared += [(p, ln) for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()]

    bad = 0
    for exe in EXES:
        cpp = ROOT / "src" / f"{exe}.cpp"
        lines = [(cpp, ln) for ln in cpp.read_text(encoding="utf-8", errors="replace").splitlines()]
        real: set[str] = set()
        for _, ln in lines + shared:
            # 帮助行以 `<<` 续接（旧式）或 `help.<方法>(...)`（统一排版器）；
            # 解析行不含这两者 → 据此区分
            if "<<" in ln or HELP_CALL_RE.search(ln) or ln.lstrip().startswith("//"):
                continue
            real |= set(FLAG_RE.findall(ln))
        phantom = sorted(cli_help_flags(exe) - real)
        print(f"  [{'BAD' if phantom else 'OK '}] {exe:16s} 幽灵选项={phantom or '无'}")
        bad += len(phantom)
    return bad


# ══════════════════════════════════════════════════════════════════════
# [3] gui.py 静态属性（self.X 读取是否都有赋值）
# ══════════════════════════════════════════════════════════════════════
BASE_ATTRS = {
    "controller", "params_frame", "output", "ctrl", "run_btn", "stop_btn",
    "status_lbl", "_running", "tabview",
    "master", "tk", "after", "after_cancel", "pack", "grid", "grid_remove",
    "grid_propagate", "grid_rowconfigure", "grid_columnconfigure",
    "destroy", "protocol", "title", "geometry", "minsize", "configure",
    "winfo_children", "winfo_width", "winfo_height", "bind", "update_idletasks",
}


def _own_names(node: ast.ClassDef) -> set[str]:
    out: set[str] = set()
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(sub.name)
        elif isinstance(sub, ast.Assign):
            for t in sub.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            out.add(sub.target.id)
    return out


def audit_gui_attrs() -> int:
    print("\n=== [3] gui.py 静态属性（self.X 读取是否都有赋值）===")
    tree = ast.parse((ROOT / "gui.py").read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    own = {c.name: _own_names(c) for c in classes}
    bases = {c.name: [b.id for b in c.bases if isinstance(b, ast.Name)] for c in classes}

    def inherited(name: str, seen: set | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in own:
            return set()
        seen.add(name)
        out = set(own[name])
        for b in bases.get(name, []):
            out |= inherited(b, seen)
        return out

    bad = 0
    for cls in classes:
        have = BASE_ATTRS | inherited(cls.name)
        for sub in ast.walk(cls):
            if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    have.add(sub.attr)
        for sub in ast.walk(cls):
            if (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load)
                    and isinstance(sub.value, ast.Name) and sub.value.id == "self"
                    and sub.attr not in have):
                print(f"  [BAD] {cls.name}:{sub.lineno} 读取 self.{sub.attr} 但从未赋值")
                bad += 1
    print(f"  类数={len(classes)} 未定义自读属性={bad}")
    return bad


# ══════════════════════════════════════════════════════════════════════
# [4] 优化器名单一致性（RV-04）
# ══════════════════════════════════════════════════════════════════════
# 名单的唯一真源 = include/neuralnet.cpp/compute_optimizer.hpp 的 kOptimizerTable：
#   · CLI --help 的 --optimizer 描述由 nn::optimizer_names("/") 生成；
#   · gui.py 的优化器下拉项已改为运行时从 CLI --help 解析（optimizer_options()），
#     仅在取不到可执行文件时退回 _OPTIMIZER_FALLBACK。
# 本节断言：CLI 确实声明了名单，且退回名单与 CLI 声明逐项一致 —— 兜底只在
# "未构建"时生效，但一旦漂移就会在门禁里暴露（否则它是最容易腐坏的一份）。
OPTIMIZER_LIST_RE = re.compile(r"优化器:([a-z0-9_/]+)")


def cli_optimizer_names(exe: str) -> list[str]:
    """CLI --help 声明的优化器名单（与 gui.py::parse_optimizer_options 同口径）。
    帮助排版器按宽度折行，故先把文本里的空白全部去掉再匹配
    `优化器:<a/b/c>`；否则跨行的列表会被截断成前缀。
    """
    path = exe_path(exe)
    if not path.exists():
        return []
    out = subprocess.run([str(path), "--help"], capture_output=True,
                         text=True, errors="replace")
    flat = re.sub(r"\s+", "", out.stdout + out.stderr)
    m = OPTIMIZER_LIST_RE.search(flat)
    return m.group(1).split("/") if m else []


def gui_fallback_optimizers() -> list[str]:
    """gui.py 的兜底名单（ast 解析；不 import gui.py —— 它需要 customtkinter）。"""
    tree = ast.parse((ROOT / "gui.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_OPTIMIZER_FALLBACK":
                    return [e.value for e in node.value.elts]
    return []


def audit_optimizer_names() -> int:
    print("\n=== [4] 优化器名单（CLI --help ↔ gui.py 兜底名单）===")
    fallback = gui_fallback_optimizers()
    if not fallback:
        print("  [BAD] gui.py 未找到 _OPTIMIZER_FALLBACK")
        return 1
    bad = 0
    for exe in ("mnist_train", "text_train"):
        cli = cli_optimizer_names(exe)
        if not cli:
            print(f"  [SKIP] {exe}: build/{exe}(.exe) 不存在或 --help 未声明名单")
            continue
        if cli != fallback:
            print(f"  [BAD] {exe}: CLI={cli} != gui.py 兜底={fallback}"
                  "（改 kOptimizerTable 后须同步 gui.py 的 _OPTIMIZER_FALLBACK）")
            bad += 1
        else:
            print(f"  [OK ] {exe:16s} 名单={cli}")
    return bad


# ══════════════════════════════════════════════════════════════════════
# [5] 声明式字段表 ↔ Controller（P2-6）
# ══════════════════════════════════════════════════════════════════════
# 逐 Tab 的 (可执行文件, gui.py 里的 Tab 类, Controller)：与 GUI 的装配一致。
TABLE_TABS = [
    ("mnist_train", "MnistTrainTab", MnistTrainController),
    ("mnist_infer", "MnistInferTab", MnistInferController),
    ("tokenizer_train", "TokenizerTrainTab", TokenizerTrainController),
    ("tokenizer_infer", "TokenizerInferTab", TokenizerInferController),
    ("text_train", "GptTrainTab", GptTrainController),
    ("text_infer", "GptInferTab", GptInferController),
]


def _load_gui_tables():
    """无 customtkinter 依赖地导入 gui.py（桩模块；与 gui_draw_logic_audit 同款）"""
    import types

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

        def __mro_entries__(self, bases):
            return (object,)

    mod = types.ModuleType("customtkinter")
    mod.__getattr__ = lambda name: _Any()
    sys.modules.setdefault("customtkinter", mod)
    import gui  # noqa: E402   （桩就位后再导入）
    return gui


def _probe_value(field):
    """给字段造一个"控制器认得出"的探测值（类型按 cast 定）"""
    if field.kind == "check":
        return True
    if field.kind == "option":
        return field.default
    if field.cast is int:
        return "1"
    if field.cast is float:
        return "0.5"
    return "probe"


def audit_field_tables() -> int:
    """表里声明的每个 kwargs 键都必须真的产生影响（否则这一行表单是死的）。

    判据：`_build_command(**{key: probe})` 与 `_build_command()` 不同（生成 flag 或
    位置参数）。虚拟项（`_v()`）也一并验 —— 它们声明的是"由代码后处理发出的键"。
    """
    print("\n=== [5] 声明式字段表 ↔ Controller（P2-6：表里每一行都发得出去）===")
    try:
        gui = _load_gui_tables()
    except Exception as exc:   # pragma: no cover
        print(f"  [BAD] 导入 gui.py 失败: {exc}")
        return 1
    bad = 0
    for exe, tab_name, ctrl_cls in TABLE_TABS:
        tab = getattr(gui, tab_name, None)
        if tab is None or not hasattr(tab, "fields"):
            print(f"  [BAD] gui.{tab_name} 没有声明式字段表")
            bad += 1
            continue
        ctrl = ctrl_cls()
        base = ctrl._build_command()
        keys, dead = 0, []
        for field in tab.fields():
            if field.kind == "label" or field.ui_only:
                continue      # label 无参数；ui_only = 由后处理映射成别的键
            keys += 1
            try:
                cmd = ctrl._build_command(**{field.key: _probe_value(field)})
            except Exception as exc:
                dead.append(f"{field.key}(异常 {exc})")
                continue
            if cmd == base:
                dead.append(field.key)
        if dead:
            print(f"  [BAD] {exe:17s} 表里声明但控制器不认/无效果: {dead}")
            bad += len(dead)
        else:
            print(f"  [OK ] {exe:17s} {keys} 个键全部有效")
    return bad


def audit_ckpt_resume_filter() -> int:
    """[6] 断点续训（.nnckpt）时 GUI 不得发出 A/B 类参数（P1-2）。

    三点：
      ① gui.py 的禁列集合 ↔ CLI 的 `kForbiddenResumeFlags`（**唯一真源**，直接读
         `cli/text_train_config.hpp`；键 → 旗标的映射用字段的 arg/name，与表格一致）；
      ② 恢复路径指向 **Checkpoint 角色**（v7+ role=1）时 `collect_args()` 不发任何
         禁列键（否则 CLI 会按 P1-2 的禁列规则报错）；
      ③ 指向 Model 角色（warm start，CLI 全权）时**照发** —— 防止把 SFT 的配方面
         一起砍掉。
    """
    print("\n=== [6] 断点续训禁列：gui.py ↔ CLI 表 ↔ 收集行为（P1-2）===")
    import re
    import tempfile

    header = (ROOT / "include/neuralnet.cpp/cli/text_train_config.hpp").read_text(
        encoding="utf-8")
    m = re.search(r"kForbiddenResumeFlags\[\]\s*=\s*\{(.*?)\};", header, re.S)
    if not m:
        print("  [BAD] 头文件里找不到 kForbiddenResumeFlags")
        return 1
    cli_flags = set(re.findall(r'"(--[a-z0-9_\-]+)"', m.group(1)))
    try:
        gui = _load_gui_tables()
    except Exception as exc:   # pragma: no cover
        print(f"  [BAD] 导入 gui.py 失败: {exc}")
        return 1

    # 例外（其余 = 键名去 `--` 再把 `_` 换成 `-`）：两个字段有 arg= 覆盖；
    # `--t1_58` 的flag 本身带下划线（保留下划线是 CLI 的既有拼写）。
    arg_overrides = {"model_type": "--model", "norm_type": "--norm", "t1_58": "--t1_58"}
    gui_flags = {arg_overrides.get(k, "--" + k.replace("_", "-"))
                 for k in gui.CKPT_RESUME_FORBIDDEN_KEYS}
    bad = 0
    only_gui, only_cli = gui_flags - cli_flags, cli_flags - gui_flags
    if only_gui or only_cli:
        print(f"  [BAD] 禁列集合与 CLI 不一致：仅 gui {sorted(only_gui)}；"
              f"仅 CLI {sorted(only_cli)}")
        bad += len(only_gui) + len(only_cli)
    else:
        print(f"  [OK ] 禁列集合与 CLI `kForbiddenResumeFlags` 逐项一致（{len(cli_flags)} 项）")

    class _W:                       # 假控件：collect_args 只需要 .get()
        def __init__(self, value):
            self._v = value

        def get(self):
            return self._v

    def make_file(role: int) -> str:
        """造一个 v8 头（magic|version|precision|role），只需前 10 字节可读。"""
        fd, path = tempfile.mkstemp(suffix=".nnckpt" if role == 1 else ".nnmodel")
        with open(fd, "wb") as f:
            f.write(b"NNNN" + (8).to_bytes(4, "little") + bytes([0, role]))
        return path

    for role, label, expect_forbidden in ((1, "Checkpoint（.nnckpt 续训）", False),
                                         (0, "Model（.nnmodel warm start）", True)):
        path = make_file(role)
        tab = gui.GptTrainTab.__new__(gui.GptTrainTab)   # 不建控件：直接装假 _w
        tab._w = {}
        for field in gui.GptTrainTab.fields():
            if field.kind == "label":
                continue
            value = path if field.key == "resume" else field.default
            tab._w[field.name] = _W(value)
        try:
            args = tab.collect_args()
        except Exception as exc:
            print(f"  [BAD] {label} 下 collect_args 异常: {exc}")
            bad += 1
            continue
        leaked = sorted(k for k in args if k in gui.CKPT_RESUME_FORBIDDEN_KEYS)
        if expect_forbidden and not leaked:
            print(f"  [BAD] {label} 应照发 A/B 类参数（warm start 的配方来源），实际一个都没发")
            bad += 1
        elif not expect_forbidden and leaked:
            print(f"  [BAD] {label} 仍发出禁列参数: {leaked}（CLI 会报错）")
            bad += len(leaked)
        else:
            tail = f"（禁列 {len(gui.CKPT_RESUME_FORBIDDEN_KEYS)} 项全部不发）" \
                if not expect_forbidden else "（照发，CLI 全权）"
            print(f"  [OK ] {label}: 参数面符合角色语义 {tail}")
        for k in ("save", "log_interval"):
            if k not in args:
                print(f"  [BAD] {label} 把 C 类参数 {k} 也砍掉了（应保留）")
                bad += 1
        Path(path).unlink(missing_ok=True)
    return bad


# ══════════════════════════════════════════════════════════════════════
# [7] GUI 多轮对话闭环（P0-5 尾项）
# ══════════════════════════════════════════════════════════════════════
# 缺口原文（18 §3.3 X1 / 27 §3 象限（二））："GUI 推理页每轮都从零起算"。
# 现行设计 = **无状态协议**：每轮把整段历史经重复的 `--turn <role>:<text>` 交回
# CLI，回复经 `--reply-out <path>` 取回（`ChatSession`）。本节把"上下文不丢"
# 落到可断言的参数面上：
#   ① 第 N 轮的 `turns` 以第 N-1 轮的 `turns` 为**前缀**（历史逐轮累积）；
#   ② 角色恒 user/assistant 交替、末条 user（= CLI `--turn` 的硬校验形状）；
#   ③ 回复 commit 后成为下一轮的第 2k 条 `--turn`（闭环：模型自己的话也进上下文）；
#   ④ `GptInferTab._on_send` 真的把这些交给 CLI（静态查"忘了传历史"）；
#   ⑤ `--turn` / `--reply-out` / `--system` 都在 `text_infer --help` 里（真的收）。
def audit_chat_multi_turn() -> int:
    print("\n=== [7] GUI 多轮对话闭环（P0-5 尾项：连续 N 轮不丢上下文）===")
    bad = 0

    def check(ok: bool, label: str):
        nonlocal bad
        if not ok:
            print(f"  [BAD] {label}")
            bad += 1

    # ① ② 上下文逐轮累积 + 交替形状
    s = ChatSession("SYS")
    prev: list = []
    for i in range(1, 6):
        cur = s.ask(f"第{i}问")
        if i > 1:
            check(cur[:len(prev)] == prev,
                  f"第 {i} 轮的参数应以上一轮为前缀（上下文丢了：{prev} → {cur}）")
        check(len(cur) == 2 * (i - 1) + 1, f"第 {i} 轮 turns 条数 = 2(i-1)+1")
        check(cur[-1] == f"user:第{i}问", f"第 {i} 轮末条必须是本轮 user 输入")
        roles = [t.split(":", 1)[0] for t in cur]
        check(all(r == ("user" if k % 2 == 0 else "assistant") for k, r in enumerate(roles)),
              f"第 {i} 轮角色必须 user/assistant 交替（CLI --turn 校验同口径）")
        s.commit(f"第{i}问", f"第{i}答")
        prev = cur
        # ③ 回复闭环：commit 之后，模型的回复出现在下一轮的历史里
        nxt = s.ask(f"第{i+1}问")
        check(f"assistant:第{i}答" in nxt, f"第 {i} 轮的回复应进下一轮上下文")
        check(nxt[len(nxt) - 2] == f"assistant:第{i}答", "回复位置 = 末条 user 之前")

    check(s.turn_count() == 5, "turn_count = 已完成的问答对数")
    check(s.transcript("新问").count("[你]") == 6, "显示：5 条历史 user + 1 条待回复")

    # 清空 = 新对话（下一轮从零起算）
    s.clear()
    check(s.ask("新问题") == ["user:新问题"], "清空后 ask 不带任何历史")
    check(s.turn_count() == 0, "清空后轮数归零")

    # ④ GUI 侧真的把历史交给 CLI（静态：_on_send 的源码必须含这三样）
    try:
        tree = ast.parse((ROOT / "gui.py").read_text(encoding="utf-8"))
        src = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GptInferTab":
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == "_on_send":
                        src = ast.get_source_segment(
                            (ROOT / "gui.py").read_text(encoding="utf-8"), sub) or ""
        missing = [tok for tok in ("self.chat.ask(", '"turns"', '"reply_out"',
                                   "self._run_args(") if tok not in src]
        check(not missing, f"GptInferTab._on_send 应把历史与回复通道交给 CLI，缺: {missing}")
    except Exception as exc:   # pragma: no cover
        check(False, f"解析 gui.py 的 _on_send 失败: {exc}")

    # ⑤ CLI 真的收这几个参数
    flags = cli_help_flags("text_infer")
    missing_cli = sorted({"--turn", "--reply-out", "--system"} - flags)
    check(not missing_cli, f"text_infer --help 缺少多轮参数: {missing_cli}")
    if bad == 0:
        print("  [OK ] 5 轮上下文逐轮累积（前缀不变式）+ 回复闭环 + 清空归零")
        print("  [OK ] GptInferTab._on_send ↔ CLI 参数面（--turn/--reply-out/--system）")
    return bad


def main() -> int:
    bad = (audit_gui_to_cli() + audit_phantom_options() + audit_gui_attrs()
           + audit_optimizer_names() + audit_field_tables()
           + audit_ckpt_resume_filter() + audit_chat_multi_turn())
    print(f"\n{'PASS' if not bad else 'FAIL'}  可行动问题={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
