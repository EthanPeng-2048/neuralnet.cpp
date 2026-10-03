# -*- coding: utf-8 -*-
"""GUI / CLI 参数一致性审计（可复现门禁）。

复现：`python bench/gui_cli_audit.py`（需先构建 build/ 下的可执行文件）

三节：
  [1] GUI → CLI flag：把每个 Tab 可能发出的完整 kwargs 喂给对应 Controller，
      提取生成的 `--flag`，与该可执行文件 `--help` 的 flag 集合求差集。
      → 抓"GUI 会发出的、CLI 不认识的参数"（多余项 / flag 名写错）。
  [2] 幽灵选项：`--help` 声明的 flag 是否真有解析分支（源码里非帮助行出现）。
      → 抓"帮助里有、实际没解析"（用户按帮助用即报未知参数）。
  [3] gui.py 静态属性：每个类里 `self.X` 的读取是否都有赋值（含模块内祖先类）。
      → 抓"控件删了、引用还在"这类只在运行时才炸的错误（无需起 GUI / 装 customtkinter）。

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
)

FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
# 统一帮助排版器（include/neuralnet.cpp/cli/cli_help.hpp）的调用行形态：
#   help.opt("--epochs <n>", "...")
# 帮助内容只出现在这种行里（而非 `<<` 字符串链），审计据此把帮助行排除在
# "解析分支"之外——否则帮助里声明的 flag 会被误当成已解析，幽灵选项检测失效。
# 见 cli_help.hpp 文件头的审计约定：每条 help.<方法>() 必须落在同一物理行。
HELP_CALL_RE = re.compile(r"\bhelp\s*\.\s*(usage|text|section|note|item|opt|flush)\s*\(")
EXES = ["mnist_train", "mnist_infer", "text_train", "text_infer",
        "tokenizer_train", "tokenizer_infer"]

# ── 有意不暴露在 GUI 的 CLI 参数（GUI 本身即交互外壳）──
INTENTIONAL_GAPS = {"--interactive"}


def cli_help_flags(exe: str) -> set[str]:
    out = subprocess.run([str(ROOT / "build" / f"{exe}.exe"), "--help"],
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

CASES.append(("mnist_infer", MnistInferController(), dict(
    model="m.bin", input="x.csv", topk=3, show_pixels=True, gpu=True)))

CASES.append(("tokenizer_train", TokenizerTrainController(), dict(
    text_file="t.txt", tokenizer="charbpe", output="v.json", vocab_size=5000,
    min_freq=2, threads=0)))

CASES.append(("tokenizer_infer", TokenizerInferController(), dict(
    vocab="v.json", encode="hi", decode="1,2", encode_file="f.txt",
    text_file="t.txt", top=10, threads=0, show_bytes=True)))

# GPT 训练
GPT_COMMON = dict(
    save="g.bin", test_file="t.txt", vocab="v.json", epochs=10, batch_size=32,
    accum_steps=1, seq_len=256, stride=0, optimizer="adam", weight_decay=0.01,
    gpu=True, lr=0.001, lr_schedule="cosine", min_lr=1e-6, warmup_epochs=1,
    d_model=128, num_heads=4, num_layers=4, d_ff=512,
    flush_interval=1, checkpoint_every=1, activation_offload=True,
    log_interval=50, save_interval=100, grad_log=True, no_cache=True, **BASE_PREC)
CASES.append(("text_train", GptTrainController(), dict(
    model="gpt", positional_encoding="alibi", activation="swiglu", norm="rmsnorm",
    **GPT_COMMON)))
CASES.append(("text_train", GptTrainController(), dict(
    **{**GPT_COMMON,
       "model": "rapt", "resume": "r.bin", "resume_epoch": 1, "resume_step": 5,
       "lr_schedule": "step_cosine", "warmup_steps": 10, "max_norm": 1.0,
       "lr_per_epoch": "0.001"})))
CASES.append(("text_train", GptTrainController(), dict(
    model="gpt", text_file="t.txt", f16=True)))

CASES.append(("text_infer", GptInferController(), dict(
    model="g.bin", vocab="v.json", prompt="hello", max_tokens=200,
    temperature=1.0, gpu=True, show_tokens=True)))


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


def main() -> int:
    bad = audit_gui_to_cli() + audit_phantom_options() + audit_gui_attrs()
    print(f"\n{'PASS' if not bad else 'FAIL'}  可行动问题={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
