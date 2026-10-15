"""
neuralnet.cpp GUI - 基于CustomTkinter的图形界面

使用CLIController封装的CLI工具，提供以下功能标签页：
1. MNIST训练 - 训练手写数字识别模型
2. MNIST推理 - 使用训练好的模型进行推理（含画布展示）
3. 分词器训练 - 训练BPE/CharBPE分词器
4. 分词器推理 - 使用分词器编码/解码文本
5. GPT训练 - 训练文本生成模型
6. GPT推理 - 使用GPT模型生成文本

所有CLI操作通过CLIController进行，支持实时输出流和指标监控。
"""

import customtkinter as ctk
import queue
import subprocess
import threading
import time
import os
import sys
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List, Tuple
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk

from cli_controllers import (
    MnistTrainController,
    MnistInferController,
    TokenizerTrainController,
    TokenizerInferController,
    GptTrainController,
    GptInferController,
    RunState,
    Metric,
)
from train_pkg import pack, PKG_EXT

# ---------- 主题 ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------- 常量 ----------
PROJECT_DIR = Path(__file__).parent
DATASETS_DIR = PROJECT_DIR / "datasets"
PRETRAINED_DIR = PROJECT_DIR / "pretrained"
BUILD_DIR = PROJECT_DIR / "build"

# ---------- 枚举选项 ----------
ARCH_OPTIONS = ["mlp", "transformer", "cnn"]
# 优化器名单（RV-04）：**唯一真源**在 C++ —— include/neuralnet.cpp/compute_optimizer.hpp
# 的 kOptimizerTable；CLI 的 --help 文本与 --optimizer 校验都由该表生成。GUI 不再
# 单独维护这份名单：向对应 CLI 取一次 --help 并解析（按可执行名缓存），取不到
# （未构建 / 平台无该进程）时退回 _OPTIMIZER_FALLBACK —— bench/gui_cli_audit.py
# 的 [4] 节会断言退回名单与 CLI 声明逐项一致，故"改了表忘了改 GUI"不会静默通过。
_OPTIMIZER_FALLBACK = ["sgd", "sgd_momentum", "adam", "adamw", "adamwquant", "muon"]
_OPTIMIZER_CACHE: Dict[str, List[str]] = {}
_OPTIMIZER_RE = re.compile(r"优化器:([a-z0-9_/]+)")


def cli_help_text(exe_name: str) -> str:
    """跑一次 CLI 的 --help 取帮助文本；未构建 / 超时 / 异常 → 空串。"""
    exe = BUILD_DIR / (exe_name + (".exe" if os.name == "nt" else ""))
    if not exe.exists():
        return ""
    try:
        proc = subprocess.run([str(exe), "--help"], capture_output=True,
                              text=True, errors="replace", timeout=15)
    except Exception:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def parse_optimizer_options(help_text: str) -> List[str]:
    """从帮助文本抽优化器名单（与 bench/gui_cli_audit.py 的 [4] 节同口径）。

    帮助渲染器按宽度折行：先去掉所有空白再匹配 `优化器:<a/b/c>`，
    折行插入的换行/缩进因此不影响解析。
    """
    m = _OPTIMIZER_RE.search(re.sub(r"\s+", "", help_text))
    return m.group(1).split("/") if m else []


def optimizer_options(exe_name: str = "mnist_train") -> List[str]:
    """GUI 优化器下拉项：CLI 声明优先，取不到时退回 _OPTIMIZER_FALLBACK。"""
    if exe_name not in _OPTIMIZER_CACHE:
        parsed = parse_optimizer_options(cli_help_text(exe_name))
        _OPTIMIZER_CACHE[exe_name] = parsed if parsed else list(_OPTIMIZER_FALLBACK)
    return _OPTIMIZER_CACHE[exe_name]
# 后端仅 CPU / Vulkan，**本项目不支持 CUDA**（AGENTS §2）→ 不提供 CUDA 选项
ENGINE_OPTIONS = ["CPU", "GPU (Vulkan)"]
LR_SCHEDULE_OPTIONS = ["fixed", "cosine"]
GPT_LR_SCHEDULE_OPTIONS = ["fixed", "cosine", "step_cosine"]
TOKENIZER_OPTIONS = ["bpe", "charbpe"]
POSITIONAL_ENCODING_OPTIONS = ["learned", "sinusoidal", "alibi", "rope"]
ACTIVATION_OPTIONS = ["gelu", "swiglu", "relu2"]
NORM_OPTIONS = ["auto", "layernorm", "rmsnorm", "batchnorm"]
# 归一化挂载位置（仅 CNN/ViT 需要；MLP 的挂载位置由结构固定，GUI 隐藏该行）
# auto = 按架构默认：CNN = conv, ViT = final（与 mnist_train 默认一致）
NORM_PLACE_OPTIONS = ["auto", "none", "conv", "head", "both", "final"]
GPT_NORM_OPTIONS = ["layernorm", "rmsnorm"]  # GPT 仅支持 LayerNorm/RMSNorm
# 混合精度选项（compute/stable/optimizer 三槽：只接受数值精度）
PRECISION_OPTIONS = ["f32", "f16"]
# param 槽额外接受三值权重（1.58-bit，BitLinear）：
#   t1_58 = 权重三值化 + STE（docs/todo/21-quantized-weights.md）。
#   已接线：文本侧 GPT（--model gpt：4 投影 + fc1/fc2 + 未 tie 的 LM head）、
#   图像侧 --arch mlp。未接线（选它=明确报错，非静默失效）：--model rapt 前置拒绝，
#   --arch transformer/cnn 在层能力校验处报错。
#   精度预设下拉里的 "t1_58 (BitLinear + f16 存储)" = --t1_58 = profile_t1_58()
#   = {param:T1_58, compute:F16, stable:F32, optimizer:F32}。
PRECISION_PARAM_OPTIONS = ["f32", "f16", "t1_58"]
# 精度预设下拉项（= CLI 的 --f16 / --t1_58 两个别名；选中即丢掉四槽位字段）
PRECISION_PRESETS = ["f32 (默认)", "f16 (存储)", "t1_58 (BitLinear + f16 存储)"]
# 模型架构：zipt(AttnZip) 已于 2026-10-01 移除（恢复前提见 docs/history/history.md / AGENTS §12）
GPT_MODEL_OPTIONS = ["gpt", "rapt"]


# ---------- 工具函数 ----------

def _make_entry_row(parent, label_text, row, default="", width=200, **grid_kw):
    """在 parent 网格中创建一行 label + entry"""
    lbl = ctk.CTkLabel(parent, text=label_text, anchor="w", width=140)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=3)
    entry = ctk.CTkEntry(parent, width=width)
    entry.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3, **grid_kw)
    if default:
        entry.insert(0, str(default))
    return entry


def _make_option_row(parent, label_text, row, options, default=None, width=200):
    """创建一行 label + CTkOptionMenu（下拉菜单）"""
    lbl = ctk.CTkLabel(parent, text=label_text, anchor="w", width=140)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=3)
    var = ctk.StringVar(value=default or options[0])
    menu = ctk.CTkOptionMenu(parent, variable=var, values=options, width=width)
    menu.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3)
    var.widget = menu  # 便于显隐切换时拿到真正的控件
    return var


def _make_file_row(parent, label_text, row, filetypes=None, width=200):
    """创建一行带文件选择按钮的输入行"""
    lbl = ctk.CTkLabel(parent, text=label_text, anchor="w", width=140)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=3)

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3)

    entry = ctk.CTkEntry(frame, width=width - 40)
    entry.pack(side="left", padx=(0, 4))

    def browse():
        path = filedialog.askopenfilename(filetypes=filetypes or [("所有文件", "*.*")])
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    btn = ctk.CTkButton(frame, text="...", width=30, command=browse)
    btn.pack(side="left")
    return entry


def _make_dir_row(parent, label_text, row, width=200):
    """创建一行带目录选择按钮的输入行"""
    lbl = ctk.CTkLabel(parent, text=label_text, anchor="w", width=140)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=3)

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3)

    entry = ctk.CTkEntry(frame, width=width - 40)
    entry.pack(side="left", padx=(0, 4))

    def browse():
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    btn = ctk.CTkButton(frame, text="...", width=30, command=browse)
    btn.pack(side="left")
    return entry


def _make_save_row(parent, label_text, row, default_name="", filetypes=None, width=200):
    """创建一行带"另存为"对话框的输入行"""
    lbl = ctk.CTkLabel(parent, text=label_text, anchor="w", width=140)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=3)

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3)

    entry = ctk.CTkEntry(frame, width=width - 40)
    entry.pack(side="left", padx=(0, 4))
    if default_name:
        entry.insert(0, default_name)

    def browse():
        path = filedialog.asksaveasfilename(
            defaultextension=".bin",
            initialfile=default_name,
            filetypes=filetypes or [("二进制文件", "*.bin"), ("所有文件", "*.*")]
        )
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    btn = ctk.CTkButton(frame, text="...", width=30, command=browse)
    btn.pack(side="left")
    return entry


def _make_checkbox_row(parent, label_text, row, default=False):
    """创建一行 checkbox"""
    var = ctk.BooleanVar(value=default)
    cb = ctk.CTkCheckBox(parent, text=label_text, variable=var)
    cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=(0, 4), pady=3)
    return var


def _make_label(parent, text, row, font_size=12, bold=True):
    """创建分隔标题行"""
    weight = "bold" if bold else "normal"
    lbl = ctk.CTkLabel(parent, text=text, font=("", font_size, weight))
    lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 4))
    return lbl


# ================================================================
#  声明式参数表（P2-6）：**UI 行与 collect_args 的唯一真源**
# ================================================================
# 背景：以前加一个 CLI 选项要手改两处（`__init__` 里建行 + `collect_args` 里
# 取值/转换/条件发送），漏一处的后果是"界面有、命令没带"或"收集了不存在的键"。
# 现在每个参数在 `fields()` 里声明一次：行怎么建（kind）、什么默认、发到
# `collect_args` 的哪个键（arg）、空值发不发（emit）、复选框的 CLI 字面量
# （literal）、属于哪个显隐分组（group）全在这里。
#
# 门禁（`bench/gui_cli_audit.py` 第 [5] 节）：**无 GUI** 导入本表的 fields()，
# 用默认值/满值两趟驱动 `collect_fields` → 把 kwargs 交给对应 Controller →
# 断言生成的 `--flag` 都真实存在；表里声明了而 Controller 不认的键、或
# Controller 认而没人声明的键都会被点名。

@dataclass(frozen=True)
class Field:
    """一个参数行。kind：

    entry/int/float —— 文本框（int/float 会转换后发送）
    option          —— 下拉框（**总是**发送；emit="always"）
    check           —— 复选框（给了 literal 就按字面量总是发送，否则仅在与默认不同时发送）
    file/dir/save   —— 文本框 + 文件/目录/另存为按钮（空则不发送）
    text            —— 多行文本框（当前只用于 prompt）
    label           —— 分组标题（不产生参数）
    virtual         —— **无 UI**：仅声明"本 Tab 会从代码里发出这个 kwargs 键"，
                       供门禁把该 Tab 的参数面收全（如 gpu / t1_58 / precision_*）
    """
    name: str
    label: str = ""
    kind: str = "entry"
    default: Any = ""
    options: Tuple[str, ...] = ()
    arg: Optional[str] = None          # collect_args 的 kwargs 键（默认 = name）
    emit: str = "changed"              # changed | always
    literal: Optional[Tuple[Any, Any]] = None   # 复选框的 (真, 假) CLI 字面量
    filetypes: Any = None
    width: int = 200
    group: str = ""                    # 显隐分组（子类给激活规则）
    cast: Optional[Callable[[str], Any]] = None
    skip_zero: bool = False            # int 字段：值 == 0 时不发送（旧手写口径）
    ui_only: bool = False              # UI 状态：由 collect_args 后处理映射成别的键，自身不进命令

    @property
    def key(self) -> str:
        """collect_args 里的 kwargs 键"""
        return self.arg or self.name


# ── 紧凑构造器（表本身要能一眼扫完，故不写 Field(...) 全参）──────────────
def _e(name, label, default="", **kw):   return Field(name, label, "entry", default, **kw)
def _i(name, label, default="", **kw):   return Field(name, label, "int", default, cast=int, **kw)
def _fl(name, label, default="", **kw):  return Field(name, label, "float", default, cast=float, **kw)
def _o(name, label, options, default=None, **kw):
    return Field(name, label, "option", options[0] if default is None else default,
                 tuple(options), emit="always", **kw)
def _b(name, label, default=False, **kw):  return Field(name, label, "check", default, **kw)
def _file(name, label, filetypes=None, **kw):
    return Field(name, label, "file", "", filetypes=filetypes, **kw)
def _dir(name, label, default="", **kw):  return Field(name, label, "dir", default, **kw)
def _save(name, label, default="", filetypes=None, **kw):
    return Field(name, label, "save", default, filetypes=filetypes, **kw)
def _sep(label, group=""):                return Field("", label, "label", group=group)
def _v(key, note=""):                     return Field(key, note, "virtual")


def collect_fields(fields, value_of, active=None) -> Dict[str, Any]:
    """按声明式表收集 kwargs —— **参数语义的唯一实现**（纯函数：门禁可直接测）。

    · option 总是发送；entry/int/float/file/dir/save 空值不发（与旧手写口径一致）；
    · int/float 按 cast 转换（值非法时抛 ValueError，与旧 `int(...)` 行为同）；
    · check 给了 literal 就总是按字面量发送（`--shuffle-steps true|false`），
      否则仅在与默认不同时发送（旧口径：`if var.get(): args[x] = True`）；
    · label / virtual 不产生参数；`active` 为假（如按架构隐藏）的字段不发。
    """
    out: Dict[str, Any] = {}
    for f in fields:
        if f.kind in ("label", "virtual"):
            continue
        if active is not None and not active(f):
            continue
        key = f.key
        value = value_of(f)
        if f.kind == "check":
            if f.literal is not None:
                out[key] = f.literal[0] if bool(value) else f.literal[1]
            elif f.emit == "always" or bool(value) != bool(f.default):
                out[key] = bool(value)
            continue
        text = "" if value is None else str(value)
        if text == "" and f.emit != "always":
            continue
        if f.skip_zero and f.cast is not None and f.cast(text) == 0:
            continue
        if f.emit == "always":
            out[key] = f.cast(text) if f.cast else text
        else:
            out[key] = f.cast(text) if f.cast else text
    return out


class ParamsMixin:
    """把 `fields()` 声明翻译成"建控件 + 取值 + 显隐"的通用实现。"""

    # 控件表 / 行表：由 `_build_fields` 填充。这里只做**类级声明**（注解不产生
    # 共享的可变默认值），同时让 `bench/gui_cli_audit.py` 的 [3] 节静态检查
    # 认得这两个属性（子类里读 self._w 不会被误报"从未赋值"）。
    _w: Dict[str, Any]
    _rows: List[Tuple["Field", Any]]

    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        return ()

    # ── 建行 ─────────────────────────────────────────────────────
    def _build_fields(self, p, start_row: int = 0) -> int:
        self._w: Dict[str, Any] = {}
        r = start_row
        self._rows: List[Tuple[Field, Any]] = []
        for f in self.fields():
            if f.kind == "virtual":
                continue
            if f.kind == "label":
                self._rows.append((f, _make_label(p, f.label, r))); r += 1
                continue
            widget, used = self._make_field_widget(p, f, r)
            self._w[f.name] = widget
            self._rows.append((f, self._w[f.name]))
            r += used
        return r

    def _apply_groups(self):
        """按 `_field_active`（默认看 group）统一显隐 —— 取代逐 Tab 手写控件清单"""
        for f, w in self._rows:
            if not f.group:
                continue
            w = getattr(w, "widget", w)
            w.grid() if self._field_active(f) else w.grid_remove()

    @staticmethod
    def _make_field_widget(p, f: Field, r: int) -> Tuple[Any, int]:
        """建一个字段的控件；返回 (控件, 占用的网格行数)"""
        if f.kind == "option":
            return _make_option_row(p, f.label, r, list(f.options), f.default,
                                    width=f.width), 1
        if f.kind == "check":
            return _make_checkbox_row(p, f.label, r, f.default), 1
        if f.kind == "file":
            w = _make_file_row(p, f.label, r, filetypes=f.filetypes, width=f.width)
            if f.default:
                w.delete(0, "end"); w.insert(0, str(f.default))
            return w, 1
        if f.kind == "dir":
            w = _make_dir_row(p, f.label, r, width=f.width)
            if f.default:
                w.delete(0, "end"); w.insert(0, str(f.default))
            return w, 1
        if f.kind == "save":
            return _make_save_row(p, f.label, r, default_name=str(f.default),
                                  filetypes=f.filetypes, width=f.width), 1
        if f.kind == "text":
            lbl = ctk.CTkLabel(p, text=f.label, anchor="w")
            lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 2))
            box = ctk.CTkTextbox(p, width=f.width, height=70)
            box.grid(row=r + 1, column=0, columnspan=2, sticky="w", pady=(0, 4))
            if f.default:
                box.insert("1.0", str(f.default))
            return box, 2
        return _make_entry_row(p, f.label, r, f.default, width=f.width), 1

    # ── 取值 / 收集 ──────────────────────────────────────────────
    def _value_of(self, f: Field):
        w = self._w[f.name]
        if f.kind == "text":
            return w.get("1.0", "end").strip()
        return w.get()

    def _field_active(self, f: Field) -> bool:
        return True

    def collect_args(self) -> Dict[str, Any]:
        return collect_fields(self.fields(), self._value_of, self._field_active)

    def _set_field(self, name: str, value):
        """按表名设置控件值（option 的 var 需经 .widget.set）"""
        w = self._w[name]
        if hasattr(w, "widget") and hasattr(w.widget, "set"):
            w.widget.set(value)
        else:
            w.delete(0, "end"); w.insert(0, str(value))


# ================================================================
#  OutputPanel - 统一的输出面板组件
# ================================================================
class _SeriesState:
    """单系列的后台降采样状态（仅 worker 线程访问）"""
    __slots__ = ("raw", "recent", "hist", "compressed",
                 "min_v", "max_v", "last", "last_merge")

    def __init__(self):
        self.raw: List[float] = []           # 全量原始点（append-only）
        self.recent: List[float] = []        # 最近 _RECENT 个点（滑动窗口）
        self.hist: List[float] = []          # 从 recent 挤出、待归并的点
        self.compressed: List[float] = []    # 历史区间压缩成的 _BUCKETS 个代表点
        self.min_v: Optional[float] = None
        self.max_v: Optional[float] = None
        self.last: Optional[float] = None
        self.last_merge: float = 0.0


class ChartWidget(ctk.CTkFrame):
    """高性能实时图表——PIL 离屏渲染 + 后台线程增量降采样

    策略：
    - 最近 _RECENT 个点全精度保留（滑动窗口，O(1) 追加）
    - 更早的点压缩到 _BUCKETS 个桶（每桶取偏离最大的点，保留尖峰/谷底）
    - 历史桶攒够 _REBUCKET 个溢出点（或超 _MERGE_AGE 秒）才归并一次，
      归并 = 对历史区间做一次全量桶采样，故归并后输出与全量重采样精确一致、
      无累积漂移；输出长度恒为 _BUCKETS + _RECENT，X 轴缩放稳定。
    - 全部降采样/归并在独立 worker 线程执行，GUI 线程只做 O(320) 离屏绘制，
      高开销计算不阻塞界面。
    - 快照为不可变引用：worker 只整体替换、不就地修改，读侧线程安全。
    """

    _RECENT = 200           # 最近 N 个点全精度
    _BUCKETS = 120          # 历史数据压缩到 M 个桶
    _REBUCKET = 32          # 历史桶攒够多少溢出点归并一次
    _MERGE_AGE = 0.5        # 距上次归并超过该秒数也归并（稀疏数据）
    _RENDER_MS = 250        # 渲染节流间隔

    def __init__(self, master, title="", **kw):
        super().__init__(master, **kw)
        self.grid_propagate(False)
        self._title = title
        self._colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
                        "#ec4899", "#06b6d4", "#84cc16"]

        # 渲染状态（GUI 线程）
        self._dirty = False
        self._timer_id: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None   # 防 GC

        # 后台降采样状态（worker 线程独占；GUI 线程只读快照）
        self._states: Dict[str, _SeriesState] = {}
        self._inbox = queue.Queue()
        self._lock = threading.Lock()
        self._snapshot: Optional[Tuple[Dict[str, List[float]],
                                       Tuple[float, float],
                                       Dict[str, float]]] = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"chart-{title or id(self)}",
            daemon=True)
        self._worker.start()
        # 主线程持续节拍器驱动渲染（add_point 可在任意线程安全入队）
        self._timer_id = self.after(self._RENDER_MS, self._tick_render)

        # 字体（带 fallback）
        try:
            self._font_sm = ImageFont.truetype("arial.ttf", 9)
            self._font_md = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            try:
                self._font_sm = ImageFont.truetype("DejaVuSans.ttf", 9)
                self._font_md = ImageFont.truetype("DejaVuSans.ttf", 11)
            except OSError:
                self._font_sm = ImageFont.load_default()
                self._font_md = self._font_sm

        # Canvas 只做贴图容器
        self._canvas = ctk.CTkCanvas(self, bg="#1e1e2e", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self._canvas.bind("<Configure>", lambda e: self._on_resize(e))

    # ------------------------------------------------------------------
    #  后台 worker：增量降采样（recent 滑动 + 历史周期性归并）
    # ------------------------------------------------------------------
    @classmethod
    def _bucket_sample(cls, values: List[float], n_buckets: int) -> List[float]:
        """桶偏离采样：每桶取与上一保留点偏差最大的点（保尖峰/谷底）"""
        n = len(values)
        if n == 0:
            return []
        if n <= n_buckets:
            return list(values)
        bucket_sz = n / n_buckets
        out: List[float] = []
        for i in range(n_buckets):
            lo = int(i * bucket_sz)
            hi = min(int((i + 1) * bucket_sz), n)
            ref = out[-1] if out else values[lo]
            best = values[lo]
            best_d = abs(best - ref)
            for j in range(lo + 1, hi):
                d = abs(values[j] - ref)
                if d > best_d:
                    best_d = d
                    best = values[j]
            out.append(best)
        return out

    def _append(self, st: _SeriesState, value: float) -> None:
        """O(1) 追加一个点：滑动 recent 窗口，溢出点进入待归并池"""
        st.raw.append(value)
        st.last = value
        if st.min_v is None:
            st.min_v = st.max_v = value
        else:
            if value < st.min_v:
                st.min_v = value
            if value > st.max_v:
                st.max_v = value
        st.recent.append(value)
        if len(st.recent) > self._RECENT:
            st.hist.append(st.recent.pop(0))

    def _rebucket(self, st: _SeriesState) -> None:
        """历史区间全量桶采样（精确，无累积漂移；输出长度恒 _BUCKETS+_RECENT）"""
        h = len(st.raw) - self._RECENT
        st.compressed = self._bucket_sample(st.raw[:h], self._BUCKETS) if h > 0 else []
        st.hist = []
        st.last_merge = time.monotonic()

    def _maybe_merge(self, st: _SeriesState) -> None:
        if not st.hist:
            return
        if (len(st.hist) >= self._REBUCKET
                or time.monotonic() - st.last_merge >= self._MERGE_AGE):
            self._rebucket(st)

    def _publish(self) -> None:
        """把当前状态打包成不可变快照（只替换引用，不就地修改，读侧安全）"""
        series: Dict[str, List[float]] = {}
        last_vals: Dict[str, float] = {}
        lo, hi = float("inf"), float("-inf")
        for name, st in self._states.items():
            out = st.compressed + st.recent
            if out:
                series[name] = out
            if st.last is not None:
                last_vals[name] = st.last
            if st.min_v is not None:
                lo = min(lo, st.min_v)
                hi = max(hi, st.max_v)
        if lo == float("inf"):
            y_range = (0.0, 1.0)
        else:
            margin = (hi - lo) * 0.05 or 0.1
            y_range = (lo - margin, hi + margin)
        with self._lock:
            self._snapshot = (series, y_range, last_vals)

    def _handle(self, item):
        if item is None:
            self._closed = True
            return
        if isinstance(item, tuple) and item and item[0] == "__reset__":
            self._states.clear()
            with self._lock:
                self._snapshot = None
            return
        name, value = item
        st = self._states.get(name)
        if st is None:
            st = _SeriesState()
            self._states[name] = st
        self._append(st, value)

    def _worker_loop(self):
        while not self._closed:
            try:
                item = self._inbox.get(timeout=self._MERGE_AGE)
            except queue.Empty:
                # 无新数据也检查时间触发的归并（兜底）
                for st in self._states.values():
                    self._maybe_merge(st)
                continue
            self._handle(item)
            # 批量消费积压，减少发布/唤醒次数
            while True:
                try:
                    nxt = self._inbox.get_nowait()
                except queue.Empty:
                    break
                self._handle(nxt)
            for st in self._states.values():
                self._maybe_merge(st)
            self._publish()

    # ------------------------------------------------------------------
    #  渲染核心 —— PIL 离屏绘制
    # ------------------------------------------------------------------
    def _on_resize(self, event):
        self._force_redraw()

    def _force_redraw(self):
        self._dirty = False
        self._render_data()

    def _tick_render(self):
        """主线程持续节拍器：每 _RENDER_MS 检查是否有新数据并渲染"""
        self._timer_id = self.after(self._RENDER_MS, self._tick_render)
        if self._dirty:
            self._dirty = False
            self._render_data()

    def _render_data(self):
        c = self._canvas
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 60)
        ml, mr, mt, mb = 50, 10, 24, 22
        pw, ph = w - ml - mr, h - mt - mb

        # 取最新快照（只复制引用，O(1)）
        with self._lock:
            snap = self._snapshot
        if snap is None:
            series, y_range, last_vals = {}, (0.0, 1.0), {}
        else:
            series, y_range, last_vals = snap
        y_min, y_max = y_range
        span = y_max - y_min or 1.0

        # --- PIL 绘制 ---
        img = Image.new("RGB", (w, h), "#1e1e2e")
        draw = ImageDraw.Draw(img)

        # 标题
        if self._title:
            bbox = draw.textbbox((0, 0), self._title, font=self._font_md)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, 4), self._title,
                      fill="#aaaaaa", font=self._font_md)

        # Y 轴网格 + 刻度
        for i in range(5):
            yv = y_min + span * i / 4
            yp = int(mt + ph - (ph * i / 4))
            draw.line([(ml, yp), (w - mr, yp)], fill="#333333", width=1)
            label = f"{yv:.3g}"
            lb = draw.textbbox((0, 0), label, font=self._font_sm)
            lw = lb[2] - lb[0]
            draw.text((ml - lw - 6, yp - 5), label,
                      fill="#777777", font=self._font_sm)

        # X 轴底部线
        draw.line([(ml, mt + ph), (w - mr, mt + ph)], fill="#333333", width=1)

        # 逐系列绘制
        series_keys = list(series.keys())
        for idx, name in enumerate(series_keys):
            vals = series[name]
            if len(vals) < 2:
                continue
            color = self._colors[idx % len(self._colors)]
            n = len(vals)
            points = []
            for i, v in enumerate(vals):
                x = int(ml + (i / max(n - 1, 1)) * pw)
                y = int(mt + ph - ((v - y_min) / span) * ph)
                points.append((x, y))
            draw.line(points, fill=color, width=2)

            # 标记尾部（最近点用圆点高亮）
            if points:
                lx, ly = points[-1]
                r = 3
                draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=color)

        # 图例（右上角）
        lx = w - mr - 4
        ly = mt + 4
        for idx, name in enumerate(series_keys):
            last = last_vals.get(name)
            if last is None:
                continue
            color = self._colors[idx % len(self._colors)]
            draw.line([(lx - 28, ly), (lx - 12, ly)], fill=color, width=2)
            label = f"{name}={last:.4g}"
            draw.text((lx - 10, ly - 5), label,
                      fill=color, font=self._font_sm, anchor="ra")
            ly += 14

        # 贴到 Canvas
        self._photo = ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self._photo)

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------
    def add_point(self, series_name: str, value: float):
        """O(1) 入队（可在任意线程调用）；渲染由主线程节拍器驱动"""
        self._inbox.put((series_name, value))
        self._dirty = True

    def clear_data(self):
        """清空所有数据（GUI 立即清显示，worker 异步重置状态）"""
        self._dirty = False
        with self._lock:
            self._snapshot = None
        self._photo = None
        self._canvas.delete("all")
        self._inbox.put(("__reset__",))

    def destroy(self):
        """销毁时停掉节拍器并通知后台线程退出（不依赖 <Destroy> 事件）"""
        self._closed = True
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
        try:
            self._inbox.put(None)
        except Exception:
            pass
        super().destroy()


class OutputPanel(ctk.CTkFrame):
    """可复用的输出面板：多图表 + 文本输出 + 实时指标

    chart_configs 格式: [{"title": str, "metrics": [str, ...]}, ...]
    每个 config 生成一个独立的 ChartWidget，metrics 中的指标会被路由到对应图表。
    """

    def __init__(self, master, chart_configs=None, **kw):
        super().__init__(master, **kw)

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        ctk.CTkLabel(toolbar, text="输出", font=("", 13, "bold")).pack(side="left")
        ctk.CTkButton(toolbar, text="清空", width=50,
                       command=self._clear).pack(side="right")

        # 多图表区域
        self._charts: List[ChartWidget] = []
        self._metric_to_chart: Dict[str, ChartWidget] = {}
        row = 1
        if chart_configs:
            for cfg in chart_configs:
                cw = ChartWidget(self, title=cfg["title"])
                cw.grid(row=row, column=0, sticky="nsew", padx=4, pady=(4, 2))
                self.grid_rowconfigure(row, minsize=200)
                self._charts.append(cw)
                for mname in cfg["metrics"]:
                    self._metric_to_chart[mname] = cw
                row += 1

        self.textbox = ctk.CTkTextbox(self, wrap="word", state="disabled",
                                       font=("Consolas", 11), height=100)
        self.textbox.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        row += 1

        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.grid_rowconfigure(row, minsize=40)

        # 设置列权重和行权重
        self.grid_columnconfigure(0, weight=1)
        if self._charts:
            # 让第一个图表行可伸缩
            self.grid_rowconfigure(1, weight=1)
        self.metric_labels: Dict[str, ctk.CTkLabel] = {}

    def append(self, text: str):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", text + "\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def set_metric(self, name: str, value: str):
        if name not in self.metric_labels:
            lbl = ctk.CTkLabel(self.metrics_frame, text="", font=("", 12))
            lbl.pack(side="left", padx=12, pady=4)
            self.metric_labels[name] = lbl
        self.metric_labels[name].configure(text=f"{name}: {value}")

    def _clear(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")
        for lbl in self.metric_labels.values():
            lbl.configure(text="")
        for cw in self._charts:
            cw.clear_data()


# ================================================================
#  TabBase - 所有标签页的基类
# ================================================================
class TabBase(ParamsMixin, ctk.CTkFrame):
    """标签页基类：左右分栏（参数 | 输出）。

    `ParamsMixin` 提供声明式字段表（P2-6）：子类声明 `fields()` 即可自动建行 +
    收集 kwargs；本类只保留布局/运行/输出这些通用逻辑。
    """

    def __init__(self, master, controller, chart_configs=None, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.controller = controller
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 左侧参数面板 ---
        self.params_frame = ctk.CTkScrollableFrame(self, label_text="参数",
                                                     width=420)
        self.params_frame.grid(row=0, column=0, sticky="nswe", padx=(8, 4), pady=8)

        # --- 右侧输出面板 ---
        self.output = OutputPanel(self, width=500,
                                   chart_configs=chart_configs)
        self.output.grid(row=0, column=1, sticky="nswe", padx=(4, 8), pady=8)

        # --- 底部控制栏 ---
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.ctrl = ctrl
        self.run_btn = ctk.CTkButton(ctrl, text="▶ 运行", width=120,
                                      fg_color="#2563eb",
                                      hover_color="#1d4ed8",
                                      command=self._on_run)
        self.run_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ctk.CTkButton(ctrl, text="■ 停止", width=80,
                                       fg_color="#dc2626",
                                       hover_color="#b91c1c",
                                       command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.status_lbl = ctk.CTkLabel(ctrl, text="就绪", text_color="#888")
        self.status_lbl.pack(side="right")

        self._running = False
        self.pack(fill="both", expand=True)

    def _on_run(self):
        if self._running:
            return
        self._running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_lbl.configure(text="运行中...", text_color="#22c55e")
        self.output._clear()

        self.controller.set_output_callback(
            lambda line: self.after(0, self.output.append, line))
        self.controller.set_error_callback(
            lambda line: self.after(0, self.output.append, f"[ERR] {line}"))
        self.controller.set_metric_callback(
            lambda m: self.after(0, self._on_metric, m))
        self.controller.set_completion_callback(
            lambda code: self.after(0, self._on_complete, code))

        args = self.collect_args()

        def worker():
            try:
                self.controller.run(**args)
            except Exception as e:
                self.after(0, self.output.append, f"[异常] {e}")
                self.after(0, self._finish)

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self):
        self.controller.cancel()
        self.after(0, self.output.append, "[系统] 已请求停止")

    def _on_metric(self, m: Metric):
        dispatch = {
            "batch_loss":       lambda: ("batch_loss", f"{m.value:.4f}"),
            "train_loss":       lambda: ("train_loss", f"{m.value:.4f}"),
            "test_loss":        lambda: ("test_loss", f"{m.value:.4f}"),
            "learning_rate":    lambda: ("lr", f"{m.value:.6f}"),
            "train_accuracy":   lambda: ("train_acc", f"{m.value:.1f}%"),
            "test_accuracy":    lambda: ("test_acc", f"{m.value:.1f}%"),
            "perplexity":       lambda: ("perplexity", f"{m.value:.2f}"),
            "grad_norm":        lambda: ("grad_norm", f"{m.value:.4f}"),
            "merge_progress":   lambda: ("merge", f"{m.value:.1f}%"),
            "vocab_size":       lambda: ("vocab", str(int(m.value))),
            "generated_tokens": lambda: ("tokens", str(int(m.value))),
            "generation_speed": lambda: ("speed", f"{m.value:.1f} tok/s"),
            "prediction":       lambda: ("prediction", str(int(m.value))),
            "generation_time":  lambda: ("gen_time", f"{m.value:.2f}s"),
        }
        if m.name in dispatch:
            name, value = dispatch[m.name]()
            self.output.set_metric(name, value)
        # 送入对应图表
        chart = self.output._metric_to_chart.get(m.name)
        if chart is not None:
            chart.add_point(m.name, m.value)

    def _on_complete(self, code: int):
        color = "#22c55e" if code == 0 else "#ef4444"
        label = "完成" if code == 0 else f"失败 (code={code})"
        self.status_lbl.configure(text=label, text_color=color)
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._running = False
        self.output.append(f"--- 进程退出，退出码: {code} ---")

    def _finish(self):
        self.status_lbl.configure(text="异常终止", text_color="#ef4444")
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._running = False

    # 训练包导出 ------------------------------------------------
    def _add_export_button(self):
        """在底部控制栏添加"导出训练包"按钮（训练标签页调用）"""
        self.export_btn = ctk.CTkButton(
            self.ctrl, text="📦 导出训练包", width=130,
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._on_export)
        self.export_btn.pack(side="left", padx=(0, 8))

    def build_pack_config(self) -> Dict[str, Any]:
        """子类实现：把当前 UI 参数构造成训练包配置 dict"""
        raise NotImplementedError

    def _on_export(self):
        """把当前训练配置 + 训练集打包成 .nnpkg，便于跨设备复现训练"""
        try:
            cfg = self.build_pack_config()
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            return
        if not (cfg.get("data") or {}).get("train"):
            messagebox.showwarning("导出训练包", "未设置训练数据路径，无法导出")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=PKG_EXT,
            initialfile=f"{cfg.get('name', 'run')}{PKG_EXT}",
            filetypes=[("训练包", f"*{PKG_EXT}"), ("所有文件", "*.*")],
        )
        if not out:
            return
        try:
            manifest = pack(cfg, out, base_dir=PROJECT_DIR)
            n = len(manifest["package"]["data"])
            messagebox.showinfo(
                "导出成功",
                f"已生成训练包:\n{out}\n\n"
                f"数据文件: {n} 个\n压缩: {manifest['package']['compress']}\n"
                f"包大小: {Path(out).stat().st_size:,} bytes",
            )
        except Exception as e:
            messagebox.showerror("导出失败", str(e))


# ================================================================
#  Tab 1: MNIST 训练
# ================================================================
class MnistTrainTab(TabBase):
    def __init__(self, master):
        super().__init__(master, MnistTrainController(),
                         chart_configs=[
                             {"title": "Loss", "metrics": ["batch_loss", "train_loss", "test_loss"]},
                             {"title": "Accuracy", "metrics": ["train_accuracy", "test_accuracy"]},
                         ])

        # 参数行全部来自声明式字段表（P2-6）：加一个 CLI 选项 = 在 fields() 里加一行
        p = self.params_frame
        self._build_fields(p, 0)

        # 架构 / 调度 / 预设切换回调（显隐统一由 _apply_groups 按 group 驱动）
        self._w["arch"].trace_add("write", lambda *a: self._apply_groups())
        self._w["lr_schedule"].trace_add("write", lambda *a: self._apply_groups())
        self._w["precision_preset"].trace_add("write", self._on_precision_preset_change)
        self._apply_groups()
        self._on_precision_preset_change()

        self._add_export_button()

    def build_pack_config(self):
        """把当前 MNIST 训练 UI 参数打包为训练包配置"""
        args = self.collect_args()
        data = {}
        if args.get("dataset"):
            data["train"] = args.pop("dataset")
        dev = "cpu"
        if args.pop("gpu", False):
            dev = "gpu"
        name = Path(args.get("save", "mnist_model.nnmodel")).stem or "mnist_run"
        return {"format_version": 1, "name": name, "task": "mnist",
                "device": dev, "data": data, "hyperparameters": args}

    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        """MNIST 训练页字段表（P2-6：UI 行与 collect_args 的唯一真源）。

        group 语义（见 `_field_active`）：mlp/tf/cnn = 按架构显示；
        place = 仅 Transformer/CNN（MLP 的归一化挂载位置由结构固定）；
        sched = 仅非 fixed 调度（fixed 下 min_lr / warmup 无意义）。
        """
        return (
            _o("arch", "架构", ARCH_OPTIONS, "mlp"),
            _dir("dataset", "数据集目录", str(DATASETS_DIR / "mnist_data")),
            _save("save_path", "保存路径", "pretrained/mnist_model.nnmodel", arg="save",
                  filetypes=[("模型文件", "*.nnmodel *.bin")]),
            _file("resume_path", "恢复路径 (训练断点 .nnckpt)", arg="resume",
                  filetypes=[("训练断点", "*.nnckpt *.bin")]),
            _i("epochs", "轮数 (epochs)", "10"),
            _i("batch_size", "批大小", "64"),
            _o("optimizer", "优化器", optimizer_options("mnist_train"), "adam"),
            _fl("weight_decay", "权重衰减", "0.01"),
            _o("engine", "计算引擎", ENGINE_OPTIONS, "CPU", ui_only=True),
            _i("max_samples", "最大样本数 (0=全部)", "0"),
            _sep("── 学习率调度 ──"),
            _fl("lr", "最大学习率 (lr)", "0.001"),
            _o("lr_schedule", "调度类型", LR_SCHEDULE_OPTIONS, "fixed"),
            _fl("min_lr", "最小学习率 (min_lr)", "1e-6", group="sched"),
            _i("warmup_epochs", "预热轮数", "0", group="sched"),
            _e("lr_per_epoch", "每轮 lr (逗号分隔, 可选)"),
            _sep("── 训练控制 ──"),
            # --shuffle-steps 收 true/false 字面量（不是开关标志）⇒ literal
            _b("shuffle_steps", "每 epoch 打乱 batch", True, literal=("true", "false")),
            _sep("── 混合精度 ──"),
            _o("precision_preset", "精度预设", PRECISION_PRESETS, PRECISION_PRESETS[0],
               ui_only=True),
            _o("precision_param", "参数精度 (param)", PRECISION_PARAM_OPTIONS, "f32"),
            _o("precision_compute", "计算精度 (compute)", PRECISION_OPTIONS, "f32"),
            _o("precision_stable", "稳定精度 (stable)", PRECISION_OPTIONS, "f32"),
            _o("precision_optimizer", "优化器精度 (optimizer)", PRECISION_OPTIONS, "f32"),
            _o("norm", "归一化层", NORM_OPTIONS, "auto"),
            _o("norm_place", "归一化位置", NORM_PLACE_OPTIONS, "auto", group="place"),
            _sep("── MLP 参数 ──", group="mlp"),
            _e("layer_dims", "layer_dims", "784,512,256,128,64,10", group="mlp"),
            _sep("── Transformer 参数 ──", group="tf"),
            _i("d_model", "d_model", "64", group="tf"),
            _i("num_heads", "num_heads", "4", group="tf"),
            _i("num_layers", "num_layers", "2", group="tf"),
            _i("d_ff", "d_ff", "128", group="tf"),
            _i("patch_size", "patch_size", "7", group="tf"),
            _i("eval_samples", "eval_samples", "200", group="tf"),
            _sep("── CNN 参数 ──", group="cnn"),
            _e("cnn_channels", "cnn_channels", "6,16", group="cnn"),
            _e("cnn_kernels", "cnn_kernels", "5,5", group="cnn"),
            _i("cnn_pool", "cnn_pool", "2", group="cnn"),
            _e("cnn_fc", "cnn_fc", "120,10", group="cnn"),
            # 无 UI：由 collect_args 的后处理发出（门禁用它们把参数面收全）
            _v("gpu", "engine == GPU (Vulkan) 时才发"),
            _v("f16", "精度预设 = f16"),
            _v("t1_58", "精度预设 = t1_58"),
        )

    def _field_active(self, f: Field) -> bool:
        if f.group == "mlp":
            return self._w["arch"].get() == "mlp"
        if f.group == "tf":
            return self._w["arch"].get() == "transformer"
        if f.group == "cnn":
            return self._w["arch"].get() == "cnn"
        if f.group == "place":
            return self._w["arch"].get() in ("transformer", "cnn")
        if f.group == "sched":
            return self._w["lr_schedule"].get() != "fixed"
        return True

    def build_pack_config(self):
        """把当前 MNIST 训练 UI 参数打包为训练包配置"""
        args = self.collect_args()
        data = {}
        if args.get("dataset"):
            data["train"] = args.pop("dataset")
        dev = "cpu"
        if args.pop("gpu", False):
            dev = "gpu"
        name = Path(args.get("save", "mnist_model.nnmodel")).stem or "mnist_run"
        return {"format_version": 1, "name": name, "task": "mnist",
                "device": dev, "data": data, "hyperparameters": args}

    def _on_precision_preset_change(self, *args):
        """精度预设切换时自动填充各精度字段（与 CLI 的 --f16 / --t1_58 逐字段对齐）：
        f16 → {param:F16, compute:F16, stable:F32, optimizer:F32}；
        t1_58 → 同 f16 但 param=T1_58（"其他默认 f16"）。语义见 precision.hpp。
        """
        preset = self._w["precision_preset"].get()
        if "t1_58" in preset:
            param, compute = "t1_58", "f16"
        elif "f16" in preset:
            param, compute = "f16", "f16"
        else:
            param, compute = "f32", "f32"
        self._set_field("precision_param", param)
        self._set_field("precision_compute", compute)
        self._set_field("precision_stable", "f32")
        self._set_field("precision_optimizer", "f32")

    def collect_args(self):
        """表驱动收集 + **两处显式后处理**（表表达不了的条件语义）。

        后处理只做"开关 / 别名的选择"，不做值搬运：
          · `engine` → `--gpu`（UI 是下拉，CLI 是布尔标志）；
          · `precision_preset` → `--f16` / `--t1_58` 别名（选中时**丢掉**四个槽位
            字段，与 CLI 的"预设 = 整份 profile 快捷方式"语义一致）。
        """
        args = collect_fields(self.fields(), self._value_of, self._field_active)
        if args.pop("engine", "CPU") == "GPU (Vulkan)":
            args["gpu"] = True
        preset = args.pop("precision_preset", PRECISION_PRESETS[0])
        if "t1_58" in preset or "f16" in preset:
            args["t1_58" if "t1_58" in preset else "f16"] = True
            for slot in ("precision_param", "precision_compute", "precision_stable",
                         "precision_optimizer"):
                args.pop(slot, None)
        return args


# ================================================================
#  Tab 2: MNIST 推理
# ================================================================
class StrokeTrack:
    """画板笔迹状态（纯逻辑，无 Tk/PIL 依赖 ⇒ 可单测）。

    P2-6「绘制节流」半边：拖动期的每帧开销封顶 + 空帧跳过。

    · **每笔只建一个画布项**：当前笔迹的点集原地增长，调用方用
      `canvas.coords(item, ...)` 刷新同一个项，不再为每个 motion 事件
      `create_line`（旧实现：一笔拖动 = N 个画布项，重放/重绘都随项数增长）。
    · **空帧跳过**：坐标未变的重复 motion 事件直接返回 False（dirty 标志），
      调用方据此不刷新画布。
    · **光栅化延迟**：笔迹段在拖动期只累积（`drain()` 之前不动 PIL），
      边界（刷新预览 / 保存 CSV / 结束一笔）一次性按折线画进高分辨率缓冲
      ⇒ 拖动期零 PIL 调用，折线段数从"每段一次调用"降为"每批一次调用"。
    """

    def __init__(self):
        self._strokes = []      # 已提交笔迹（dict；_draw_grid 据此重放）
        self._live = None       # 当前笔迹（dict，pts 为点列表）
        self._pending = []      # 待光栅化的折线（点列表的列表）

    # ── 一笔的生命周期 ────────────────────────────────────────────
    def begin(self, x, y, brush, canvas_w, canvas_h):
        """开始一笔：记录起点与当时的画布尺寸（重放时按尺寸缩放）"""
        self._live = {"type": "poly", "pts": [(x, y)], "width": int(brush),
                      "w": canvas_w, "h": canvas_h}
        self._strokes.append(self._live)

    def extend(self, x, y):
        """追加一个点；返回 True = 有新笔迹（需刷新画布），False = 空帧"""
        if self._live is None:
            return False
        pts = self._live["pts"]
        if pts and pts[-1] == (x, y):
            return False        # 空帧：坐标与上一帧相同
        pts.append((x, y))
        return True

    def end(self):
        """结束一笔（进入待光栅化队列）；返回是否有待画内容"""
        live = self._live
        self._live = None
        if live is None:
            return False
        pts = live["pts"]
        # 单击（只有 1 个点）：用"同坐标两点 + 圆头"表示一个点
        self._pending.append(pts if len(pts) >= 2 else [pts[0], pts[0]])
        return True

    def drain(self):
        """取走待光栅化的折线（取走后队列清空；调用方负责真正画出来）"""
        out = self._pending
        self._pending = []
        return out

    def clear(self):
        self._strokes.clear()
        self._live = None
        self._pending.clear()

    # ── 只读视图 ──────────────────────────────────────────────────
    def strokes(self):
        """已提交笔迹（含当前未结束的一笔）"""
        return self._strokes

    def live_points(self):
        """当前笔迹的点列表（未开始一笔时为空）"""
        return self._live["pts"] if self._live is not None else []


class MnistInferTab(TabBase):
    _PAINT_MS = 16          # 画布刷新节流（P2-6：每个 motion 事件最多每帧刷一次）

    def __init__(self, master):
        super().__init__(master, MnistInferController())

        import tempfile
        import tkinter as _tk
        self._tempfile = tempfile
        self._tk = _tk

        p = self.params_frame
        # 普通字段走声明式表（P2-6）；输入方式/画板是自定义控件，见本类注释
        r = self._build_fields(p, 0)

        # 单个文件输入行
        self._single_file_label = ctk.CTkLabel(p, text="图片文件 (CSV)", anchor="w", width=140)
        self._single_file_label.grid(row=r, column=0, sticky="w", padx=(0, 4), pady=3)
        _single_frame = ctk.CTkFrame(p, fg_color="transparent")
        _single_frame.grid(row=r, column=1, sticky="w", padx=(0, 4), pady=3)
        self.single_file_path = ctk.CTkEntry(_single_frame, width=150)
        self.single_file_path.pack(side="left", padx=(0, 4))
        def _browse_single():
            path = filedialog.askopenfilename(filetypes=[("CSV 图片", "*.csv"), ("所有文件", "*.*")])
            if path:
                self.single_file_path.delete(0, "end")
                self.single_file_path.insert(0, path)
        ctk.CTkButton(_single_frame, text="...", width=30,
                       command=_browse_single).pack(side="left")
        self._single_file_row = _single_frame
        r += 1

        # 目录输入行
        self._dir_label = ctk.CTkLabel(p, text="图片目录", anchor="w", width=140)
        self._dir_label.grid(row=r, column=0, sticky="w", padx=(0, 4), pady=3)
        _dir_frame = ctk.CTkFrame(p, fg_color="transparent")
        _dir_frame.grid(row=r, column=1, sticky="w", padx=(0, 4), pady=3)
        self.dir_path = ctk.CTkEntry(_dir_frame, width=150)
        self.dir_path.pack(side="left", padx=(0, 4))
        def _browse_dir():
            path = filedialog.askdirectory()
            if path:
                self.dir_path.delete(0, "end")
                self.dir_path.insert(0, path)
        ctk.CTkButton(_dir_frame, text="...", width=30,
                       command=_browse_dir).pack(side="left")
        self._dir_row = _dir_frame
        r += 1

        # --- 画板区域（28×28 正方形，每格 10px） ---
        CANVAS_SIZE = 280
        self.canvas_label = _make_label(p, "── 画板 (鼠标绘制 28×28) ──", r); r += 1
        self.canvas_frame = ctk.CTkFrame(p)
        self.canvas_frame.grid(row=r, column=0, columnspan=2, pady=4)
        self.canvas_frame.grid_propagate(False)
        self.canvas_frame.configure(width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="#1a1a2e",
                                     highlightthickness=0, cursor="crosshair",
                                     width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.pack(padx=2, pady=2)
        self._last_x = None
        self._last_y = None
        self._brush_size = 14
        r += 1

        # 画板控制按钮
        self._canvas_btn_frame = ctk.CTkFrame(p, fg_color="transparent")
        self._canvas_btn_frame.grid(row=r, column=0, columnspan=2, sticky="w",
                                     padx=4, pady=2)
        ctk.CTkButton(self._canvas_btn_frame, text="清空画板", width=80,
                       fg_color="#6b7280", hover_color="#4b5563",
                       command=self._clear_canvas).pack(side="left", padx=2)
        ctk.CTkButton(self._canvas_btn_frame, text="缩小笔刷", width=70,
                       fg_color="#6b7280", hover_color="#4b5563",
                       command=self._brush_smaller).pack(side="left", padx=2)
        ctk.CTkButton(self._canvas_btn_frame, text="放大笔刷", width=70,
                       fg_color="#6b7280", hover_color="#4b5563",
                       command=self._brush_bigger).pack(side="left", padx=2)
        self._brush_label = ctk.CTkLabel(self._canvas_btn_frame,
                                          text=f"笔刷: {self._brush_size}px",
                                          font=("", 11))
        self._brush_label.pack(side="left", padx=6)
        r += 1

        # --- 结果预览区域（与画板同为正方形） ---
        self._preview_label = _make_label(p, "── 预览结果 ──", r); r += 1
        self._preview_frame = ctk.CTkFrame(p)
        self._preview_frame.grid(row=r, column=0, columnspan=2, pady=4)
        self._preview_frame.grid_propagate(False)
        self._preview_frame.configure(width=CANVAS_SIZE, height=CANVAS_SIZE)
        self._preview_canvas = ctk.CTkCanvas(self._preview_frame, bg="#1a1a2e",
                                              highlightthickness=0,
                                              width=CANVAS_SIZE, height=CANVAS_SIZE)
        self._preview_canvas.pack(padx=2, pady=2)
        self._preview_img_ref = None  # 防止 GC
        self._CANVAS_SIZE = CANVAS_SIZE
        self._preview_placeholder = self._preview_canvas.create_text(
            CANVAS_SIZE // 2, CANVAS_SIZE // 2, text="推理结果图片将在此显示",
            fill="#666", font=("", 11))
        r += 1

        # 高分辨率像素缓冲（与画布同尺寸），绘制时使用
        from PIL import Image as _PILImage
        self._pil = _PILImage
        self._pixel_buf_hires = _PILImage.new("L", (self._CANVAS_SIZE, self._CANVAS_SIZE), 0)
        self._pixel_buf = _PILImage.new("L", (28, 28), 0)  # 28x28 用于推理
        # 笔迹状态（P2-6 绘制节流）：每笔一个画布项 + 空帧跳过 + 光栅化延迟
        self._track = StrokeTrack()
        self._live_item = None   # 当前笔迹的画布项（每帧 coords 刷新，不新建）
        self._last_paint = 0.0   # 上次画布刷新时刻（按帧节流；数据不受节流）
        self._canvas_w = 1
        self._canvas_h = 1

        # 绑定鼠标事件
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        # 输入方式切换回调
        self._w["input_method"].trace_add("write", self._on_input_method_change)
        self._on_input_method_change()

    def _on_canvas_configure(self, event):
        """画布尺寸变化时重绘网格并更新缩放比例"""
        self._canvas_w = max(event.width, 1)
        self._canvas_h = max(event.height, 1)
        self._draw_grid()

    def _on_input_method_change(self, *args):
        """切换输入方式时显示/隐藏对应控件：单个文件 / 目录 / 画板"""
        method = self._w["input_method"].get()
        is_canvas = method == "画板绘制"
        is_single = method == "单个文件"
        is_dir = method == "目录"
        # 单个文件
        self._single_file_label.grid() if is_single else self._single_file_label.grid_remove()
        self._single_file_row.grid() if is_single else self._single_file_row.grid_remove()
        # 目录
        self._dir_label.grid() if is_dir else self._dir_label.grid_remove()
        self._dir_row.grid() if is_dir else self._dir_row.grid_remove()
        # 画板
        self.canvas_label.grid() if is_canvas else self.canvas_label.grid_remove()
        self.canvas_frame.grid() if is_canvas else self.canvas_frame.grid_remove()
        self._canvas_btn_frame.grid() if is_canvas else self._canvas_btn_frame.grid_remove()

    def _draw_grid(self):
        """在画布上绘制 28x28 网格，并重放已保存的笔迹（按当前尺寸缩放）"""
        self.canvas.delete("all")
        self._live_item = None   # delete("all") 后旧项引用失效；当前笔迹由下面的重放画回
        w = self._canvas_w
        h = self._canvas_h
        cell_w = w / 28
        cell_h = h / 28
        for i in range(29):
            x = int(i * cell_w)
            self.canvas.create_line(x, 0, x, h, fill="#333355", width=1)
        for j in range(29):
            y = int(j * cell_h)
            self.canvas.create_line(0, y, w, y, fill="#333355", width=1)
        # 重放笔迹（按当前画布尺寸缩放，保持视觉比例）
        for s in self._track.strokes():
            sw = s.get("w") or 1
            sh = s.get("h") or 1
            sx = self._canvas_w / sw
            sy = self._canvas_h / sh
            pts = s.get("pts") or []
            flat = [c for p in pts for c in p]
            if len(pts) < 2:   # 单击：同坐标两点 = 一个点
                flat = flat + flat[:2]
            width = max(1, int(s.get("width", self._brush_size) * sx))
            self.canvas.create_line(*[v * (sx if i % 2 == 0 else sy)
                                      for i, v in enumerate(flat)],
                                     fill="white", width=width,
                                     capstyle="round", smooth=True)

    def _canvas_to_pixel(self, cx, cy):
        """将画布坐标映射到 28x28 像素缓冲坐标"""
        px = int(cx / self._canvas_w * 28)
        py = int(cy / self._canvas_h * 28)
        return max(0, min(27, px)), max(0, min(27, py))

    def _on_draw_start(self, event):
        """开始一笔：**只记录起点**，画布项在 _paint_live 里首次创建"""
        self._last_x = event.x
        self._last_y = event.y
        self._track.begin(event.x, event.y, self._brush_size,
                          self._canvas_w, self._canvas_h)
        self._live_item = None
        self._paint_live()     # 单击也要立刻可见（同坐标两点 + 圆头）

    def _on_draw_move(self, event):
        """追加一个点：空帧（坐标未变）直接跳过；画布刷新**按帧节流**

        数据侧永不节流（`extend` 每个事件都记），只节流视觉刷新：快速拖动时
        Tk 每帧可能吐出上百个 motion 事件，逐个 `create_line`/`coords` 是
        拖动卡顿的主因。刷新上限 = 每 `_PAINT_MS` 一次（结束一笔时强制补一次）。
        """
        if not self._track.extend(event.x, event.y):
            return              # 空帧：无新笔迹，不刷新
        self._last_x = event.x
        self._last_y = event.y
        now = time.monotonic()
        if now - self._last_paint >= self._PAINT_MS / 1000.0:
            self._last_paint = now
            self._paint_live()

    def _paint_live(self):
        """把当前笔迹刷到画布：复用 **同一个** 画布项（P2-6：不为每个
        motion 事件新建 item，拖动期画布项数 = 笔数）"""
        pts = self._track.live_points()
        if not pts:
            return
        flat = [c for p in pts for c in p]
        if len(pts) == 1:
            flat = flat + flat[:2]
        if self._live_item is None:
            self._live_item = self.canvas.create_line(
                *flat, fill="white", width=self._brush_size,
                capstyle="round", smooth=True)
        else:
            self.canvas.coords(self._live_item, *flat)

    def _on_draw_end(self, event):
        self._paint_live()     # 补一次视觉刷新（节流期间最后一段也要看得见）
        self._track.end()      # 进入待光栅化队列（拖动期零 PIL 调用）
        self._live_item = None
        self._last_x = None
        self._last_y = None
        self._update_preview_from_buffer()

    def _clear_canvas(self):
        self._track.clear()
        self._live_item = None
        self.canvas.delete("all")
        self._draw_grid()
        self._pixel_buf_hires = self._pil.new("L", (self._CANVAS_SIZE, self._CANVAS_SIZE), 0)
        self._pixel_buf = self._pil.new("L", (28, 28), 0)
        self._update_preview_from_buffer()

    def _brush_smaller(self):
        self._brush_size = max(4, self._brush_size - 2)
        self._brush_label.configure(text=f"笔刷: {self._brush_size}px")

    def _brush_bigger(self):
        self._brush_size = min(40, self._brush_size + 2)
        self._brush_label.configure(text=f"笔刷: {self._brush_size}px")

    # ── 预览绘制 ─────────────────────────────────────────────
    def _update_preview_from_buffer(self):
        """从画板像素缓冲刷新预览（黑底白字，与 MNIST 一致）"""
        self._downsample_buffer()
        pixels = [self._pixel_buf.getpixel((x, y)) / 255.0
                  for y in range(28) for x in range(28)]
        self._render_preview(pixels)

    def _update_preview_from_csv(self, path: str):
        """从 CSV 文件读取 784 个像素值并刷新预览"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
            values = [float(v) for v in line.strip().split(",") if v.strip()]
            if len(values) < 784:
                return
            self._render_preview(values[:784])
        except Exception:
            pass

    def _render_preview(self, pixels: List[float]):
        """将 784 个 0~1 像素放大绘制到预览 canvas（每格 10px 正方形）"""
        c = self._preview_canvas
        c.delete("all")
        if self._preview_placeholder:
            try:
                c.delete(self._preview_placeholder)
            except Exception:
                pass
            self._preview_placeholder = None
        size = self._CANVAS_SIZE
        cell = size / 28
        for i in range(28):
            for j in range(28):
                v = pixels[i * 28 + j]
                if v <= 0.01:
                    continue  # 背景（黑）不画
                shade = max(0, min(255, int(v * 255)))
                color = f"#{shade:02x}{shade:02x}{shade:02x}"
                x0, y0 = j * cell, i * cell
                x1, y1 = x0 + cell + 1, y0 + cell + 1
                c.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    def _downsample_buffer(self):
        """从高分辨率缓冲降采样到 28×28，使用双线性插值产生灰度抗锯齿效果"""
        # P2-6：先把待光栅化的笔迹一次性画进高分辨率缓冲（拖动期零 PIL
        # 调用；本函数是"读缓冲"的唯一入口，因此也是唯一的刷新点）
        self._rasterize_pending()
        self._pixel_buf = self._pixel_buf_hires.resize((28, 28), self._pil.BILINEAR)

    def _rasterize_pending(self):
        """把待光栅化笔迹按折线批量画进高分辨率缓冲（每批一次 draw.line）"""
        pending = self._track.drain()
        if not pending:
            return
        from PIL import ImageDraw as _Draw
        draw = _Draw.Draw(self._pixel_buf_hires)
        for pts in pending:
            flat = [c for p in pts for c in p]
            draw.line(flat, fill=255, width=self._brush_size, joint="curve")

    def _save_canvas_as_csv(self) -> str:
        """将画板内容保存为 CSV 文件（784 个 0~1 像素值），返回文件路径
        画板为黑底白字（背景=0, 笔迹=255），MNIST 训练数据同为黑底白字，无需反转。
        """
        self._downsample_buffer()
        pixels = []
        for y in range(28):
            for x in range(28):
                v = self._pixel_buf.getpixel((x, y))
                pixels.append(str(round(v / 255.0, 4)))
        csv_line = ",".join(pixels)
        tmp = self._tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="")
        tmp.write(csv_line + "\n")
        tmp.close()
        return tmp.name

    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        """MNIST 推理页字段表（P2-6）。

        **只收普通参数行**：`input_method` 的三态（单个文件 / 目录 / 画板绘制）
        各自带自定义控件（文件行、目录行、画板 + 预览），其取值与 `input` 的映射
        在 `collect_args` 里显式写（这类"按选项切换整块 UI"的语义表表达不了）。
        `topk` / `show_pixels` / `engine` 在自定义块**之后**建行，故 `fields()` 里
        声明、由 `_build_fields` 一次性建成（顺序 = 表顺序）。
        """
        return (
            _file("model_path", "模型路径", arg="model",
                  filetypes=[("模型文件", "*.nnmodel *.bin *.onnx")]),
            _o("input_method", "输入方式", ["单个文件", "目录", "画板绘制"], "单个文件",
               ui_only=True),
            _i("topk", "显示TopK", "3"),
            _b("show_pixels", "显示像素 (调试)", emit="always"),
            _o("engine", "计算引擎", ENGINE_OPTIONS, "CPU", ui_only=True),
            _v("gpu", "engine == GPU (Vulkan) 时才发"),
            _v("input", "由输入方式（文件/目录/画板 CSV）决定"),
        )

    def collect_args(self):
        args = collect_fields(self.fields(), self._value_of, self._field_active)
        args.pop("input_method", None)   # UI 状态，不进命令
        # 输入方式的三态各自带自定义控件（画板会现场导出 CSV）——表表达不了，显式写
        method = self._w["input_method"].get()
        if method == "画板绘制":
            args["input"] = self._save_canvas_as_csv()
            self._update_preview_from_buffer()
        elif method == "单个文件":
            if self.single_file_path.get():
                args["input"] = self.single_file_path.get()
                self._update_preview_from_csv(self.single_file_path.get())
        elif method == "目录":
            if self.dir_path.get():
                args["input"] = self.dir_path.get()
        if args.pop("engine", "CPU") == "GPU (Vulkan)":
            args["gpu"] = True
        return args


# ================================================================
#  Tab 3: 分词器训练
# ================================================================
class TokenizerTrainTab(TabBase):
    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        return (
            _file("text_file", "训练文本文件", filetypes=[("文本文件", "*.txt")]),
            _o("tokenizer_type", "分词器类型", TOKENIZER_OPTIONS, "bpe", arg="tokenizer"),
            _save("output_path", "输出路径", "bpe_vocab.nnvocab", arg="output",
                  filetypes=[("词表文件", "*.nnvocab")]),
            _i("vocab_size", "词表大小", "5000"),
            _i("min_freq", "最小频率", "2"),
            _i("threads", "预分词线程数 (0=自动)", "0"),
        )

    def __init__(self, master):
        super().__init__(master, TokenizerTrainController())
        self._build_fields(self.params_frame, 0)


# ================================================================
#  Tab 4: 分词器推理
# ================================================================
class TokenizerInferTab(TabBase):
    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        return (
            _file("vocab_path", "词表路径", arg="vocab",
                  filetypes=[("词表文件", "*.nnvocab")]),
            _e("encode_text", "编码文本", "", arg="encode"),
            _e("decode_ids", "解码ID列表", "", arg="decode"),
            _file("encode_file", "编码文件", filetypes=[("文本文件", "*.txt")]),
            # 位置参数：直接给文本文件即统计每行 token 数（--top 控制排行榜行数）
            _file("stats_file", "统计文件 (最长行 token 数)", arg="text_file",
                  filetypes=[("文本文件", "*.txt")]),
            _i("top", "最长行排行榜 (top)", "10"),
            _i("threads", "encode 线程数 (0=自动)", "0"),
            _b("show_bytes_var", "显示字节", arg="show_bytes"),
        )

    def __init__(self, master):
        super().__init__(master, TokenizerInferController())
        self._build_fields(self.params_frame, 0)


# ================================================================
#  Tab 5: GPT 训练
# ================================================================
class GptTrainTab(TabBase):
    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        """GPT/RAPT 训练页字段表（P2-6）。

        group 语义（见 `_field_active`，与旧手写回调逐条对齐）：
          sched       = 非 fixed 调度（min_lr / max_norm）
          cosine      = 仅 cosine（warmup_epochs）
          step_cosine = 仅 step_cosine（warmup_steps）
        注意两个"总是发送"的口径与旧代码一致：GPU 保护/日志的整数（flush_interval /
        checkpoint_every / log_interval / save_interval）用 emit="always"，两个复选框
        （activation_offload / grad_log）也用 emit="always"（False 也要发）。
        其余整数走 skip_zero（旧 `_int(..., skip_vals=("0",))`：0 = 未设置）。
        """
        return (
            _file("text_file", "数据集文件 (.nndataset)",
                  filetypes=[("数据集文件", "*.nndataset")]),
            _save("save_path", "保存路径", "gpt_model.nnmodel", arg="save"),
            _file("resume_path", "恢复路径 (训练断点 .nnckpt)", arg="resume",
                  filetypes=[("训练断点", "*.nnckpt *.bin *.nnpkg")]),
            _i("resume_epoch", "续训起始 epoch (0-based)", "0", skip_zero=True),
            _i("resume_step", "续训起始 step (0-based)", "0", skip_zero=True),
            _sep("── 训练参数 ──"),
            _i("epochs", "轮数", "10", skip_zero=True),
            _i("batch_size", "批大小", "32", skip_zero=True),
            _i("accum_steps", "梯度累积步数", "1", skip_zero=True),
            _i("seq_len", "序列长度", "256", skip_zero=True),
            _i("stride", "滑动窗口步长 (0=seq_len)", "0", skip_zero=True),
            _o("optimizer", "优化器", optimizer_options("text_train"), "adam"),
            _fl("weight_decay", "权重衰减", "0.01"),
            _fl("beta1", "Adam β1 (空=默认 0.9)"),
            _fl("beta2", "Adam β2 (空=默认 0.999)"),
            _o("engine", "计算引擎", ENGINE_OPTIONS, "CPU", ui_only=True),
            _sep("── 学习率调度 ──"),
            _fl("lr", "最大学习率 (lr)", "0.001"),
            _o("lr_schedule", "调度类型", GPT_LR_SCHEDULE_OPTIONS, "fixed"),
            _fl("min_lr", "最小学习率 (min_lr)", "1e-6", group="sched"),
            _i("warmup_epochs", "预热轮数 (cosine)", "0", group="cosine", skip_zero=True),
            _i("warmup_steps", "预热步数 (step_cosine)", "0", group="step_cosine",
               skip_zero=True),
            _fl("max_norm", "梯度裁剪 (0=不裁剪)", "0", group="sched"),
            _e("lr_per_epoch", "每轮 lr (逗号分隔, 可选)"),
            _sep("── 限步 ──"),
            _i("max_steps", "最大训练步数 (0=不限)", "0", skip_zero=True),
            _sep("── 混合精度 ──"),
            _o("precision_preset", "精度预设", PRECISION_PRESETS, PRECISION_PRESETS[0],
               ui_only=True),
            _o("precision_param", "参数精度 (param)", PRECISION_PARAM_OPTIONS, "f32"),
            _o("precision_compute", "计算精度 (compute)", PRECISION_OPTIONS, "f32"),
            _o("precision_stable", "稳定精度 (stable)", PRECISION_OPTIONS, "f32"),
            _o("precision_optimizer", "优化器精度 (optimizer)", PRECISION_OPTIONS, "f32"),
            _sep("── 模型参数 ──"),
            _i("d_model", "d_model", "128", skip_zero=True),
            _i("num_heads", "num_heads", "4", skip_zero=True),
            _i("num_layers", "num_layers", "4", skip_zero=True),
            _i("d_ff", "d_ff", "512", skip_zero=True),
            _sep("── 架构细节 ──"),
            _o("positional_encoding", "位置编码", POSITIONAL_ENCODING_OPTIONS, "learned"),
            _o("activation", "FFN 激活", ACTIVATION_OPTIONS, "gelu"),
            _o("norm_type", "归一化层", GPT_NORM_OPTIONS, "layernorm", arg="norm"),
            _o("model_type", "模型架构", GPT_MODEL_OPTIONS, "gpt", arg="model"),
            _sep("── GPU 保护 ──"),
            _i("flush_interval", "flush 间隔", "1", emit="always"),
            _i("checkpoint_every", "梯度检查点间隔", "0", emit="always"),
            _b("activation_offload_var", "activation offload", arg="activation_offload",
               emit="always"),
            _sep("── 日志与保存 ──"),
            _i("log_interval", "日志间隔 (steps)", "50", emit="always"),
            _i("save_interval", "保存间隔 (steps)", "100", emit="always"),
            _b("grad_log_var", "梯度日志", arg="grad_log", emit="always"),
            _v("gpu", "engine == GPU (Vulkan) 时才发"),
            _v("f16", "精度预设 = f16"),
            _v("t1_58", "精度预设 = t1_58"),
        )

    def __init__(self, master):
        super().__init__(master, GptTrainController(),
                         chart_configs=[
                             {"title": "Loss", "metrics": ["batch_loss", "train_loss", "test_loss"]},
                             {"title": "Perplexity", "metrics": ["perplexity"]},
                         ])
        # 参数行全部来自声明式字段表（P2-6）
        self._build_fields(self.params_frame, 0)
        self._w["lr_schedule"].trace_add("write", lambda *a: self._apply_groups())
        self._w["precision_preset"].trace_add("write", self._on_precision_preset_change)
        self._apply_groups()
        self._on_precision_preset_change()
        self._add_export_button()

    def build_pack_config(self):
        """把当前 GPT 训练 UI 参数打包为训练包配置"""
        args = self.collect_args()
        data = {}
        for k, role in (("text_file", "train"),):
            if args.get(k):
                data[role] = args.pop(k)
        dev = "cpu"
        if args.pop("gpu", False):
            dev = "gpu"
        name = Path(args.get("save", "gpt_model.nnmodel")).stem or "gpt_run"
        return {"format_version": 1, "name": name, "task": "gpt",
                "device": dev, "data": data, "hyperparameters": args}

    def _field_active(self, f: Field) -> bool:
        sched = self._w["lr_schedule"].get()
        if f.group == "sched":
            return sched != "fixed"
        if f.group == "cosine":
            return sched == "cosine"
        if f.group == "step_cosine":
            return sched == "step_cosine"
        return True

    def _on_precision_preset_change(self, *args):
        """精度预设切换时自动填充各精度字段（与 CLI 的 --f16 / --t1_58 逐字段对齐）。

        text_train 的 `--f16` = profile_f16() = {param:F16, compute:F16, stable:F32,
        optimizer:F32}（**不是** mnist_train 的 master-weights 配方）；`--t1_58` 就是
        --f16 再把 param 换成三值。语义见 precision.hpp。
        """
        preset = self._w["precision_preset"].get()
        if "t1_58" in preset:
            param, compute = "t1_58", "f16"
        elif "f16" in preset:
            param, compute = "f16", "f16"
        else:
            param, compute = "f32", "f32"
        self._set_field("precision_param", param)
        self._set_field("precision_compute", compute)
        self._set_field("precision_stable", "f32")
        self._set_field("precision_optimizer", "f32")

    def collect_args(self):
        """表驱动收集 + 两处后处理（engine → --gpu；preset → --f16/--t1_58 别名）"""
        args = collect_fields(self.fields(), self._value_of, self._field_active)
        if args.pop("engine", "CPU") == "GPU (Vulkan)":
            args["gpu"] = True
        preset = args.pop("precision_preset", PRECISION_PRESETS[0])
        if "t1_58" in preset or "f16" in preset:
            args["t1_58" if "t1_58" in preset else "f16"] = True
            for slot in ("precision_param", "precision_compute", "precision_stable",
                         "precision_optimizer"):
                args.pop(slot, None)
        return args


# ================================================================
#  Tab 6: GPT 推理
# ================================================================
class GptInferTab(TabBase):
    @classmethod
    def fields(cls) -> Tuple[Field, ...]:
        """GPT 推理页字段表（P2-6）。prompt 用多行文本框（kind="text"；
        `_value_of` 会 strip，空则不发 —— 与旧 `get("1.0","end").strip()` 同）。
        采样开关留空/为 0 时不发（`--top-k 0` 与不传等价，CLI 侧默认即关）。"""
        return (
            _file("model_path", "模型路径", arg="model",
                  filetypes=[("模型文件", "*.nnmodel *.bin *.pt *.nnpkg")]),
            _file("vocab_path", "词表路径 (模型未嵌入时)", arg="vocab",
                  filetypes=[("词表文件", "*.nnvocab")]),
            _i("max_tokens", "最大生成token数", "200"),
            _fl("temperature", "温度 (0=贪心)", "1.0"),
            _i("top_k", "top-k (0=关闭)", "0"),
            _fl("top_p", "top-p (1.0=关闭)", "1.0"),
            _fl("repeat_penalty", "重复惩罚 (1.0=关闭)", "1.0"),
            _i("repeat_window", "重复惩罚窗口 (0=全部)", "0"),
            _i("seed", "随机种子 (留空=随机)"),
            _o("engine", "计算引擎", ENGINE_OPTIONS, "CPU", ui_only=True),
            _b("show_tokens_var", "显示token", arg="show_tokens", emit="always"),
            _sep("── 提示文本 (prompt) ──"),
            Field("prompt_box", "", "text", "Once upon a time", arg="prompt", width=340),
            _v("gpu", "engine == GPU (Vulkan) 时才发"),
        )

    def __init__(self, master):
        super().__init__(master, GptInferController())
        self._build_fields(self.params_frame, 0)

    def collect_args(self):
        """表驱动收集 + 一处后处理（engine → --gpu）"""
        args = collect_fields(self.fields(), self._value_of, self._field_active)
        if args.pop("engine", "CPU") == "GPU (Vulkan)":
            args["gpu"] = True
        return args


# ================================================================
#  主窗口
# ================================================================
class NeuralNetApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("neuralnet.cpp 控制台")
        self.geometry("1100x780")
        self.minsize(950, 650)

        # --- 顶部标题 ---
        header = ctk.CTkFrame(self, height=50, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="🧠 neuralnet.cpp",
                      font=("", 20, "bold")).pack(side="left", padx=16, pady=8)
        ctk.CTkLabel(header, text="C++26 神经网络框架 · 训练 & 推理控制台",
                      text_color="#888", font=("", 13)).pack(side="left", padx=8)

        # --- 主体 TabView ---
        self.tabview = ctk.CTkTabview(self, corner_radius=8)
        self.tabview.pack(fill="both", expand=True, padx=12, pady=8)

        tabs = [
            ("MNIST 训练", MnistTrainTab),
            ("MNIST 推理", MnistInferTab),
            ("分词器训练", TokenizerTrainTab),
            ("分词器推理", TokenizerInferTab),
            ("GPT 训练",   GptTrainTab),
            ("GPT 推理",   GptInferTab),
        ]
        for name, cls in tabs:
            tab = self.tabview.add(name)
            cls(tab)

    def on_closing(self):
        self.destroy()


# ================================================================
#  入口
# ================================================================
if __name__ == "__main__":
    app = NeuralNetApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
