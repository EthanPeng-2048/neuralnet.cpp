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
import threading
import time
import os
import sys
import re
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
ARCH_OPTIONS = ["mlp", "transformer"]
OPTIMIZER_OPTIONS = ["sgd", "sgd_momentum", "adam", "adamw", "muon"]
ENGINE_OPTIONS = ["CPU", "GPU (Vulkan)", "CUDA"]
LR_SCHEDULE_OPTIONS = ["fixed", "cosine"]
GPT_LR_SCHEDULE_OPTIONS = ["fixed", "cosine", "step_cosine"]
TOKENIZER_OPTIONS = ["bpe", "charbpe"]
POSITIONAL_ENCODING_OPTIONS = ["learned", "sinusoidal", "alibi", "rope"]
ACTIVATION_OPTIONS = ["gelu", "swiglu"]
NORM_OPTIONS = ["layernorm", "rmsnorm", "batchnorm"]
GPT_NORM_OPTIONS = ["layernorm", "rmsnorm"]  # GPT 仅支持 LayerNorm/RMSNorm


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
class TabBase(ctk.CTkFrame):
    """标签页基类：左右分栏（参数 | 输出）"""

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

    # subclass hook ------------------------------------------------
    def collect_args(self) -> Dict[str, Any]:
        return {}

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

        p = self.params_frame
        r = 0

        # --- 基本参数 ---
        self.arch = _make_option_row(p, "架构", r, ARCH_OPTIONS, "transformer"); r += 1
        self.dataset = _make_dir_row(p, "数据集目录", r)
        self.dataset.delete(0, "end")
        self.dataset.insert(0, str(DATASETS_DIR / "mnist_data"))
        r += 1
        self.save_path = _make_save_row(p, "保存路径", r,
                                         default_name="pretrained/mnist_model.bin"); r += 1
        self.resume_path = _make_file_row(p, "恢复路径", r,
                                           filetypes=[("模型文件", "*.bin")]); r += 1
        self.epochs = _make_entry_row(p, "轮数 (epochs)", r, "10"); r += 1
        self.batch_size = _make_entry_row(p, "批大小", r, "32"); r += 1
        self.optimizer = _make_option_row(p, "优化器", r, OPTIMIZER_OPTIONS, "adam"); r += 1
        self.weight_decay = _make_entry_row(p, "权重衰减", r, "0.0"); r += 1
        self.engine = _make_option_row(p, "计算引擎", r, ENGINE_OPTIONS, "CPU"); r += 1
        self.max_samples = _make_entry_row(p, "最大样本数 (0=全部)", r, "0"); r += 1

        # --- 学习率调度 ---
        self.lr_sep_label = _make_label(p, "── 学习率调度 ──", r); r += 1
        self.lr = _make_entry_row(p, "最大学习率 (lr)", r, "0.001"); r += 1
        self.lr_schedule = _make_option_row(p, "调度类型", r, LR_SCHEDULE_OPTIONS, "fixed"); r += 1
        self.min_lr = _make_entry_row(p, "最小学习率 (min_lr)", r, "1e-6"); r += 1
        self.warmup_epochs = _make_entry_row(p, "预热轮数", r, "0"); r += 1

        # --- MLP 专用参数 ---
        self.mlp_sep_label = _make_label(p, "── MLP 参数 ──", r); r += 1
        self.layer_dims = _make_entry_row(p, "layer_dims", r, "784,512,256,128,64,10"); r += 1
        self.norm = _make_option_row(p, "归一化层", r, NORM_OPTIONS, "layernorm"); r += 1
        self._mlp_widgets = [self.mlp_sep_label, self.layer_dims, self.norm]

        # --- Transformer 专用参数 ---
        self.tf_sep_label = _make_label(p, "── Transformer 参数 ──", r); r += 1
        self.d_model = _make_entry_row(p, "d_model", r, "128"); r += 1
        self.num_heads = _make_entry_row(p, "num_heads", r, "4"); r += 1
        self.num_layers = _make_entry_row(p, "num_layers", r, "4"); r += 1
        self.d_ff = _make_entry_row(p, "d_ff", r, "512"); r += 1
        self.patch_size = _make_entry_row(p, "patch_size", r, ""); r += 1
        self.eval_samples = _make_entry_row(p, "eval_samples", r, "100"); r += 1
        self._tf_widgets = [self.tf_sep_label, self.d_model, self.num_heads,
                            self.num_layers, self.d_ff, self.patch_size, self.eval_samples]

        # --- LR 调度相关控件（fixed 时隐藏） ---
        self._lr_schedule_widgets = [self.min_lr, self.warmup_epochs]

        # 架构/调度切换回调
        self.arch.trace_add("write", self._on_arch_change)
        self.lr_schedule.trace_add("write", self._on_lr_schedule_change)
        self._on_arch_change()
        self._on_lr_schedule_change()

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
        if args.pop("cuda", False):
            dev = "cuda"
        name = Path(args.get("save", "mnist_model.bin")).stem or "mnist_run"
        return {"format_version": 1, "name": name, "task": "mnist",
                "device": dev, "data": data, "hyperparameters": args}

    def _on_arch_change(self, *args):
        """根据架构选择显示 MLP 或 Transformer 参数"""
        is_mlp = self.arch.get() == "mlp"
        for w in self._mlp_widgets:
            w = getattr(w, "widget", w)
            w.grid() if is_mlp else w.grid_remove()
        for w in self._tf_widgets:
            w = getattr(w, "widget", w)
            w.grid() if not is_mlp else w.grid_remove()

    def _on_lr_schedule_change(self, *args):
        """fixed 调度下隐藏 min_lr / warmup"""
        is_fixed = self.lr_schedule.get() == "fixed"
        for w in self._lr_schedule_widgets:
            w = getattr(w, "widget", w)
            if is_fixed:
                w.grid_remove()
            else:
                w.grid()

    def collect_args(self):
        args = {"arch": self.arch.get()}
        if self.dataset.get(): args["dataset"] = self.dataset.get()
        if self.save_path.get(): args["save"] = self.save_path.get()
        if self.resume_path.get(): args["resume"] = self.resume_path.get()
        if self.epochs.get(): args["epochs"] = int(self.epochs.get())
        if self.lr.get(): args["lr"] = float(self.lr.get())
        if self.batch_size.get(): args["batch_size"] = int(self.batch_size.get())
        args["optimizer"] = self.optimizer.get()
        if self.weight_decay.get(): args["weight_decay"] = float(self.weight_decay.get())
        engine = self.engine.get()
        if engine == "GPU (Vulkan)":
            args["gpu"] = True
        elif engine == "CUDA":
            args["cuda"] = True
        if self.max_samples.get(): args["max_samples"] = int(self.max_samples.get())
        if self.lr_schedule.get(): args["lr_schedule"] = self.lr_schedule.get()
        if self.min_lr.get(): args["min_lr"] = float(self.min_lr.get())
        if self.warmup_epochs.get(): args["warmup_epochs"] = int(self.warmup_epochs.get())
        # 架构特定参数
        if self.arch.get() == "mlp":
            if self.layer_dims.get(): args["layer_dims"] = self.layer_dims.get()
            args["norm"] = self.norm.get()
        else:
            if self.d_model.get(): args["d_model"] = int(self.d_model.get())
            if self.num_heads.get(): args["num_heads"] = int(self.num_heads.get())
            if self.num_layers.get(): args["num_layers"] = int(self.num_layers.get())
            if self.d_ff.get(): args["d_ff"] = int(self.d_ff.get())
            if self.patch_size.get(): args["patch_size"] = int(self.patch_size.get())
            if self.eval_samples.get(): args["eval_samples"] = int(self.eval_samples.get())
        return args


# ================================================================
#  Tab 2: MNIST 推理
# ================================================================
class MnistInferTab(TabBase):
    def __init__(self, master):
        super().__init__(master, MnistInferController())

        import tempfile
        import tkinter as _tk
        self._tempfile = tempfile
        self._tk = _tk

        p = self.params_frame
        r = 0
        self.model_path = _make_file_row(p, "模型路径", r,
                                          filetypes=[("模型文件", "*.bin *.onnx")]); r += 1
        self.input_method = _make_option_row(p, "输入方式", r,
                                              ["单个文件", "目录", "画板绘制"],
                                              "单个文件"); r += 1

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
        self.topk = _make_entry_row(p, "显示TopK", r, "3"); r += 1
        self.show_pixels = _make_checkbox_row(p, "显示像素 (调试)", r); r += 1
        self.engine = _make_option_row(p, "计算引擎", r, ENGINE_OPTIONS, "CPU"); r += 1

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
        self._strokes = []  # 持久化笔迹，供 _draw_grid 重放，避免重绘时丢失
        self._canvas_w = 1
        self._canvas_h = 1

        # 绑定鼠标事件
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        # 输入方式切换回调
        self.input_method.trace_add("write", self._on_input_method_change)
        self._on_input_method_change()

    def _on_canvas_configure(self, event):
        """画布尺寸变化时重绘网格并更新缩放比例"""
        self._canvas_w = max(event.width, 1)
        self._canvas_h = max(event.height, 1)
        self._draw_grid()

    def _on_input_method_change(self, *args):
        """切换输入方式时显示/隐藏对应控件：单个文件 / 目录 / 画板"""
        method = self.input_method.get()
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
        for s in self._strokes:
            sw = s.get("w") or 1
            sh = s.get("h") or 1
            sx = self._canvas_w / sw
            sy = self._canvas_h / sh
            x1, y1, x2, y2 = s["coords"]
            if s["type"] == "oval":
                self.canvas.create_oval(x1 * sx, y1 * sy, x2 * sx, y2 * sy,
                                         fill="white", outline="white")
            else:
                width = max(1, int(s.get("width", self._brush_size) * sx))
                self.canvas.create_line(x1 * sx, y1 * sy, x2 * sx, y2 * sy,
                                         fill="white", width=width,
                                         capstyle="round", smooth=True)

    def _canvas_to_pixel(self, cx, cy):
        """将画布坐标映射到 28x28 像素缓冲坐标"""
        px = int(cx / self._canvas_w * 28)
        py = int(cy / self._canvas_h * 28)
        return max(0, min(27, px)), max(0, min(27, py))

    def _on_draw_start(self, event):
        self._last_x = event.x
        self._last_y = event.y
        # 画圆点到画布，并记录笔迹
        br = self._brush_size // 2
        coords = (event.x - br, event.y - br, event.x + br, event.y + br)
        self.canvas.create_oval(*coords, fill="white", outline="white")
        self._strokes.append({"type": "oval", "coords": coords,
                               "w": self._canvas_w, "h": self._canvas_h})
        # 绘制到高分辨率缓冲（保持画布精度，后续降采样时产生灰度）
        from PIL import ImageDraw as _Draw
        draw = _Draw.Draw(self._pixel_buf_hires)
        draw.ellipse(coords, fill=255)

    def _on_draw_move(self, event):
        if self._last_x is not None:
            # 画布上线条，并记录笔迹
            coords = (self._last_x, self._last_y, event.x, event.y)
            self.canvas.create_line(*coords, fill="white",
                                     width=self._brush_size,
                                     capstyle="round", smooth=True)
            self._strokes.append({"type": "line", "coords": coords,
                                   "width": self._brush_size,
                                   "w": self._canvas_w, "h": self._canvas_h})
            # 绘制到高分辨率缓冲（保持画布精度，后续降采样时产生灰度）
            from PIL import ImageDraw as _Draw
            draw = _Draw.Draw(self._pixel_buf_hires)
            draw.line(coords, fill=255, width=self._brush_size)
        self._last_x = event.x
        self._last_y = event.y

    def _on_draw_end(self, event):
        self._last_x = None
        self._last_y = None
        self._update_preview_from_buffer()

    def _clear_canvas(self):
        self._strokes.clear()
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
        self._pixel_buf = self._pixel_buf_hires.resize((28, 28), self._pil.BILINEAR)

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

    def collect_args(self):
        args = {}
        if self.model_path.get(): args["model"] = self.model_path.get()
        # 根据输入方式决定 input
        method = self.input_method.get()
        if method == "画板绘制":
            tmp_path = self._save_canvas_as_csv()
            args["input"] = tmp_path
            self._update_preview_from_buffer()
        elif method == "单个文件":
            if self.single_file_path.get():
                args["input"] = self.single_file_path.get()
                self._update_preview_from_csv(self.single_file_path.get())
        elif method == "目录":
            if self.dir_path.get():
                args["input"] = self.dir_path.get()
        if self.topk.get(): args["topk"] = int(self.topk.get())
        args["show_pixels"] = self.show_pixels.get()
        engine = self.engine.get()
        if engine == "GPU (Vulkan)":
            args["gpu"] = True
        elif engine == "CUDA":
            args["cuda"] = True
        return args


# ================================================================
#  Tab 3: 分词器训练
# ================================================================
class TokenizerTrainTab(TabBase):
    def __init__(self, master):
        super().__init__(master, TokenizerTrainController())

        p = self.params_frame
        r = 0
        self.text_file = _make_file_row(p, "训练文本文件", r,
                                         filetypes=[("文本文件", "*.txt")]); r += 1
        self.tokenizer_type = _make_option_row(p, "分词器类型", r,
                                                TOKENIZER_OPTIONS, "bpe"); r += 1
        self.output_path = _make_save_row(p, "输出路径", r,
                                           default_name="bpe_vocab.json",
                                           filetypes=[("JSON", "*.json")]); r += 1
        self.vocab_size = _make_entry_row(p, "词表大小", r, "5000"); r += 1
        self.min_freq = _make_entry_row(p, "最小频率", r, "2"); r += 1

    def collect_args(self):
        args = {}
        if self.text_file.get(): args["text_file"] = self.text_file.get()
        args["tokenizer"] = self.tokenizer_type.get()
        if self.output_path.get(): args["output"] = self.output_path.get()
        if self.vocab_size.get(): args["vocab_size"] = int(self.vocab_size.get())
        if self.min_freq.get(): args["min_freq"] = int(self.min_freq.get())
        return args


# ================================================================
#  Tab 4: 分词器推理
# ================================================================
class TokenizerInferTab(TabBase):
    def __init__(self, master):
        super().__init__(master, TokenizerInferController())

        p = self.params_frame
        r = 0
        self.vocab_path = _make_file_row(p, "词表路径", r,
                                          filetypes=[("JSON", "*.json")]); r += 1
        self.encode_text = _make_entry_row(p, "编码文本", r, "", width=200); r += 1
        self.decode_ids = _make_entry_row(p, "解码ID列表", r, "", width=200); r += 1
        self.encode_file = _make_file_row(p, "编码文件", r,
                                           filetypes=[("文本文件", "*.txt")]); r += 1
        self.show_bytes_var = _make_checkbox_row(p, "显示字节", r); r += 1

    def collect_args(self):
        args = {}
        if self.vocab_path.get(): args["vocab"] = self.vocab_path.get()
        if self.encode_text.get(): args["encode"] = self.encode_text.get()
        if self.decode_ids.get(): args["decode"] = self.decode_ids.get()
        if self.encode_file.get(): args["encode_file"] = self.encode_file.get()
        args["show_bytes"] = self.show_bytes_var.get()
        return args


# ================================================================
#  Tab 5: GPT 训练
# ================================================================
class GptTrainTab(TabBase):
    def __init__(self, master):
        super().__init__(master, GptTrainController(),
                         chart_configs=[
                             {"title": "Loss", "metrics": ["batch_loss", "train_loss", "test_loss"]},
                             {"title": "Perplexity", "metrics": ["perplexity"]},
                         ])

        p = self.params_frame
        r = 0

        # --- 基本参数 ---
        self.text_file = _make_file_row(p, "训练文本文件", r,
                                         filetypes=[("文本文件", "*.txt")]); r += 1
        self.save_path = _make_save_row(p, "保存路径", r,
                                         default_name="gpt_model.bin"); r += 1
        self.resume_path = _make_file_row(p, "恢复路径", r,
                                           filetypes=[("模型文件", "*.bin")]); r += 1
        self.test_file = _make_file_row(p, "测试集 (可选)", r,
                                         filetypes=[("文本文件", "*.txt")]); r += 1
        self.vocab_path = _make_file_row(p, "词表路径", r,
                                          filetypes=[("JSON", "*.json")]); r += 1

        # --- 训练参数 ---
        _make_label(p, "── 训练参数 ──", r); r += 1
        self.epochs = _make_entry_row(p, "轮数", r, "10"); r += 1
        self.batch_size = _make_entry_row(p, "批大小", r, "4"); r += 1
        self.accum_steps = _make_entry_row(p, "梯度累积步数", r, "1"); r += 1
        self.seq_len = _make_entry_row(p, "序列长度", r, "256"); r += 1
        self.stride = _make_entry_row(p, "滑动窗口步长 (0=seq_len)", r, "0"); r += 1
        self.optimizer = _make_option_row(p, "优化器", r, OPTIMIZER_OPTIONS, "adam"); r += 1
        self.weight_decay = _make_entry_row(p, "权重衰减", r, "0.01"); r += 1
        self.engine = _make_option_row(p, "计算引擎", r, ENGINE_OPTIONS, "CPU"); r += 1

        # --- 学习率调度 ---
        _make_label(p, "── 学习率调度 ──", r); r += 1
        self.lr = _make_entry_row(p, "最大学习率 (lr)", r, "0.001"); r += 1
        self.lr_schedule = _make_option_row(p, "调度类型", r,
                                             GPT_LR_SCHEDULE_OPTIONS, "fixed"); r += 1
        self.min_lr = _make_entry_row(p, "最小学习率 (min_lr)", r, "1e-6"); r += 1
        self.warmup_epochs = _make_entry_row(p, "预热轮数 (cosine)", r, "0"); r += 1
        self.warmup_steps = _make_entry_row(p, "预热步数 (step_cosine)", r, "0"); r += 1
        self.max_norm = _make_entry_row(p, "梯度裁剪 (0=不裁剪)", r, "0"); r += 1

        # LR 调度相关控件（fixed 时隐藏）
        self._lr_schedule_widgets = [self.min_lr, self.warmup_epochs,
                                      self.warmup_steps, self.max_norm]
        self.lr_schedule.trace_add("write", self._on_lr_schedule_change)
        self._on_lr_schedule_change()

        # --- 模型参数 ---
        _make_label(p, "── 模型参数 ──", r); r += 1
        self.d_model = _make_entry_row(p, "d_model", r, "256"); r += 1
        self.num_heads = _make_entry_row(p, "num_heads", r, "4"); r += 1
        self.num_layers = _make_entry_row(p, "num_layers", r, "4"); r += 1
        self.d_ff = _make_entry_row(p, "d_ff", r, "1024"); r += 1

        # --- 架构细节 ---
        _make_label(p, "── 架构细节 ──", r); r += 1
        self.positional_encoding = _make_option_row(p, "位置编码", r,
                                                      POSITIONAL_ENCODING_OPTIONS, "learned"); r += 1
        self.activation = _make_option_row(p, "FFN 激活", r,
                                            ACTIVATION_OPTIONS, "gelu"); r += 1
        self.norm_type = _make_option_row(p, "归一化层", r,
                                           GPT_NORM_OPTIONS, "layernorm"); r += 1

        # --- GPU 保护 ---
        _make_label(p, "── GPU 保护 ──", r); r += 1
        self.tdr_retry = _make_option_row(p, "TDR 超时重试", r,
                                           ["on", "off"], "on"); r += 1
        self.max_tdr_retries = _make_entry_row(p, "最大重试次数", r, "4"); r += 1
        self.flush_interval = _make_entry_row(p, "flush 间隔", r, "0"); r += 1
        self.checkpoint_every = _make_entry_row(p, "梯度检查点间隔", r, "0"); r += 1

        # --- 日志与保存 ---
        _make_label(p, "── 日志与保存 ──", r); r += 1
        self.log_interval = _make_entry_row(p, "日志间隔 (steps)", r, "50"); r += 1
        self.save_interval = _make_entry_row(p, "保存间隔 (steps)", r, "100"); r += 1
        self.grad_log_var = _make_checkbox_row(p, "梯度日志", r); r += 1

        self._add_export_button()

    def build_pack_config(self):
        """把当前 GPT 训练 UI 参数打包为训练包配置"""
        args = self.collect_args()
        data = {}
        for k, role in (("text_file", "train"), ("test_file", "test"), ("vocab", "vocab")):
            if args.get(k):
                data[role] = args.pop(k)
        dev = "cpu"
        if args.pop("gpu", False):
            dev = "gpu"
        if args.pop("cuda", False):
            dev = "cuda"
        name = Path(args.get("save", "gpt_model.bin")).stem or "gpt_run"
        return {"format_version": 1, "name": name, "task": "gpt",
                "device": dev, "data": data, "hyperparameters": args}

    def _on_lr_schedule_change(self, *args):
        """fixed 调度下隐藏 min_lr / warmup / grad_clip"""
        sched = self.lr_schedule.get()
        is_fixed = sched == "fixed"
        is_cosine = sched in ("cosine", "step_cosine")
        for w in self._lr_schedule_widgets:
            w = getattr(w, "widget", w)
            if is_fixed:
                w.grid_remove()
            else:
                w.grid()
        # step_cosine 特有: warmup_steps 可见, warmup_epochs 隐藏
        if not is_fixed:
            if sched == "cosine":
                self.warmup_epochs.grid()
                self.warmup_steps.grid_remove()
            elif sched == "step_cosine":
                self.warmup_epochs.grid_remove()
                self.warmup_steps.grid()

    def collect_args(self):
        def _int(entry, key, skip_vals=("0",)):
            v = entry.get()
            if v and v not in skip_vals:
                return {key: int(v)}
            return {}

        def _float(entry, key):
            v = entry.get()
            if v:
                return {key: float(v)}
            return {}

        def _str(entry, key):
            v = entry.get()
            if v:
                return {key: v}
            return {}

        args = {}
        args.update(_str(self.text_file, "text_file"))
        args.update(_str(self.save_path, "save"))
        args.update(_str(self.resume_path, "resume"))
        args.update(_str(self.test_file, "test_file"))
        args.update(_str(self.vocab_path, "vocab"))
        args.update(_int(self.epochs, "epochs"))
        args.update(_int(self.batch_size, "batch_size"))
        args.update(_int(self.accum_steps, "accum_steps"))
        args.update(_int(self.seq_len, "seq_len"))
        args.update(_int(self.stride, "stride"))
        args["optimizer"] = self.optimizer.get()
        args.update(_float(self.weight_decay, "weight_decay"))
        engine = self.engine.get()
        if engine == "GPU (Vulkan)":
            args["gpu"] = True
        elif engine == "CUDA":
            args["cuda"] = True
        # 学习率
        args.update(_float(self.lr, "lr"))
        sched = self.lr_schedule.get()
        args["lr_schedule"] = sched
        if sched != "fixed":
            args.update(_float(self.min_lr, "min_lr"))
            if sched == "cosine":
                args.update(_int(self.warmup_epochs, "warmup_epochs"))
            elif sched == "step_cosine":
                args.update(_int(self.warmup_steps, "warmup_steps"))
            args.update(_float(self.max_norm, "max_norm"))
        # 模型
        args.update(_int(self.d_model, "d_model"))
        args.update(_int(self.num_heads, "num_heads"))
        args.update(_int(self.num_layers, "num_layers"))
        args.update(_int(self.d_ff, "d_ff"))
        # 架构细节
        args["positional_encoding"] = self.positional_encoding.get()
        args["activation"] = self.activation.get()
        args["norm"] = self.norm_type.get()
        # GPU 保护
        args["tdr_retry"] = self.tdr_retry.get()
        args.update(_int(self.max_tdr_retries, "max_tdr_retries"))
        args.update(_int(self.flush_interval, "flush_interval", skip_vals=()))
        args.update(_int(self.checkpoint_every, "checkpoint_every", skip_vals=()))
        # 日志
        args.update(_int(self.log_interval, "log_interval", skip_vals=()))
        args.update(_int(self.save_interval, "save_interval", skip_vals=()))
        args["grad_log"] = self.grad_log_var.get()
        return args


# ================================================================
#  Tab 6: GPT 推理
# ================================================================
class GptInferTab(TabBase):
    def __init__(self, master):
        super().__init__(master, GptInferController())

        p = self.params_frame
        r = 0
        self.model_path = _make_file_row(p, "模型路径", r,
                                          filetypes=[("模型文件", "*.bin *.pt")]); r += 1
        self.max_tokens = _make_entry_row(p, "最大生成token数", r, "200"); r += 1
        self.temperature = _make_entry_row(p, "温度 (0=贪心)", r, "0.8"); r += 1
        self.engine = _make_option_row(p, "计算引擎", r, ENGINE_OPTIONS, "CPU"); r += 1
        self.show_tokens_var = _make_checkbox_row(p, "显示token", r); r += 1

        # --- 提示文本（仅多行输入） ---
        _make_label(p, "── 提示文本 (prompt) ──", r); r += 1
        self.prompt_box = ctk.CTkTextbox(p, height=150, width=340)
        self.prompt_box.grid(row=r, column=0, columnspan=2, sticky="w",
                              padx=4, pady=4)
        self.prompt_box.insert("1.0", "Once upon a time")
        r += 1

    def collect_args(self):
        args = {}
        if self.model_path.get(): args["model"] = self.model_path.get()
        prompt = self.prompt_box.get("1.0", "end").strip()
        if prompt:
            args["prompt"] = prompt
        if self.max_tokens.get(): args["max_tokens"] = int(self.max_tokens.get())
        if self.temperature.get(): args["temperature"] = float(self.temperature.get())
        engine = self.engine.get()
        if engine == "GPU (Vulkan)":
            args["gpu"] = True
        elif engine == "CUDA":
            args["cuda"] = True
        args["show_tokens"] = self.show_tokens_var.get()
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
