"""
neuralnet.cpp GUI - 图形化训练与推理界面
依赖: Python 3.8+, tkinter (内置), Pillow (pip install Pillow)
"""

import os
import sys
import csv
import math
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ── 常量 ──────────────────────────────────────────────
BUILD_DIR = Path(__file__).parent / "build"
_EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
TRAIN_EXE = BUILD_DIR / f"mnist_train{_EXE_SUFFIX}"
INFER_EXE = BUILD_DIR / f"mnist_infer{_EXE_SUFFIX}"
TEXT_TRAIN_EXE = BUILD_DIR / f"text_train{_EXE_SUFFIX}"
TEXT_INFER_EXE = BUILD_DIR / f"text_infer{_EXE_SUFFIX}"
TOKENIZER_TRAIN_EXE = BUILD_DIR / f"tokenizer_train{_EXE_SUFFIX}"
TOKENIZER_INFER_EXE = BUILD_DIR / f"tokenizer_infer{_EXE_SUFFIX}"
DEFAULT_MODEL = Path(__file__).parent / "pretrained" / "model.bin"
DEFAULT_GPT_MODEL = Path(__file__).parent / "gpt_model.bin"
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "mnist_data"
PIXEL_SIZE = 8          # 每像素的显示尺寸 (28×28 → 224×224 画布)
IMG_DIM = 28


# ── 工具函数 ──────────────────────────────────────────
def load_csv_pixels(csv_path: str) -> list[float]:
    """读取单行 CSV，返回 784 个 [0,1] 浮点数"""
    with open(csv_path, "r") as f:
        line = f.readline().strip()
    return [float(v) for v in line.split(",")]


def draw_digit(canvas: tk.Canvas, pixels: list[float], offset_x=0, offset_y=0):
    """在 Canvas 上绘制 28×28 手写数字"""
    canvas.delete("all")
    for r in range(IMG_DIM):
        for c in range(IMG_DIM):
            v = pixels[r * IMG_DIM + c]
            gray = int(v * 255)
            color = f"#{gray:02x}{gray:02x}{gray:02x}"
            x1 = offset_x + c * PIXEL_SIZE
            y1 = offset_y + r * PIXEL_SIZE
            canvas.create_rectangle(x1, y1, x1 + PIXEL_SIZE, y1 + PIXEL_SIZE,
                                    fill=color, outline="")


# ── GUI 主类 ─────────────────────────────────────────
class NeuralNetGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("neuralnet.cpp — MNIST & GPT 训练 & 推理")
        self.resizable(False, False)

        # 主题风格
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=4)
        style.configure("TLabel", padding=2)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

        self._process: subprocess.Popen | None = None

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # ── 训练 Tab ──
        self.train_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.train_frame, text="  🏋️ 训练  ")
        self._build_train_tab()

        # ── 推理 Tab ──
        self.infer_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.infer_frame, text="  🔍 推理  ")
        self._build_infer_tab()

        # ── 图片查看 Tab ──
        self.viewer_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.viewer_frame, text="  🖼️ 图片查看  ")
        self._build_viewer_tab()

        # ── GPT 训练 Tab ──
        self.text_train_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.text_train_frame, text="  📝 GPT 训练  ")
        self._build_text_train_tab()

        # ── GPT 推理 Tab ──
        self.text_infer_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.text_infer_frame, text="  💬 GPT 推理  ")
        self._build_text_infer_tab()

        # ── 分词器训练 Tab ──
        self.tokenizer_train_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.tokenizer_train_frame, text="  🔤 分词器训练  ")
        self._build_tokenizer_train_tab()

        # ── 分词器推理 Tab ──
        self.tokenizer_infer_frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.tokenizer_infer_frame, text="  🔡 分词器推理  ")
        self._build_tokenizer_infer_tab()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================
    #  训练 Tab
    # ============================================================
    def _build_train_tab(self):
        f = self.train_frame
        row = 0

        # 数据集路径
        ttk.Label(f, text="数据集目录:").grid(row=row, column=0, sticky="w")
        self.train_dataset_var = tk.StringVar(value=str(DEFAULT_DATASET))
        ttk.Entry(f, textvariable=self.train_dataset_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_dataset).grid(row=row, column=2)
        row += 1

        # 保存路径
        ttk.Label(f, text="模型保存路径:").grid(row=row, column=0, sticky="w")
        self.train_save_var = tk.StringVar(value="mnist_model.bin")
        ttk.Entry(f, textvariable=self.train_save_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_save).grid(row=row, column=2)
        row += 1

        # 恢复训练
        self.train_resume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="从已有模型恢复训练", variable=self.train_resume_var,
                        command=self._toggle_resume).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.resume_frame = ttk.Frame(f)
        self.resume_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(self.resume_frame, text="模型路径:").pack(side="left")
        self.train_resume_path_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.resume_entry = ttk.Entry(self.resume_frame, textvariable=self.train_resume_path_var, width=48)
        self.resume_entry.pack(side="left", padx=4)
        self.resume_btn = ttk.Button(self.resume_frame, text="浏览…", command=self._browse_resume)
        self.resume_btn.pack(side="left")
        self._toggle_resume()
        row += 1

        # 超参数
        param_frame = ttk.LabelFrame(f, text="超参数", padding=6)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        ttk.Label(param_frame, text="轮数:").grid(row=0, column=0, sticky="w")
        self.train_epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(param_frame, from_=1, to=1000, width=6,
                     textvariable=self.train_epochs_var).grid(row=0, column=1, padx=(0, 16))

        ttk.Label(param_frame, text="学习率:").grid(row=0, column=2, sticky="w")
        self.train_lr_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(param_frame, from_=0.0001, to=1.0, increment=0.001, width=8,
                     textvariable=self.train_lr_var, format="%.4f").grid(row=0, column=3, padx=(0, 16))

        ttk.Label(param_frame, text="批大小:").grid(row=0, column=4, sticky="w")
        self.train_batch_var = tk.IntVar(value=64)
        ttk.Spinbox(param_frame, from_=1, to=4096, width=6,
                     textvariable=self.train_batch_var).grid(row=0, column=5)

        ttk.Label(param_frame, text="优化器:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.train_opt_var = tk.StringVar(value="adam")
        opt_combo = ttk.Combobox(param_frame, textvariable=self.train_opt_var, width=14, state="readonly",
                                 values=["sgd", "sgd_momentum", "adam", "adamw", "muon"])
        opt_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(param_frame, text="权重衰减:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.train_weight_decay_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.001, width=8,
                     textvariable=self.train_weight_decay_var, format="%.3f").grid(row=1, column=3, padx=(0, 16), pady=(6, 0))

        ttk.Label(param_frame, text="模型类型:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.train_model_type_var = tk.StringVar(value="mlp")
        model_type_combo = ttk.Combobox(param_frame, textvariable=self.train_model_type_var, width=14, state="readonly",
                                        values=["mlp", "transformer"])
        model_type_combo.grid(row=2, column=1, sticky="w", pady=(6, 0))
        model_type_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_model_params())
        row += 1

        # ── 模型架构参数（根据模型类型动态切换） ──
        self.model_params_frame = ttk.LabelFrame(f, text="模型架构", padding=6)
        self.model_params_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        # MLP 参数
        self.mlp_params = ttk.Frame(self.model_params_frame)
        ttk.Label(self.mlp_params, text="层维度 (逗号分隔):").pack(side="left")
        self.train_layer_dims_var = tk.StringVar(value="784,512,256,128,64,10")
        ttk.Entry(self.mlp_params, textvariable=self.train_layer_dims_var, width=42).pack(side="left", padx=4)

        # Transformer 参数
        self.transformer_params = ttk.Frame(self.model_params_frame)
        ttk.Label(self.transformer_params, text="模型维度:").grid(row=0, column=0, sticky="w")
        self.train_d_model_var = tk.IntVar(value=64)
        ttk.Spinbox(self.transformer_params, from_=16, to=512, width=6,
                     textvariable=self.train_d_model_var).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(self.transformer_params, text="注意力头:").grid(row=0, column=2, sticky="w")
        self.train_num_heads_var = tk.IntVar(value=4)
        ttk.Spinbox(self.transformer_params, from_=1, to=16, width=6,
                     textvariable=self.train_num_heads_var).grid(row=0, column=3, padx=(0, 12))

        ttk.Label(self.transformer_params, text="FFN维度:").grid(row=0, column=4, sticky="w")
        self.train_d_ff_var = tk.IntVar(value=128)
        ttk.Spinbox(self.transformer_params, from_=32, to=2048, width=6,
                     textvariable=self.train_d_ff_var).grid(row=0, column=5, padx=(0, 12))

        ttk.Label(self.transformer_params, text="层数:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.train_num_layers_var = tk.IntVar(value=2)
        ttk.Spinbox(self.transformer_params, from_=1, to=16, width=6,
                     textvariable=self.train_num_layers_var).grid(row=1, column=1, pady=(4, 0))

        ttk.Label(self.transformer_params, text="Patch大小:").grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.train_patch_size_var = tk.IntVar(value=7)
        ttk.Spinbox(self.transformer_params, from_=2, to=14, width=6,
                     textvariable=self.train_patch_size_var).grid(row=1, column=3, pady=(4, 0))

        self._toggle_model_params()  # 默认显示 MLP 参数
        row += 1

        # ── GPU 加速 ──
        self.train_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="启用 GPU 加速 (需要 Vulkan SDK)", variable=self.train_gpu_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # ── 打乱 batch 顺序 ──
        self.train_shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="每 epoch 打乱 batch 顺序 (--shuffle-steps)", variable=self.train_shuffle_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # ── 最大训练样本数 ──
        max_samples_frame = ttk.Frame(f)
        max_samples_frame.grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Label(max_samples_frame, text="最大训练样本数:").pack(side="left")
        self.train_max_samples_var = tk.IntVar(value=0)
        ttk.Spinbox(max_samples_frame, from_=0, to=100000, width=8,
                     textvariable=self.train_max_samples_var).pack(side="left", padx=4)
        ttk.Label(max_samples_frame, text="(0=全部)").pack(side="left")
        row += 1

        # 按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=6)
        row += 1

        self.train_start_btn = ttk.Button(btn_frame, text="▶  开始训练", command=self._start_training)
        self.train_start_btn.pack(side="left", padx=4)
        self.train_stop_btn = ttk.Button(btn_frame, text="⏹  停止", command=self._stop_training, state="disabled")
        self.train_stop_btn.pack(side="left", padx=4)
        row += 1

        # 日志输出
        self.train_log = tk.Text(f, height=14, width=72, state="disabled",
                                 font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                                 insertbackground="white")
        self.train_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.train_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.train_log["yscrollcommand"] = scroll.set

    # ---- 浏览按钮 ----
    def _browse_dataset(self):
        path = filedialog.askdirectory(title="选择数据集目录")
        if path:
            self.train_dataset_var.set(path)

    def _browse_save(self):
        path = filedialog.asksaveasfilename(title="模型保存路径", defaultextension=".bin",
                                            filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.train_save_var.set(path)

    def _browse_resume(self):
        path = filedialog.askopenfilename(title="选择恢复模型", filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.train_resume_path_var.set(path)

    def _toggle_resume(self):
        state = "normal" if self.train_resume_var.get() else "disabled"
        self.resume_entry.config(state=state)
        self.resume_btn.config(state=state)

    def _toggle_model_params(self):
        """根据模型类型显示 MLP 或 Transformer 参数"""
        if self.train_model_type_var.get() == "mlp":
            self.transformer_params.pack_forget()
            self.mlp_params.pack(fill="x")
        else:
            self.mlp_params.pack_forget()
            self.transformer_params.pack(fill="x")

    # ---- 训练 ----
    def _start_training(self):
        exe = str(TRAIN_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到训练可执行文件:\n{exe}\n请先构建项目:\n  cmake -B build -G Ninja && ninja -C build")
            return

        cmd = [exe,
               "--dataset", self.train_dataset_var.get(),
               "--save", self.train_save_var.get(),
               "--epochs", str(self.train_epochs_var.get()),
               "--lr", str(self.train_lr_var.get()),
               "--batch-size", str(self.train_batch_var.get()),
               "--optimizer", self.train_opt_var.get(),
               "--weight-decay", str(self.train_weight_decay_var.get()),
               "--arch", self.train_model_type_var.get()]
        if self.train_resume_var.get():
            cmd += ["--resume", self.train_resume_path_var.get()]

        # 最大训练样本数
        if self.train_max_samples_var.get() > 0:
            cmd += ["--max-samples", str(self.train_max_samples_var.get())]

        # 打乱 batch 顺序
        cmd += ["--shuffle-steps", str(self.train_shuffle_var.get()).lower()]

        # 模型架构参数
        if self.train_model_type_var.get() == "mlp":
            cmd += ["--layer-dims", self.train_layer_dims_var.get()]
        else:
            cmd += ["--d-model", str(self.train_d_model_var.get()),
                    "--num-heads", str(self.train_num_heads_var.get()),
                    "--d-ff", str(self.train_d_ff_var.get()),
                    "--num-layers", str(self.train_num_layers_var.get()),
                    "--patch-size", str(self.train_patch_size_var.get())]

        # GPU 加速
        if self.train_gpu_var.get():
            cmd.append("--gpu")

        self._log_train(f"$ {' '.join(cmd)}\n")
        self.train_start_btn.config(state="disabled")
        self.train_stop_btn.config(state="normal")

        threading.Thread(target=self._run_process, args=(cmd, self._log_train, self._train_done),
                         daemon=True).start()

    def _stop_training(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log_train("\n[已终止]\n")
        self._train_done()

    def _train_done(self):
        self.train_start_btn.config(state="normal")
        self.train_stop_btn.config(state="disabled")

    def _log_train(self, text: str):
        self.train_log.config(state="normal")
        self.train_log.insert("end", text)
        self.train_log.see("end")
        self.train_log.config(state="disabled")

    # ============================================================
    #  推理 Tab
    # ============================================================
    def _build_infer_tab(self):
        f = self.infer_frame
        row = 0

        # 模型路径
        ttk.Label(f, text="模型文件:").grid(row=row, column=0, sticky="w")
        self.infer_model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        ttk.Entry(f, textvariable=self.infer_model_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_model).grid(row=row, column=2)
        row += 1

        # 输入路径
        ttk.Label(f, text="图片/目录:").grid(row=row, column=0, sticky="w")
        self.infer_input_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.infer_input_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_input).grid(row=row, column=2)
        row += 1

        # Top-K + 模型类型
        param_frame = ttk.Frame(f)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        row += 1
        ttk.Label(param_frame, text="Top-K:").pack(side="left")
        self.infer_topk_var = tk.IntVar(value=3)
        ttk.Spinbox(param_frame, from_=1, to=10, width=4,
                     textvariable=self.infer_topk_var).pack(side="left", padx=4)
        ttk.Label(param_frame, text="  架构: 自动识别").pack(side="left")
        ttk.Label(param_frame, text="    ").pack(side="left")
        self.infer_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="GPU 加速", variable=self.infer_gpu_var).pack(side="left", padx=4)
        self.infer_show_pixels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="显示像素", variable=self.infer_show_pixels_var).pack(side="left", padx=4)
        row += 1

        # 按钮
        ttk.Button(f, text="▶  开始推理", command=self._start_infer).grid(row=row, column=0, columnspan=3, pady=4)
        row += 1
        # ── 手写板 ──────────────────────────────────────
        self._handwriting_infer = False
        DRAW_SCALE = 10
        DRAW_SIZE = IMG_DIM * DRAW_SCALE  # 280
        self._draw_size = DRAW_SIZE
        self._draw_scale = DRAW_SCALE
        self._brush_radius = 14
        self._draw_pixels = [0.0] * (DRAW_SIZE * DRAW_SIZE)
        self._drawing = False

        draw_frame = ttk.LabelFrame(f, text="✍️ 手写识别", padding=6)
        draw_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)

        draw_left = ttk.Frame(draw_frame)
        draw_left.pack(side="left", padx=(0, 12))

        self._draw_canvas = tk.Canvas(draw_left, width=DRAW_SIZE, height=DRAW_SIZE,
                                      bg="white", cursor="pencil", relief="sunken", bd=2)
        self._draw_canvas.pack()
        self._draw_canvas.bind("<Button-1>", self._on_draw_start)
        self._draw_canvas.bind("<B1-Motion>", self._on_draw_move)
        self._draw_canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        btn_row = ttk.Frame(draw_left)
        btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text="🗑️ 清除", command=self._clear_drawing).pack(side="left", padx=4)
        ttk.Button(btn_row, text="🔍 识别手写数字", command=self._recognize_drawing).pack(side="left", padx=4)

        # 右侧：28×28 预览
        draw_right = ttk.Frame(draw_frame)
        draw_right.pack(side="left", fill="both", expand=True)
        ttk.Label(draw_right, text="28×28 预览:", font=("Segoe UI", 9)).pack(anchor="w")
        preview_size = IMG_DIM * PIXEL_SIZE
        self._draw_preview = tk.Canvas(draw_right, width=preview_size, height=preview_size, bg="black")
        self._draw_preview.pack(pady=4)

        row += 1
        # 结果区域
        result_frame = ttk.Frame(f)
        result_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        row += 1

        # 左侧: 图片预览
        preview_frame = ttk.LabelFrame(result_frame, text="图片预览", padding=4)
        preview_frame.pack(side="left", padx=(0, 8))
        canvas_size = IMG_DIM * PIXEL_SIZE
        self.infer_canvas = tk.Canvas(preview_frame, width=canvas_size, height=canvas_size, bg="black")
        self.infer_canvas.pack()

        # 右侧: 预测结果
        right_frame = ttk.Frame(result_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        self.infer_filename_label = ttk.Label(right_frame, text="", font=("Segoe UI", 9))
        self.infer_filename_label.pack(anchor="w")

        self.infer_bars_frame = ttk.Frame(right_frame)
        self.infer_bars_frame.pack(fill="both", expand=True, pady=4)

        # 日志
        row += 1
        self.infer_log = tk.Text(f, height=10, width=72, state="disabled",
                                 font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.infer_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.infer_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.infer_log["yscrollcommand"] = scroll.set

    def _browse_model(self):
        path = filedialog.askopenfilename(title="选择模型文件", filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.infer_model_var.set(path)

    def _browse_input(self):
        path = filedialog.askopenfilename(title="选择图片 CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            path = filedialog.askdirectory(title="选择图片目录")
        if path:
            self.infer_input_var.set(path)

    # ---- 手写板事件 ----
    def _on_draw_start(self, event):
        self._drawing = True
        self._paint_at(event.x, event.y)

    def _on_draw_move(self, event):
        if self._drawing:
            self._paint_at(event.x, event.y)

    def _on_draw_end(self, event):
        self._drawing = False

    def _paint_at(self, cx, cy):
        """在画板 (cx, cy) 处绘制笔触"""
        r = self._brush_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq > r * r:
                    continue
                px, py = cx + dx, cy + dy
                if 0 <= px < self._draw_size and 0 <= py < self._draw_size:
                    dist = math.sqrt(dist_sq)
                    sigma = r / 2.0
                    intensity = math.exp(-dist_sq / (2 * sigma * sigma))
                    idx = py * self._draw_size + px
                    self._draw_pixels[idx] = max(self._draw_pixels[idx], intensity)
        self._draw_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                      fill="black", outline="")

    def _clear_drawing(self):
        """清除手写板"""
        self._draw_pixels = [0.0] * (self._draw_size * self._draw_size)
        self._draw_canvas.delete("all")
        self._draw_preview.delete("all")

    def _downsample_drawing(self) -> list[float]:
        """将 280×280 画板下采样为 28×28 像素值"""
        scale = self._draw_scale
        pixels = []
        for r in range(IMG_DIM):
            for c in range(IMG_DIM):
                total = 0.0
                for dy in range(scale):
                    for dx in range(scale):
                        y = r * scale + dy
                        x = c * scale + dx
                        total += self._draw_pixels[y * self._draw_size + x]
                avg = total / (scale * scale)
                pixels.append(avg)
        return pixels

    def _recognize_drawing(self):
        """识别手写板上的数字"""
        exe = str(INFER_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到推理可执行文件:\n{exe}\n请先构建项目")
            return

        # 检查是否有内容
        if max(self._draw_pixels) < 0.01:
            messagebox.showwarning("提示", "请先在手写板上写一个数字")
            return

        pixels = self._downsample_drawing()

        # 显示 28×28 预览
        draw_digit(self._draw_preview, pixels)
        draw_digit(self.infer_canvas, pixels)
        self.infer_filename_label.config(text="手写输入")

        # 写入临时 CSV
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        tmp.write(",".join(f"{v:.6f}" for v in pixels))
        tmp.close()

        cmd = [exe, tmp.name,
               "--model", self.infer_model_var.get(),
               "--topk", str(self.infer_topk_var.get())]
        if self.infer_show_pixels_var.get():
            cmd.append("--show-pixels")
        if self.infer_gpu_var.get():
            cmd.append("--gpu")

        self._log_infer(f"[手写识别] $ {' '.join(cmd)}\n")
        self._handwriting_infer = True

        def _run():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                self._log_infer(result.stdout)
                if result.stderr:
                    self._log_infer(result.stderr)
                self._parse_and_display_infer(result.stdout)
            except FileNotFoundError:
                self._log_infer("错误: 可执行文件未找到\n")
            except subprocess.TimeoutExpired:
                self._log_infer("错误: 推理超时\n")
            except Exception as e:
                self._log_infer(f"错误: {e}\n")
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def _start_infer(self):
        self._handwriting_infer = False
        exe = str(INFER_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到推理可执行文件:\n{exe}\n请先构建项目:\n  cmake -B build -G Ninja && ninja -C build")
            return

        input_path = self.infer_input_var.get().strip()
        if not input_path:
            messagebox.showwarning("提示", "请指定图片文件或目录")
            return

        cmd = [exe, input_path,
               "--model", self.infer_model_var.get(),
               "--topk", str(self.infer_topk_var.get())]
        if self.infer_show_pixels_var.get():
            cmd.append("--show-pixels")
        if self.infer_gpu_var.get():
            cmd.append("--gpu")

        self._log_infer(f"$ {' '.join(cmd)}\n")
        threading.Thread(target=self._run_infer_process, args=(cmd,), daemon=True).start()

    def _run_infer_process(self, cmd: list[str]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self._log_infer(result.stdout)
            if result.stderr:
                self._log_infer(result.stderr)

            # 尝试解析结果并显示图片
            self._parse_and_display_infer(result.stdout)
        except FileNotFoundError:
            self._log_infer("错误: 可执行文件未找到\n")
        except subprocess.TimeoutExpired:
            self._log_infer("错误: 推理超时\n")

    def _parse_and_display_infer(self, output: str):
        """从推理输出中解析结果并显示在 Canvas 上"""
        lines = output.strip().split("\n")
        for line in lines:
            if " -> " in line:
                # 格式: filename -> 3 (98.2%)  7 (1.2%)  5 (0.3%)
                parts = line.split(" -> ")
                if len(parts) == 2:
                    filename = parts[0].strip()
                    predictions_str = parts[1].strip()

                    # 显示文件名
                    if self._handwriting_infer:
                        self.after(0, lambda: self.infer_filename_label.config(text="手写输入"))
                    else:
                        self.after(0, lambda f=filename: self.infer_filename_label.config(text=f))

                    # 解析预测结果
                    predictions = []
                    import re
                    for m in re.finditer(r"(\d+)\s*\((\d+\.?\d*)%\)", predictions_str):
                        predictions.append((int(m.group(1)), float(m.group(2))))

                    # 显示结果条形图
                    self.after(0, lambda p=predictions: self._show_prediction_bars(p))

                    # 尝试显示图片（手写模式跳过，已有预览）
                    if not self._handwriting_infer:
                        input_path = self.infer_input_var.get().strip()
                        if os.path.isfile(input_path):
                            self.after(0, lambda p=input_path: self._display_csv_image(p))
                        elif os.path.isdir(input_path):
                            img_path = os.path.join(input_path, filename)
                            if os.path.isfile(img_path):
                                self.after(0, lambda p=img_path: self._display_csv_image(p))
                break

    def _show_prediction_bars(self, predictions: list[tuple[int, float]]):
        for w in self.infer_bars_frame.winfo_children():
            w.destroy()
        for digit, conf in predictions:
            row = ttk.Frame(self.infer_bars_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{digit}", width=2, font=("Consolas", 11, "bold")).pack(side="left")
            bar_canvas = tk.Canvas(row, height=18, bg="#333", highlightthickness=0)
            bar_canvas.pack(side="left", fill="x", expand=True, padx=4)
            bar_canvas.update_idletasks()
            w = bar_canvas.winfo_width()
            bar_canvas.create_rectangle(0, 0, int(w * conf / 100), 18,
                                        fill="#4ec9b0", outline="")
            ttk.Label(row, text=f"{conf:.1f}%", width=6).pack(side="left")

    def _display_csv_image(self, csv_path: str):
        try:
            pixels = load_csv_pixels(csv_path)
            draw_digit(self.infer_canvas, pixels)
        except Exception as e:
            self._log_infer(f"图片加载失败: {e}\n")

    def _log_infer(self, text: str):
        def _do():
            self.infer_log.config(state="normal")
            self.infer_log.insert("end", text)
            self.infer_log.see("end")
            self.infer_log.config(state="disabled")
        self.after(0, _do)

    # ============================================================
    #  GPT 训练 Tab
    # ============================================================
    def _build_text_train_tab(self):
        f = self.text_train_frame
        row = 0

        # 文本文件路径
        ttk.Label(f, text="训练文本文件:").grid(row=row, column=0, sticky="w")
        self.text_train_file_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.text_train_file_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_text_file).grid(row=row, column=2)
        row += 1

        # 保存路径
        ttk.Label(f, text="模型保存路径:").grid(row=row, column=0, sticky="w")
        self.text_train_save_var = tk.StringVar(value=str(DEFAULT_GPT_MODEL))
        ttk.Entry(f, textvariable=self.text_train_save_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_text_save).grid(row=row, column=2)
        row += 1

        # 词表路径
        ttk.Label(f, text="词表 JSON 路径:").grid(row=row, column=0, sticky="w")
        self.text_train_vocab_var = tk.StringVar(value="bpe_vocab.json")
        ttk.Entry(f, textvariable=self.text_train_vocab_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_text_train_vocab).grid(row=row, column=2)
        row += 1

        # 恢复训练
        self.text_train_resume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="从已有模型恢复训练", variable=self.text_train_resume_var,
                        command=self._toggle_text_resume).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.text_resume_frame = ttk.Frame(f)
        self.text_resume_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(self.text_resume_frame, text="模型路径:").pack(side="left")
        self.text_train_resume_path_var = tk.StringVar(value=str(DEFAULT_GPT_MODEL))
        self.text_resume_entry = ttk.Entry(self.text_resume_frame, textvariable=self.text_train_resume_path_var, width=48)
        self.text_resume_entry.pack(side="left", padx=4)
        self.text_resume_btn = ttk.Button(self.text_resume_frame, text="浏览…", command=self._browse_text_resume)
        self.text_resume_btn.pack(side="left")
        self._toggle_text_resume()
        row += 1

        # 超参数
        param_frame = ttk.LabelFrame(f, text="超参数", padding=6)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        ttk.Label(param_frame, text="轮数:").grid(row=0, column=0, sticky="w")
        self.text_train_epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=1000, width=6,
                     textvariable=self.text_train_epochs_var).grid(row=0, column=1, padx=(0, 16))

        ttk.Label(param_frame, text="学习率:").grid(row=0, column=2, sticky="w")
        self.text_train_lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(param_frame, from_=0.00001, to=1.0, increment=0.0001, width=8,
                     textvariable=self.text_train_lr_var, format="%.5f").grid(row=0, column=3, padx=(0, 16))

        ttk.Label(param_frame, text="批大小:").grid(row=0, column=4, sticky="w")
        self.text_train_batch_var = tk.IntVar(value=32)
        ttk.Spinbox(param_frame, from_=1, to=256, width=6,
                     textvariable=self.text_train_batch_var).grid(row=0, column=5)

        ttk.Label(param_frame, text="序列长度:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.text_train_seq_len_var = tk.IntVar(value=256)
        ttk.Spinbox(param_frame, from_=16, to=1024, width=6,
                     textvariable=self.text_train_seq_len_var).grid(row=1, column=1, pady=(6, 0))

        ttk.Label(param_frame, text="优化器:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.text_train_opt_var = tk.StringVar(value="adam")
        ttk.Combobox(param_frame, textvariable=self.text_train_opt_var, width=14, state="readonly",
                     values=["sgd", "sgd_momentum", "adam", "adamw", "muon"]).grid(row=1, column=3, sticky="w", pady=(6, 0))

        ttk.Label(param_frame, text="权重衰减:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.text_train_weight_decay_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.001, width=8,
                     textvariable=self.text_train_weight_decay_var, format="%.3f").grid(row=1, column=5, pady=(6, 0))

        ttk.Label(param_frame, text="日志间隔:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.text_train_log_interval_var = tk.IntVar(value=50)
        ttk.Spinbox(param_frame, from_=1, to=1000, width=6,
                     textvariable=self.text_train_log_interval_var).grid(row=2, column=1, pady=(6, 0))

        # 模型架构参数
        arch_frame = ttk.LabelFrame(f, text="模型架构", padding=6)
        arch_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        ttk.Label(arch_frame, text="模型维度:").grid(row=0, column=0, sticky="w")
        self.text_train_d_model_var = tk.IntVar(value=128)
        ttk.Spinbox(arch_frame, from_=32, to=512, width=6,
                     textvariable=self.text_train_d_model_var).grid(row=0, column=1, padx=(0, 16))

        ttk.Label(arch_frame, text="注意力头:").grid(row=0, column=2, sticky="w")
        self.text_train_num_heads_var = tk.IntVar(value=4)
        ttk.Spinbox(arch_frame, from_=1, to=16, width=6,
                     textvariable=self.text_train_num_heads_var).grid(row=0, column=3, padx=(0, 16))

        ttk.Label(arch_frame, text="Transformer 层数:").grid(row=0, column=4, sticky="w")
        self.text_train_num_layers_var = tk.IntVar(value=4)
        ttk.Spinbox(arch_frame, from_=1, to=16, width=6,
                     textvariable=self.text_train_num_layers_var).grid(row=0, column=5)

        ttk.Label(arch_frame, text="FFN 维度:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.text_train_d_ff_var = tk.IntVar(value=512)
        ttk.Spinbox(arch_frame, from_=64, to=2048, width=6,
                     textvariable=self.text_train_d_ff_var).grid(row=1, column=1, pady=(6, 0))
        row += 1

        # ── 位置编码 ──
        pe_frame = ttk.Frame(f)
        pe_frame.grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Label(pe_frame, text="位置编码:").pack(side="left")
        self.text_train_pos_enc_var = tk.StringVar(value="learned")
        ttk.Combobox(pe_frame, textvariable=self.text_train_pos_enc_var, width=14, state="readonly",
                     values=["learned", "sinusoidal", "alibi"]).pack(side="left", padx=4)
        ttk.Label(pe_frame, text="(learned=可学习, sinusoidal=正弦波, alibi=线性偏置)").pack(side="left")
        row += 1

        # ── GPU 加速 ──
        self.text_train_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="启用 GPU 加速 (需要 Vulkan SDK)", variable=self.text_train_gpu_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # ── 梯度日志 ──
        self.text_train_grad_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="显示梯度统计 (范数/最大值/均值)", variable=self.text_train_grad_log_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # 按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=6)
        row += 1

        self.text_train_start_btn = ttk.Button(btn_frame, text="▶  开始训练", command=self._start_text_training)
        self.text_train_start_btn.pack(side="left", padx=4)
        self.text_train_stop_btn = ttk.Button(btn_frame, text="⏹  停止", command=self._stop_text_training, state="disabled")
        self.text_train_stop_btn.pack(side="left", padx=4)
        row += 1

        # 日志输出
        self.text_train_log = tk.Text(f, height=14, width=72, state="disabled",
                                      font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                                      insertbackground="white")
        self.text_train_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.text_train_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.text_train_log["yscrollcommand"] = scroll.set

    # ---- GPT 训练浏览 ----
    def _browse_text_file(self):
        path = filedialog.askopenfilename(title="选择训练文本文件",
                                          filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            self.text_train_file_var.set(path)

    def _browse_text_save(self):
        path = filedialog.asksaveasfilename(title="模型保存路径", defaultextension=".bin",
                                            filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.text_train_save_var.set(path)

    def _browse_text_train_vocab(self):
        path = filedialog.askopenfilename(title="选择词表 JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.text_train_vocab_var.set(path)

    def _browse_text_resume(self):
        path = filedialog.askopenfilename(title="选择恢复模型", filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.text_train_resume_path_var.set(path)

    def _toggle_text_resume(self):
        state = "normal" if self.text_train_resume_var.get() else "disabled"
        self.text_resume_entry.config(state=state)
        self.text_resume_btn.config(state=state)

    # ---- GPT 训练 ----
    def _start_text_training(self):
        exe = str(TEXT_TRAIN_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到 text_train 可执行文件:\n{exe}\n请先构建项目")
            return

        text_file = self.text_train_file_var.get().strip()
        if not text_file:
            messagebox.showwarning("提示", "请选择训练文本文件")
            return

        cmd = [exe, text_file,
               "--save", self.text_train_save_var.get(),
               "--vocab", self.text_train_vocab_var.get(),
               "--epochs", str(self.text_train_epochs_var.get()),
               "--lr", str(self.text_train_lr_var.get()),
               "--batch-size", str(self.text_train_batch_var.get()),
               "--seq-len", str(self.text_train_seq_len_var.get()),
               "--optimizer", self.text_train_opt_var.get(),
               "--weight-decay", str(self.text_train_weight_decay_var.get()),
               "--d-model", str(self.text_train_d_model_var.get()),
               "--num-heads", str(self.text_train_num_heads_var.get()),
               "--num-layers", str(self.text_train_num_layers_var.get()),
               "--d-ff", str(self.text_train_d_ff_var.get()),
               "--log-interval", str(self.text_train_log_interval_var.get())]
        if self.text_train_resume_var.get():
            cmd += ["--resume", self.text_train_resume_path_var.get()]
        cmd += ["--positional-encoding", self.text_train_pos_enc_var.get()]
        if self.text_train_gpu_var.get():
            cmd.append("--gpu")
        if self.text_train_grad_log_var.get():
            cmd.append("--grad-log")

        self._log_text_train(f"$ {' '.join(cmd)}\n")
        self.text_train_start_btn.config(state="disabled")
        self.text_train_stop_btn.config(state="normal")

        threading.Thread(target=self._run_process, args=(cmd, self._log_text_train, self._text_train_done),
                         daemon=True).start()

    def _stop_text_training(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log_text_train("\n[已终止]\n")
        self._text_train_done()

    def _text_train_done(self):
        self.text_train_start_btn.config(state="normal")
        self.text_train_stop_btn.config(state="disabled")

    def _log_text_train(self, text: str):
        self.text_train_log.config(state="normal")
        self.text_train_log.insert("end", text)
        self.text_train_log.see("end")
        self.text_train_log.config(state="disabled")

    # ============================================================
    #  GPT 推理 Tab
    # ============================================================
    def _build_text_infer_tab(self):
        f = self.text_infer_frame
        row = 0

        # 模型路径
        ttk.Label(f, text="模型文件:").grid(row=row, column=0, sticky="w")
        self.text_infer_model_var = tk.StringVar(value=str(DEFAULT_GPT_MODEL))
        ttk.Entry(f, textvariable=self.text_infer_model_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_text_infer_model).grid(row=row, column=2)
        row += 1

        # 词表路径
        ttk.Label(f, text="词表 JSON 路径:").grid(row=row, column=0, sticky="w")
        self.text_infer_vocab_var = tk.StringVar(value="bpe_vocab.json")
        ttk.Entry(f, textvariable=self.text_infer_vocab_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_text_infer_vocab).grid(row=row, column=2)
        row += 1

        # 生成参数
        param_frame = ttk.LabelFrame(f, text="生成参数", padding=6)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        ttk.Label(param_frame, text="最大 token 数:").grid(row=0, column=0, sticky="w")
        self.text_infer_max_tokens_var = tk.IntVar(value=200)
        ttk.Spinbox(param_frame, from_=1, to=2048, width=6,
                     textvariable=self.text_infer_max_tokens_var).grid(row=0, column=1, padx=(0, 16))

        ttk.Label(param_frame, text="温度:").grid(row=0, column=2, sticky="w")
        self.text_infer_temp_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(param_frame, from_=0.0, to=2.0, increment=0.1, width=6,
                     textvariable=self.text_infer_temp_var, format="%.1f").grid(row=0, column=3, padx=(0, 16))
        row += 1

        # ── GPU 加速 ──
        self.text_infer_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="启用 GPU 加速 (需要 Vulkan SDK)", variable=self.text_infer_gpu_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # ── 显示 Token ID ──
        self.text_infer_show_tokens_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="显示 Token ID (调试用)", variable=self.text_infer_show_tokens_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # 输入提示
        ttk.Label(f, text="输入提示:").grid(row=row, column=0, sticky="w")
        self.text_infer_prompt_var = tk.StringVar(value="Hello")
        prompt_entry = ttk.Entry(f, textvariable=self.text_infer_prompt_var, width=48)
        prompt_entry.grid(row=row, column=1, padx=4)
        prompt_entry.bind("<Return>", lambda e: self._start_text_infer())
        row += 1

        # ── 交互模式 ──
        self.text_infer_interactive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="交互式生成模式 (interactive)", variable=self.text_infer_interactive_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # 按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=4)
        row += 1

        self.text_infer_start_btn = ttk.Button(btn_frame, text="▶  生成文本", command=self._start_text_infer)
        self.text_infer_start_btn.pack(side="left", padx=4)
        row += 1

        # 输出区域
        output_frame = ttk.LabelFrame(f, text="生成结果", padding=6)
        output_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        self.text_infer_output = tk.Text(output_frame, height=12, width=72, state="disabled",
                                         font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4",
                                         wrap="word")
        self.text_infer_output.pack(fill="both", expand=True)

        # 日志
        self.text_infer_log = tk.Text(f, height=6, width=72, state="disabled",
                                      font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.text_infer_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.text_infer_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.text_infer_log["yscrollcommand"] = scroll.set

    def _browse_text_infer_model(self):
        path = filedialog.askopenfilename(title="选择 GPT 模型文件",
                                          filetypes=[("Binary", "*.bin"), ("All", "*.*")])
        if path:
            self.text_infer_model_var.set(path)

    def _browse_text_infer_vocab(self):
        path = filedialog.askopenfilename(title="选择词表 JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.text_infer_vocab_var.set(path)

    # ---- GPT 推理 ----
    def _start_text_infer(self):
        exe = str(TEXT_INFER_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到 text_infer 可执行文件:\n{exe}\n请先构建项目")
            return

        prompt = self.text_infer_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示文本")
            return

        cmd = [exe,
               "--model", self.text_infer_model_var.get(),
               "--vocab", self.text_infer_vocab_var.get(),
               "--max-tokens", str(self.text_infer_max_tokens_var.get()),
               "--temperature", str(self.text_infer_temp_var.get())]
        if self.text_infer_interactive_var.get():
            cmd.append("--interactive")
        else:
            cmd += ["--prompt", prompt]
        if self.text_infer_show_tokens_var.get():
            cmd.append("--show-tokens")
        if self.text_infer_gpu_var.get():
            cmd.append("--gpu")

        self._log_text_infer(f"$ {' '.join(cmd)}\n")
        self.text_infer_start_btn.config(state="disabled")

        # 清空输出
        self.text_infer_output.config(state="normal")
        self.text_infer_output.delete("1.0", "end")
        self.text_infer_output.config(state="disabled")

        if self.text_infer_interactive_var.get():
            # 交互模式：通过 stdin 发送 prompt，然后发送 quit 退出
            def _run_interactive():
                try:
                    proc = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                    # 发送 prompt + quit
                    proc.stdin.write(prompt + "\n")
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                    output, _ = proc.communicate(timeout=120)
                    self.after(0, lambda: self._log_text_infer(output))
                    self.after(0, lambda: self._show_text_output(output, prompt))
                except FileNotFoundError:
                    self.after(0, lambda: self._log_text_infer("错误: 可执行文件未找到\n"))
                except subprocess.TimeoutExpired:
                    self.after(0, lambda: self._log_text_infer("错误: 生成超时\n"))
                    proc.kill()
                except Exception as e:
                    self.after(0, lambda: self._log_text_infer(f"错误: {e}\n"))
                finally:
                    self.after(0, lambda: self.text_infer_start_btn.config(state="normal"))
            threading.Thread(target=_run_interactive, daemon=True).start()
        else:
            # 单次推理模式
            def _run():
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    self.after(0, lambda: self._log_text_infer(result.stdout))
                    if result.stderr:
                        self.after(0, lambda: self._log_text_infer(result.stderr))
                    self.after(0, lambda: self._show_text_output(result.stdout, prompt))
                except FileNotFoundError:
                    self.after(0, lambda: self._log_text_infer("错误: 可执行文件未找到\n"))
                except subprocess.TimeoutExpired:
                    self.after(0, lambda: self._log_text_infer("错误: 生成超时\n"))
                except Exception as e:
                    self.after(0, lambda: self._log_text_infer(f"错误: {e}\n"))
                finally:
                    self.after(0, lambda: self.text_infer_start_btn.config(state="normal"))
            threading.Thread(target=_run, daemon=True).start()

    def _show_text_output(self, output: str, prompt: str):
        """解析 text_infer 输出并显示"""
        # text_infer 输出格式: prompt + generated_text
        lines = output.strip().split("\n")
        # 找到生成的文本行（通常是第一行非空输出）
        generated = ""
        for line in lines:
            if line.strip() and not line.startswith("[") and not line.startswith("加载"):
                generated = line
                break

        self.text_infer_output.config(state="normal")
        self.text_infer_output.delete("1.0", "end")
        self.text_infer_output.insert("1.0", generated if generated else output)
        self.text_infer_output.config(state="disabled")

    def _log_text_infer(self, text: str):
        def _do():
            self.text_infer_log.config(state="normal")
            self.text_infer_log.insert("end", text)
            self.text_infer_log.see("end")
            self.text_infer_log.config(state="disabled")
        self.after(0, _do)

    # ============================================================
    #  分词器训练 Tab
    # ============================================================
    def _build_tokenizer_train_tab(self):
        f = self.tokenizer_train_frame
        row = 0

        # 文本文件路径
        ttk.Label(f, text="训练文本文件:").grid(row=row, column=0, sticky="w")
        self.tokenizer_train_file_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.tokenizer_train_file_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_tokenizer_train_file).grid(row=row, column=2)
        row += 1

        # 输出路径
        ttk.Label(f, text="输出 JSON 路径:").grid(row=row, column=0, sticky="w")
        self.tokenizer_train_output_var = tk.StringVar(value="bpe_vocab.json")
        ttk.Entry(f, textvariable=self.tokenizer_train_output_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_tokenizer_train_output).grid(row=row, column=2)
        row += 1

        # 参数
        param_frame = ttk.LabelFrame(f, text="训练参数", padding=6)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        ttk.Label(param_frame, text="分词器类型:").grid(row=0, column=0, sticky="w")
        self.tokenizer_train_type_var = tk.StringVar(value="bpe")
        ttk.Combobox(param_frame, textvariable=self.tokenizer_train_type_var, width=14, state="readonly",
                     values=["bpe", "charbpe"]).grid(row=0, column=1, sticky="w")
        ttk.Label(param_frame, text="(bpe=字节级, charbpe=字符级, 中文推荐)").grid(row=0, column=2, sticky="w", columnspan=2)

        ttk.Label(param_frame, text="词表大小:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.tokenizer_train_vocab_size_var = tk.IntVar(value=5000)
        ttk.Spinbox(param_frame, from_=256, to=100000, width=8,
                     textvariable=self.tokenizer_train_vocab_size_var).grid(row=1, column=1, padx=(0, 16), pady=(6, 0))

        ttk.Label(param_frame, text="最小合并频率:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.tokenizer_train_min_freq_var = tk.IntVar(value=2)
        ttk.Spinbox(param_frame, from_=1, to=100, width=6,
                     textvariable=self.tokenizer_train_min_freq_var).grid(row=1, column=3, pady=(6, 0))
        row += 1

        # 按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=6)
        row += 1

        self.tokenizer_train_start_btn = ttk.Button(btn_frame, text="▶  开始训练", command=self._start_tokenizer_train)
        self.tokenizer_train_start_btn.pack(side="left", padx=4)
        self.tokenizer_train_stop_btn = ttk.Button(btn_frame, text="⏹  停止", command=self._stop_tokenizer_train, state="disabled")
        self.tokenizer_train_stop_btn.pack(side="left", padx=4)
        row += 1

        # 日志输出
        self.tokenizer_train_log = tk.Text(f, height=14, width=72, state="disabled",
                                           font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                                           insertbackground="white")
        self.tokenizer_train_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.tokenizer_train_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.tokenizer_train_log["yscrollcommand"] = scroll.set

    def _browse_tokenizer_train_file(self):
        path = filedialog.askopenfilename(title="选择训练文本文件",
                                          filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            self.tokenizer_train_file_var.set(path)

    def _browse_tokenizer_train_output(self):
        path = filedialog.asksaveasfilename(title="输出 JSON 路径", defaultextension=".json",
                                            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.tokenizer_train_output_var.set(path)

    def _start_tokenizer_train(self):
        exe = str(TOKENIZER_TRAIN_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到 tokenizer_train 可执行文件:\n{exe}\n请先构建项目")
            return

        text_file = self.tokenizer_train_file_var.get().strip()
        if not text_file:
            messagebox.showwarning("提示", "请选择训练文本文件")
            return

        cmd = [exe, text_file,
               "--tokenizer", self.tokenizer_train_type_var.get(),
               "--vocab-size", str(self.tokenizer_train_vocab_size_var.get()),
               "--min-freq", str(self.tokenizer_train_min_freq_var.get()),
               "--output", self.tokenizer_train_output_var.get()]

        self._log_tokenizer_train(f"$ {' '.join(cmd)}\n")
        self.tokenizer_train_start_btn.config(state="disabled")
        self.tokenizer_train_stop_btn.config(state="normal")

        threading.Thread(target=self._run_process, args=(cmd, self._log_tokenizer_train, self._tokenizer_train_done),
                         daemon=True).start()

    def _stop_tokenizer_train(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log_tokenizer_train("\n[已终止]\n")
        self._tokenizer_train_done()

    def _tokenizer_train_done(self):
        self.tokenizer_train_start_btn.config(state="normal")
        self.tokenizer_train_stop_btn.config(state="disabled")

    def _log_tokenizer_train(self, text: str):
        self.tokenizer_train_log.config(state="normal")
        self.tokenizer_train_log.insert("end", text)
        self.tokenizer_train_log.see("end")
        self.tokenizer_train_log.config(state="disabled")

    # ============================================================
    #  分词器推理 Tab
    # ============================================================
    def _build_tokenizer_infer_tab(self):
        f = self.tokenizer_infer_frame
        row = 0

        # 模型路径
        ttk.Label(f, text="词表 JSON 路径:").grid(row=row, column=0, sticky="w")
        self.tokenizer_infer_model_var = tk.StringVar(value="bpe_vocab.json")
        ttk.Entry(f, textvariable=self.tokenizer_infer_model_var, width=48).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_tokenizer_infer_model).grid(row=row, column=2)
        row += 1

        # 操作模式
        param_frame = ttk.LabelFrame(f, text="操作", padding=6)
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        self.tokenizer_infer_mode_var = tk.StringVar(value="text")
        ttk.Radiobutton(param_frame, text="编码+解码验证", variable=self.tokenizer_infer_mode_var,
                        value="text").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(param_frame, text="仅编码", variable=self.tokenizer_infer_mode_var,
                        value="encode").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(param_frame, text="仅解码", variable=self.tokenizer_infer_mode_var,
                        value="decode").grid(row=0, column=2, sticky="w", padx=(8, 0))

        self.tokenizer_infer_show_tokens_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="显示 Token 详情",
                        variable=self.tokenizer_infer_show_tokens_var).grid(row=0, column=3, sticky="w", padx=(16, 0))

        # 编码文件
        self.tokenizer_infer_encode_file_var = tk.StringVar(value="")
        ttk.Label(param_frame, text="").grid(row=0, column=4)  # spacer
        row += 1

        # 编码文件路径
        encode_file_frame = ttk.Frame(f)
        encode_file_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(encode_file_frame, text="或编码文件:").pack(side="left")
        self.tokenizer_infer_file_var = tk.StringVar()
        ttk.Entry(encode_file_frame, textvariable=self.tokenizer_infer_file_var, width=38).pack(side="left", padx=4)
        ttk.Button(encode_file_frame, text="浏览…", command=self._browse_tokenizer_infer_file).pack(side="left")
        row += 1

        # 输入
        ttk.Label(f, text="输入文本/ID列表:").grid(row=row, column=0, sticky="w")
        self.tokenizer_infer_input_var = tk.StringVar()
        input_entry = ttk.Entry(f, textvariable=self.tokenizer_infer_input_var, width=60)
        input_entry.grid(row=row, column=1, columnspan=2, padx=4, sticky="ew")
        input_entry.bind("<Return>", lambda e: self._start_tokenizer_infer())
        row += 1

        # 按钮
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=4)
        row += 1

        self.tokenizer_infer_start_btn = ttk.Button(btn_frame, text="▶  执行", command=self._start_tokenizer_infer)
        self.tokenizer_infer_start_btn.pack(side="left", padx=4)
        row += 1

        # 输出区域
        output_frame = ttk.LabelFrame(f, text="输出结果", padding=6)
        output_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        self.tokenizer_infer_output = tk.Text(output_frame, height=12, width=72, state="disabled",
                                              font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4",
                                              wrap="word")
        self.tokenizer_infer_output.pack(fill="both", expand=True)

        # 日志
        self.tokenizer_infer_log = tk.Text(f, height=4, width=72, state="disabled",
                                           font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.tokenizer_infer_log.grid(row=row, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.tokenizer_infer_log.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.tokenizer_infer_log["yscrollcommand"] = scroll.set

    def _browse_tokenizer_infer_model(self):
        path = filedialog.askopenfilename(title="选择词表 JSON 文件",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.tokenizer_infer_model_var.set(path)

    def _start_tokenizer_infer(self):
        exe = str(TOKENIZER_INFER_EXE)
        if not Path(exe).exists():
            messagebox.showerror("错误", f"找不到 tokenizer_infer 可执行文件:\n{exe}\n请先构建项目")
            return

        input_text = self.tokenizer_infer_input_var.get().strip()
        if not input_text:
            messagebox.showwarning("提示", "请输入文本或 ID 列表")
            return

        mode = self.tokenizer_infer_mode_var.get()
        encode_file = self.tokenizer_infer_file_var.get().strip()
        cmd = [exe, "--vocab", self.tokenizer_infer_model_var.get()]
        if encode_file:
            # 如果指定了文件路径，使用 --encode-file
            cmd += ["--encode-file", encode_file]
        elif mode == "text" or mode == "encode":
            cmd += ["--encode", input_text]
        elif mode == "decode":
            cmd += ["--decode", input_text]

        if self.tokenizer_infer_show_tokens_var.get():
            cmd.append("--show-bytes")

        self._log_tokenizer_infer(f"$ {' '.join(cmd)}\n")
        self.tokenizer_infer_start_btn.config(state="disabled")

        # 清空输出
        self.tokenizer_infer_output.config(state="normal")
        self.tokenizer_infer_output.delete("1.0", "end")
        self.tokenizer_infer_output.config(state="disabled")

        def _run():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                output = result.stdout + result.stderr
                self.after(0, lambda: self._log_tokenizer_infer(result.stdout))
                if result.stderr:
                    self.after(0, lambda: self._log_tokenizer_infer(result.stderr))
                self.after(0, lambda o=output: self._show_tokenizer_output(o))
            except FileNotFoundError:
                self.after(0, lambda: self._log_tokenizer_infer("错误: 可执行文件未找到\n"))
            except subprocess.TimeoutExpired:
                self.after(0, lambda: self._log_tokenizer_infer("错误: 执行超时\n"))
            except Exception as e:
                self.after(0, lambda: self._log_tokenizer_infer(f"错误: {e}\n"))
            finally:
                self.after(0, lambda: self.tokenizer_infer_start_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_tokenizer_output(self, output: str):
        self.tokenizer_infer_output.config(state="normal")
        self.tokenizer_infer_output.delete("1.0", "end")
        self.tokenizer_infer_output.insert("1.0", output)
        self.tokenizer_infer_output.config(state="disabled")

    def _browse_tokenizer_infer_file(self):
        path = filedialog.askopenfilename(title="选择要编码的文本文件",
                                          filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            self.tokenizer_infer_file_var.set(path)

    def _log_tokenizer_infer(self, text: str):
        def _do():
            self.tokenizer_infer_log.config(state="normal")
            self.tokenizer_infer_log.insert("end", text)
            self.tokenizer_infer_log.see("end")
            self.tokenizer_infer_log.config(state="disabled")
        self.after(0, _do)

    # ============================================================
    #  图片查看 Tab
    # ============================================================
    def _build_viewer_tab(self):
        f = self.viewer_frame
        row = 0

        ttk.Label(f, text="CSV 文件:").grid(row=row, column=0, sticky="w")
        self.viewer_csv_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.viewer_csv_var, width=52).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_viewer_csv).grid(row=row, column=2)
        ttk.Button(f, text="显示", command=self._show_viewer_image).grid(row=row, column=3, padx=4)
        row += 1

        # 目录浏览
        ttk.Label(f, text="或选择目录:").grid(row=row, column=0, sticky="w")
        self.viewer_dir_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.viewer_dir_var, width=52).grid(row=row, column=1, padx=4)
        ttk.Button(f, text="浏览…", command=self._browse_viewer_dir).grid(row=row, column=2)
        ttk.Button(f, text="加载", command=self._load_viewer_dir).grid(row=row, column=3, padx=4)
        row += 1

        # 画布
        canvas_size = IMG_DIM * PIXEL_SIZE
        self.viewer_canvas = tk.Canvas(f, width=canvas_size, height=canvas_size, bg="black")
        self.viewer_canvas.grid(row=row, column=0, columnspan=4, pady=8)
        row += 1

        # 导航
        nav_frame = ttk.Frame(f)
        nav_frame.grid(row=row, column=0, columnspan=4)
        self.viewer_prev_btn = ttk.Button(nav_frame, text="◀ 上一张", command=self._viewer_prev, state="disabled")
        self.viewer_prev_btn.pack(side="left", padx=4)
        self.viewer_label = ttk.Label(nav_frame, text="— / —", font=("Segoe UI", 10))
        self.viewer_label.pack(side="left", padx=8)
        self.viewer_next_btn = ttk.Button(nav_frame, text="下一张 ▶", command=self._viewer_next, state="disabled")
        self.viewer_next_btn.pack(side="left", padx=4)

        self._viewer_files: list[Path] = []
        self._viewer_idx: int = 0

    def _browse_viewer_csv(self):
        path = filedialog.askopenfilename(title="选择 CSV", filetypes=[("CSV", "*.csv")])
        if path:
            self.viewer_csv_var.set(path)
            self._viewer_files = [Path(path)]
            self._viewer_idx = 0
            self._show_viewer_image()

    def _browse_viewer_dir(self):
        path = filedialog.askdirectory(title="选择图片目录")
        if path:
            self.viewer_dir_var.set(path)
            self._load_viewer_dir()

    def _load_viewer_dir(self):
        d = self.viewer_dir_var.get().strip()
        if not d or not Path(d).is_dir():
            return
        self._viewer_files = sorted(Path(d).glob("*.csv"))
        self._viewer_idx = 0
        self._update_viewer_nav()
        if self._viewer_files:
            self._show_viewer_image()

    def _show_viewer_image(self):
        if not self._viewer_files:
            # 尝试从单文件路径加载
            p = self.viewer_csv_var.get().strip()
            if p and Path(p).is_file():
                self._viewer_files = [Path(p)]
                self._viewer_idx = 0
            else:
                return
        path = self._viewer_files[self._viewer_idx]
        try:
            pixels = load_csv_pixels(str(path))
            draw_digit(self.viewer_canvas, pixels)
            self._update_viewer_nav()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _viewer_prev(self):
        if self._viewer_idx > 0:
            self._viewer_idx -= 1
            self._show_viewer_image()

    def _viewer_next(self):
        if self._viewer_idx < len(self._viewer_files) - 1:
            self._viewer_idx += 1
            self._show_viewer_image()

    def _update_viewer_nav(self):
        n = len(self._viewer_files)
        self.viewer_label.config(text=f"{self._viewer_idx + 1} / {n}" if n else "— / —")
        state_n = "normal" if self._viewer_idx < n - 1 else "disabled"
        state_p = "normal" if self._viewer_idx > 0 else "disabled"
        self.viewer_next_btn.config(state=state_n)
        self.viewer_prev_btn.config(state=state_p)

    # ============================================================
    #  通用
    # ============================================================
    def _run_process(self, cmd: list[str], log_fn, done_fn):
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self.after(0, lambda t=line: log_fn(t))
            self._process.wait()
        except FileNotFoundError:
            self.after(0, lambda: log_fn("错误: 可执行文件未找到\n"))
        except Exception as e:
            self.after(0, lambda: log_fn(f"错误: {e}\n"))
        finally:
            self._process = None
            self.after(0, done_fn)

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.destroy()


# ── 入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    app = NeuralNetGUI()
    app.mainloop()
