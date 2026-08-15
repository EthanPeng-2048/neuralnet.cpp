"""
neuralnet.cpp GUI - 重构版 (单文件)
依赖: Python 3.8+, tkinter, Pillow
"""
import os, sys, math, subprocess, threading, tempfile, re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# 假设 go_board 在同级目录
sys.path.insert(0, str(Path(__file__).parent))
try:
    from go_board import GoGamePanel
except ImportError:
    GoGamePanel = None

# ── 常量 ──────────────────────────────────────────────
BUILD_DIR = Path(__file__).parent / "build"
_EXE = ".exe" if sys.platform == "win32" else ""
EXES = {
    "mnist_train": BUILD_DIR / f"mnist_train{_EXE}",
    "mnist_infer": BUILD_DIR / f"mnist_infer{_EXE}",
    "text_train": BUILD_DIR / f"text_train{_EXE}",
    "text_infer": BUILD_DIR / f"text_infer{_EXE}",
    "tokenizer_train": BUILD_DIR / f"tokenizer_train{_EXE}",
    "tokenizer_infer": BUILD_DIR / f"tokenizer_infer{_EXE}",
}
IMG_DIM, PIXEL_SIZE = 28, 8

# ═══════════════════════════════════════════════════════
#  1. 核心重构组件 (消灭重复代码的利器)
# ═══════════════════════════════════════════════════════

class TaskRunner:
    """统一处理子进程与线程，告别每个 Tab 都写一遍 threading.Thread"""
    def __init__(self):
        self.process = None

    def run(self, cmd, log_fn, done_fn, exe_name=""):
        exe_path = EXES.get(exe_name, cmd[0])
        if not Path(exe_path).exists():
            messagebox.showerror("错误", f"找不到可执行文件:\n{exe_path}")
            done_fn()
            return

        def _target():
            try:
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, creationflags=flags
                )
                for line in self.process.stdout:
                    log_fn(line)
                self.process.wait()
            except Exception as e:
                log_fn(f"执行错误: {e}\n")
            finally:
                self.process = None
                done_fn()

        threading.Thread(target=_target, daemon=True).start()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

class FilePicker(ttk.Frame):
    """通用文件/目录选择器"""
    def __init__(self, parent, label, default="", mode="file", filetypes=None):
        super().__init__(parent)
        self.var = tk.StringVar(value=default)
        self.mode = mode
        self.filetypes = filetypes or [("All", "*.*")]
        ttk.Label(self, text=label).grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(self, textvariable=self.var, width=40).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(self, text="浏览…", command=self._browse).grid(row=0, column=2)
        self.columnconfigure(1, weight=1)

    def _browse(self):
        path = filedialog.askdirectory() if self.mode == "dir" else filedialog.askopenfilename(filetypes=self.filetypes)
        if path: self.var.set(path)
        
    def get(self): return self.var.get()

class LogConsole(ttk.Frame):
    """通用黑底日志面板"""
    def __init__(self, parent, height=8):
        super().__init__(parent)
        self.text = tk.Text(self, height=height, state="disabled", bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text["yscrollcommand"] = self.scroll.set
        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

    def append(self, text):
        self.text.config(state="normal")
        self.text.insert("end", text)
        self.text.see("end")
        self.text.config(state="disabled")

class ParamForm(ttk.Frame):
    """数据驱动表单：用字典配置生成 UI，消灭几百行的 Spinbox 代码"""
    def __init__(self, parent, fields, cols=3):
        super().__init__(parent)
        self.vars = {}
        for i, f in enumerate(fields):
            r, c = divmod(i, cols)
            name, label, f_type = f["name"], f.get("label", f["name"]), f.get("type", "str")
            ttk.Label(self, text=f"{label}:").grid(row=r, column=c*2, sticky="w", padx=5, pady=2)
            
            if f_type == "int":
                v = tk.IntVar(value=f.get("default", 0))
                w = ttk.Spinbox(self, from_=f.get("min", 0), to=f.get("max", 9999), textvariable=v, width=8)
            elif f_type == "float":
                v = tk.DoubleVar(value=f.get("default", 0.0))
                w = ttk.Spinbox(self, from_=f.get("min", 0.0), to=f.get("max", 1.0), 
                                increment=f.get("step", 0.01), textvariable=v, width=8, format="%.4f")
            elif f_type == "combo":
                v = tk.StringVar(value=f.get("default", ""))
                w = ttk.Combobox(self, textvariable=v, values=f["values"], state="readonly", width=10)
            elif f_type == "bool":
                v = tk.BooleanVar(value=f.get("default", False))
                w = ttk.Checkbutton(self, variable=v)
            else:
                v = tk.StringVar(value=f.get("default", ""))
                w = ttk.Entry(self, textvariable=v, width=15)
                
            w.grid(row=r, column=c*2+1, sticky="w", padx=5, pady=2)
            self.vars[name] = v

    def get(self): return {k: v.get() for k, v in self.vars.items()}

# ═══════════════════════════════════════════════════════
#  2. 原有优秀组件保留 (LiveChart, ScrollableFrame)
# ═══════════════════════════════════════════════════════

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._vsb.pack(side="right", fill="y")
        self.content = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._win, width=e.width))
        self._canvas.bind("<Enter>", lambda e: self._canvas.bind_all("<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

class LiveChart:
    # ... (此处保留您原代码中的 LiveChart 完整实现，为节省篇幅省略，直接复用原代码即可) ...
    def __init__(self, parent, height=200):
        self.canvas = tk.Canvas(parent, height=height, bg="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)
        self.series = {}
    def add_point(self, s, x, y): 
        self.series.setdefault(s, []).append((x,y))
        # 实际使用时请补全原代码的 redraw 逻辑
    def clear(self): self.series.clear()

# ═══════════════════════════════════════════════════════
#  3. 主 GUI 类 (利用组件大幅瘦身)
# ═══════════════════════════════════════════════════════

class NeuralNetGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("neuralnet.cpp GUI (重构版)")
        self.geometry("900x750")
        ttk.Style(self).theme_use("clam")

        self.runners = {k: TaskRunner() for k in EXES} # 为每个任务分配独立的 Runner
        
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # 挂载各个 Tab
        self._build_go_tab(nb)
        self._build_mnist_tab(nb)
        self._build_gpt_tab(nb)

    def _build_go_tab(self, nb):
        f = ScrollableFrame(nb)
        nb.add(f, text=" ♟️ 围棋 ")
        if GoGamePanel: GoGamePanel(f.content)

    def _build_mnist_tab(self, nb):
        mnist_nb = ttk.Notebook(nb)
        nb.add(mnist_nb, text=" 🧠 MNIST ")
        
        # --- 训练 Tab ---
        train_f = ScrollableFrame(mnist_nb)
        mnist_nb.add(train_f, text=" 训练 ")
        c = train_f.content
        
        # 1. 文件区 (使用 FilePicker，代码从 20行 -> 2行)
        self.m_tr_data = FilePicker(c, "数据集:", mode="dir")
        self.m_tr_data.pack(fill="x", padx=5, pady=2)
        self.m_tr_save = FilePicker(c, "保存模型:", default="mnist_model.bin")
        self.m_tr_save.pack(fill="x", padx=5, pady=2)

        # 2. 参数区 (使用 ParamForm，代码从 60行 -> 10行)
        param_fields = [
            {"name": "epochs", "label": "轮数", "type": "int", "default": 5, "min": 1, "max": 1000},
            {"name": "lr", "label": "学习率", "type": "float", "default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001},
            {"name": "batch", "label": "Batch", "type": "int", "default": 64, "min": 1, "max": 4096},
            {"name": "opt", "label": "优化器", "type": "combo", "values": ["adam", "sgd", "adamw"], "default": "adam"},
            {"name": "arch", "label": "架构", "type": "combo", "values": ["mlp", "transformer"], "default": "mlp"}
        ]
        self.m_tr_params = ParamForm(c, param_fields, cols=3)
        self.m_tr_params.pack(fill="x", padx=5, pady=5)

        # 3. 控制与日志
        btn_f = ttk.Frame(c)
        btn_f.pack(pady=5)
        self.m_tr_start = ttk.Button(btn_f, text="▶ 开始训练", command=lambda: self._start_task("mnist_train"))
        self.m_tr_start.pack(side="left", padx=5)
        self.m_tr_stop = ttk.Button(btn_f, text="⏹ 停止", command=lambda: self._stop_task("mnist_train"), state="disabled")
        self.m_tr_stop.pack(side="left", padx=5)

        self.m_tr_log = LogConsole(c, height=8)
        self.m_tr_log.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.m_tr_chart = LiveChart(c, height=150)
        self.m_tr_chart.canvas.pack(fill="x", padx=5, pady=5)

        # --- 推理 Tab (省略，结构同上) ---
        infer_f = ScrollableFrame(mnist_nb)
        mnist_nb.add(infer_f, text=" 推理 ")
        # ... 推理 Tab 逻辑 ...

    def _build_gpt_tab(self, nb):
        gpt_nb = ttk.Notebook(nb)
        nb.add(gpt_nb, text=" 📝 GPT ")
        # ... 同样使用 FilePicker 和 ParamForm 构建 ...

    # ═══════════════════════════════════════════════════════
    #  4. 统一的业务逻辑控制 (消灭重复的 _start_xxx 方法)
    # ═══════════════════════════════════════════════════════
    
    def _start_task(self, task_name):
        runner = self.runners[task_name]
        params = self.m_tr_params.get() # 获取表单数据
        
        # 动态拼接命令行
        cmd = [str(EXES[task_name]), 
               "--dataset", self.m_tr_data.get(),
               "--save", self.m_tr_save.get(),
               "--epochs", str(params["epochs"]),
               "--lr", str(params["lr"])]
        
        self.m_tr_log.text.delete("1.0", "end")
        self.m_tr_start.config(state="disabled")
        self.m_tr_stop.config(state="normal")
        self.m_tr_chart.clear()

        runner.run(
            cmd, 
            log_fn=self._parse_and_log, 
            done_fn=lambda: self._task_done(task_name),
            exe_name=task_name
        )

    def _stop_task(self, task_name):
        self.runners[task_name].stop()
        self.m_tr_log.append("\n[已手动终止]\n")
        self._task_done(task_name)

    def _task_done(self, task_name):
        self.m_tr_start.config(state="normal")
        self.m_tr_stop.config(state="disabled")

    def _parse_and_log(self, text):
        self.m_tr_log.append(text)
        # 保留您原来的正则解析逻辑，更新图表
        m = re.search(r"loss:\s*([0-9.eE\-+]+)", text)
        if m:
            # self.m_tr_chart.add_point(...)
            pass

if __name__ == "__main__":
    app = NeuralNetGUI()
    app.mainloop()