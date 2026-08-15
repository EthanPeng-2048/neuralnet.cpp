"""Web GUI —— 基于 Gradio，复用 gui/cli.py（工具注册表）与 gui/parser.py（指标解析）。

页面：MNIST 训练 / MNIST 推理 / 文本训练 / 文本推理 / 分词器 / 基准与验证 / 构建
参数表单由 cli.py 的 Tool 定义程序化生成，日志实时流式输出，训练带实时曲线。
"""
from __future__ import annotations

import os
import re
import tempfile
import time

import gradio as gr
import pandas as pd
from PIL import Image

from .cli import TOOLS, build_command
from .parser import MetricParser, MNIST_TRAIN_METRICS, TEXT_TRAIN_METRICS
from .paths import (
    BUILD_DIR,
    PROJECT_ROOT,
    available_executables,
    exe_path,
    scan_datasets,
    scan_models,
)
from .stream import StreamRunner

_TOPK_ITEM = re.compile(r"(\d)\s*\(([\d.]+)%\)")
_SEPARATOR = re.compile(r"^-{5,}$")


# ══════════════════════════════════════════════════════════════════
# 组件 / 工具函数
# ══════════════════════════════════════════════════════════════════

def _norm_path(v):
    """FileExplorer 返回嵌套列表 [[路径]]，统一归一化为字符串路径。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        flat = []
        for item in v:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return str(flat[0]) if flat else ""
    return str(v)


def _as_list(v):
    """把组件值规整为字符串路径列表。"""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        out = []
        for item in v:
            if isinstance(item, (list, tuple)):
                out.extend(str(x) for x in item)
            else:
                out.append(str(item))
        return [x for x in out if x]
    return [str(v)]


def _dir_from_files(v):
    """目录上传返回文件列表，取公共父目录作为目录路径。"""
    ps = _as_list(v)
    if not ps:
        return ""
    if len(ps) == 1 and os.path.isdir(ps[0]):
        return ps[0]  # 直接给的是目录本身
    d = os.path.dirname(ps[0])
    return d or ps[0]


def _normalize_browse_values(tool, values):
    """browse 参数归一化：文件取单个路径，目录取公共父目录。"""
    dir_keys = {p.key for p in (tool.positional + tool.params) if p.browse == "dir"}
    file_keys = {p.key for p in (tool.positional + tool.params) if p.browse == "file"}
    out = {}
    for k, v in values.items():
        if k in dir_keys:
            out[k] = _dir_from_files(v)
        elif k in file_keys:
            out[k] = _norm_path(v)
        else:
            out[k] = v
    return out


def _is_small(p):
    """数值/下拉类小控件，可两两一行；路径/布尔/长文本占整行。"""
    return p.kind in ("int", "float", "choice", "engine") and not p.browse


def _file_types(default):
    """按默认值后缀推断 gr.File 的文件类型过滤。"""
    d = str(default or "")
    if "." in d:
        ext = d.rsplit(".", 1)[-1].lower()
        if ext and len(ext) <= 5 and ext.isalnum():
            return [ext]
    return None


def make_param(p):
    """按 Param 定义生成 Gradio 组件。
    选择文件 → gr.File（标准系统文件对话框）；目录/保存路径 → 单行输入框。
    """
    if p.kind == "int":
        return gr.Number(value=int(p.default or 0), label=p.label or p.key,
                         step=1, precision=0)
    if p.kind == "float":
        return gr.Number(value=float(p.default or 0.0), label=p.label or p.key)
    if p.kind == "bool":
        return gr.Checkbox(value=bool(p.default), label=p.label or p.key)
    if p.kind in ("choice", "engine"):
        return gr.Dropdown(choices=list(p.choices or []), value=p.default,
                           label=p.label or p.key)
    if p.kind == "text":
        return gr.Textbox(value=str(p.default or ""), label=p.label or p.key, lines=3)
    if p.browse == "file":
        return gr.UploadButton(label=p.label or p.key, type="filepath",
                               file_count="single", size="sm",
                               file_types=_file_types(p.default))
    if p.browse == "dir":
        return gr.UploadButton(label=p.label or p.key, type="filepath",
                               file_count="directory", size="sm")
    # save → 输入框（输出路径避免写入临时目录）
    return gr.Textbox(value=str(p.default or ""), label=p.label or p.key)


PAIR_SIZE = 4  # 每行放几个小参数


def pair_up(params):
    """可配对的小参数每 4 个一行，其余（路径/布尔/长文本）独占一行。"""
    units = []
    small = []
    for p in params:
        if _is_small(p):
            small.append(p)
            if len(small) == PAIR_SIZE:
                units.append(tuple(small))
                small = []
        else:
            if small:
                units.append(tuple(small))
                small = []
            units.append((p,))
    if small:
        units.append(tuple(small))
    return units


def render_params(tool):
    """按分组渲染全部参数（同类放一起），返回 key→组件 映射。"""
    comps: dict = {}
    all_params = list(tool.positional) + list(tool.params)
    groups, gmap = [], {}
    for p in all_params:
        if p.group not in gmap:
            gmap[p.group] = []
            groups.append(p.group)
        gmap[p.group].append(p)
    for g in groups:
        gr.Markdown(f"### {g}")
        with gr.Group():
            for unit in pair_up(gmap[g]):
                if len(unit) > 1:
                    with gr.Row():
                        for p in unit:
                            comps[p.key] = make_param(p)
                else:
                    comps[unit[0].key] = make_param(unit[0])
    return comps


def series_df(parser: MetricParser, names: list[str]) -> pd.DataFrame:
    """把指标系列转成 LinePlot 长表。"""
    rows = []
    for name in names:
        s = parser.series.get(name)
        if not s:
            continue
        for x, y in zip(s.x, s.y):
            rows.append({"x": x, "y": y, "series": name})
    return pd.DataFrame(rows, columns=["x", "y", "series"])


def _pack(log_lines: list[str], parser: MetricParser, chart_specs: list) -> tuple:
    out: list = ["\n".join(log_lines)]
    for _, names in chart_specs:
        out.append(series_df(parser, names))
    return tuple(out)


def _stream(tool_key: str, values: dict, metric_config: list,
            chart_specs: list, collect_lines: list | None = None):
    """运行工具并流式产出 (日志, *图表)。"""
    tool = TOOLS[tool_key]
    values = _normalize_browse_values(tool, values)
    cmd = build_command(tool, values)
    runner = StreamRunner()
    runner.start(cmd, str(PROJECT_ROOT))
    parser = MetricParser(metric_config)
    log_lines: list[str] = []
    heart = time.time()
    sent = False
    while True:
        events = runner.drain(0.05)
        for ev in events:
            kind = ev[0]
            if kind == "line":
                log_lines.append(ev[1])
                parser.feed(ev[1])
            elif kind == "progress":
                if log_lines:
                    log_lines[-1] = ev[1]
                else:
                    log_lines.append(ev[1])
                parser.feed(ev[1])
            else:  # done
                log_lines.append(f"—— 进程退出，退出码 {ev[1]} ——")
                if collect_lines is not None:
                    collect_lines[:] = list(log_lines)
                yield _pack(log_lines, parser, chart_specs)
                return
        now = time.time()
        if not sent or now - heart >= 1.0:
            sent = True
            heart = now
            yield _pack(log_lines, parser, chart_specs)
        time.sleep(0.05)


def _run_capture(tool_key: str, values: dict) -> list[str]:
    """运行工具并等待完成，返回全部输出行。"""
    lines: list[str] = []
    for _ in _stream(tool_key, values, [], [], collect_lines=lines):
        pass
    return lines


# ══════════════════════════════════════════════════════════════════
# 页面构建
# ══════════════════════════════════════════════════════════════════

def build_tool_block(tool_key: str, metric_config: list | None = None,
                     chart_specs: list | None = None, open_acc: bool = True):
    """通用「工具运行器」：参数 + 输出日志 + 可选图表。"""
    tool = TOOLS[tool_key]
    metric_config = metric_config or []
    chart_specs = chart_specs or []
    with gr.Accordion(tool.title, open=open_acc):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                comps = render_params(tool)
                btn = gr.Button("▶ 运行", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="输出", lines=16, interactive=False)
                charts = [
                    gr.LinePlot(label=t, x="x", y="y", color="series", height=220)
                    for t, _ in chart_specs
                ]

    def gen(*vals):
        values = dict(zip(comps.keys(), vals))
        yield from _stream(tool_key, values, metric_config, chart_specs)

    btn.click(gen, inputs=list(comps.values()), outputs=[log] + charts)
    return comps, log, charts


def build_train_tab(tool_key: str, metric_config: list, chart_specs: list):
    """训练页：未训练显示参数；点开始后 loss 图覆盖参数区；结束后自动切回参数。"""
    tool = TOOLS[tool_key]
    with gr.Row(visible=True) as param_panel:
        with gr.Column():
            comps = render_params(tool)
            start_btn = gr.Button("▶ 开始训练", variant="primary")
    with gr.Row(visible=False) as chart_panel:
        with gr.Column():
            log = gr.Textbox(label="输出", lines=16, interactive=False)
            charts = [
                gr.LinePlot(label=t, x="x", y="y", color="series", height=240)
                for t, _ in chart_specs
            ]

    def gen(*vals):
        values = _normalize_browse_values(tool, dict(zip(comps.keys(), vals)))
        empty = _pack([], MetricParser(metric_config), chart_specs)
        # 开始：参数隐藏、图表覆盖
        yield (gr.update(visible=False), gr.update(visible=True)) + empty
        last = empty
        for out in _stream(tool_key, values, metric_config, chart_specs):
            last = out
            yield (gr.update(visible=False), gr.update(visible=True)) + out
        # 结束：切回参数界面
        yield (gr.update(visible=True), gr.update(visible=False)) + last

    start_btn.click(gen, inputs=list(comps.values()),
                    outputs=[param_panel, chart_panel, log] + charts)
    return comps, log, charts


def build_mnist_train_tab():
    build_train_tab("mnist_train", MNIST_TRAIN_METRICS, [
        ("损失 Loss", ["loss"]),
        ("准确率 Accuracy", ["train_acc", "test_acc"]),
    ])


def build_text_train_tab():
    build_train_tab("text_train", TEXT_TRAIN_METRICS, [
        ("损失 Loss", ["loss", "test_loss"]),
        ("学习率 LR", ["lr"]),
    ])


def _mnist_infer_lines(path: str, model: str, topk: int, engine: str) -> list[str]:
    values = {"input": path, "model": model, "topk": int(topk),
              "show_pixels": False, "engine": engine}
    lines: list[str] = []
    for _ in _stream("mnist_infer", values, [], [], collect_lines=lines):
        pass
    return lines


def _topk_markdown(lines: list[str]) -> str:
    rows = []
    for ln in lines:
        m = re.search(r"->\s+(.+)$", ln)
        if not m:
            continue
        for d, c in _TOPK_ITEM.findall(m.group(1)):
            rows.append(f"| {d} | {c}% |")
    return ("## 预测结果\n\n| 数字 | 置信度 |\n|---|---|\n" + "\n".join(rows)
            if rows else "## 预测结果\n（无结果）")


def build_mnist_infer_tab():
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            sketch = gr.Sketchpad(
                value=Image.new("RGB", (280, 280), "black"),
                type="pil",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed",
                               default_size=14),
                label="手写数字（黑底白字，鼠标/触屏书写）",
            )
            model = gr.UploadButton(label="模型文件（.bin，可选）", type="filepath",
                                    file_count="single", size="sm",
                                    file_types=["bin"])
            engine = gr.Dropdown(choices=["cpu", "vulkan", "cuda"], value="cpu",
                                 label="计算引擎")
            topk = gr.Number(value=3, label="Top-K", step=1, precision=0)
            recognize_btn = gr.Button("▶ 识别手写数字", variant="primary")
            gr.Markdown("---\n或从 CSV 图片推理：")
            input_file = gr.UploadButton(label="图片 CSV 文件", type="filepath",
                                         file_count="single", size="sm",
                                         file_types=["csv"])
            csv_btn = gr.Button("推理指定图片")
        with gr.Column(scale=2):
            result = gr.Markdown("## 预测结果\n（在左侧写字板上书写数字后点「识别」）")
            log = gr.Textbox(label="输出", lines=10, interactive=False)

    def _to_gray(img):
        """把 Sketchpad/ImageEditor 值转成 28x28 灰度图（兼容 PIL 对象/路径/图层）。"""
        comp = None
        if isinstance(img, dict):
            comp = (img.get("composite") or (img.get("layers") or [None])[0]
                    or img.get("background"))
        elif isinstance(img, Image.Image):
            comp = img
        else:
            comp = img
        if comp is None:
            return None
        try:
            if isinstance(comp, Image.Image):
                gray = comp.convert("L")
            else:
                gray = Image.open(comp).convert("L")
            return gray.resize((28, 28), Image.Resampling.LANCZOS)
        except Exception:
            return None

    def recognize(img, model_path, eng, k):
        model_path = _norm_path(model_path) or "pretrained/MNIST_MLP.bin"
        gray = _to_gray(img)
        if gray is None:
            return "（请先在写字板上书写数字）", ""
        vals = list(gray.getdata())
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                          mode="w", newline="", encoding="utf-8")
        try:
            tmp.write(",".join(str(int(v)) for v in vals))
            tmp.close()
            lines = _mnist_infer_lines(tmp.name, model_path, int(k), eng)
            return "\n".join(lines), _topk_markdown(lines)
        finally:
            try:
                import os
                os.unlink(tmp.name)
            except OSError:
                pass

    def infer_csv(path, model_path, eng, k):
        model_path = _norm_path(model_path) or "pretrained/MNIST_MLP.bin"
        if not (path or "").strip():
            return "（请选择图片 CSV 文件）", ""
        lines = _mnist_infer_lines(path, model_path, int(k), eng)
        return "\n".join(lines), _topk_markdown(lines)

    recognize_btn.click(recognize, inputs=[sketch, model, engine, topk],
                        outputs=[log, result])
    csv_btn.click(infer_csv, inputs=[input_file, model, engine, topk],
                  outputs=[log, result])


def build_text_infer_tab():
    tool = TOOLS["text_infer"]
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            prompt = gr.Textbox(label="提示词 (Prompt)", lines=3,
                                placeholder="例如：Once upon a time")
            comps = render_params(tool)
            btn = gr.Button("▶ 生成", variant="primary")
        with gr.Column(scale=2):
            result = gr.Textbox(label="生成结果", lines=12, interactive=False)
            log = gr.Textbox(label="日志", lines=8, interactive=False)

    def gen(p_text, *vals):
        if not (p_text or "").strip():
            yield "（请输入提示词）", ""
            return
        values = dict(zip(comps.keys(), vals))
        values = _normalize_browse_values(tool, values)
        cmd = build_command(tool, values)
        cmd += ["--prompt", p_text]
        runner = StreamRunner()
        runner.start(cmd, str(PROJECT_ROOT))
        lines: list[str] = []
        heart = time.time()
        sent = False
        while True:
            for ev in runner.drain(0.05):
                if ev[0] == "line":
                    lines.append(ev[1])
                elif ev[0] == "progress":
                    if lines:
                        lines[-1] = ev[1]
                    else:
                        lines.append(ev[1])
                else:
                    text = _extract_generation(lines, p_text)
                    yield "\n".join(lines), text
                    return
            if not sent or time.time() - heart >= 1.0:
                sent = True
                heart = time.time()
                yield "\n".join(lines), "…生成中…"
            time.sleep(0.05)

    btn.click(gen, inputs=[prompt] + list(comps.values()), outputs=[log, result])


def _extract_generation(lines: list[str], prompt: str) -> str:
    sep_idx = None
    for i, ln in enumerate(lines):
        if _SEPARATOR.match(ln.strip()):
            sep_idx = i
            break
    if sep_idx is None:
        return "\n".join(lines)
    text = "\n".join(lines[sep_idx + 1:]).strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
    return text or "（无输出）"


def build_tokenizer_tab():
    # 训练
    build_tool_block("tokenizer_train", open_acc=True)
    # 编解码工具
    with gr.Accordion("编解码工具", open=True):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                vocab = gr.UploadButton(label="词表 JSON 文件", type="filepath",
                                        file_count="single", size="sm",
                                        file_types=["json"])
                enc_in = gr.Textbox(label="编码输入（文本）", lines=3)
                enc_btn = gr.Button("编码 → token IDs", variant="primary")
                dec_in = gr.Textbox(label="解码输入（token IDs，逗号分隔）")
                dec_btn = gr.Button("← 解码为文本", variant="primary")
            with gr.Column(scale=2):
                enc_out = gr.Textbox(label="编码结果", lines=6, interactive=False)
                dec_out = gr.Textbox(label="解码结果", lines=6, interactive=False)

    def run(vocab_path, arg, flag):
        if not (arg or "").strip():
            return "（请输入内容）"
        vp = _norm_path(vocab_path) or "bpe_vocab.json"
        cmd = [str(exe_path("tokenizer_infer")), "--vocab", vp, flag, arg]
        runner = StreamRunner()
        runner.start(cmd, str(PROJECT_ROOT))
        lines: list[str] = []
        while True:
            for ev in runner.drain(0.05):
                if ev[0] == "line":
                    lines.append(ev[1])
                elif ev[0] == "progress":
                    if lines:
                        lines[-1] = ev[1]
                    else:
                        lines.append(ev[1])
                else:
                    return "\n".join(lines) or "（无输出）"
            time.sleep(0.05)

    enc_btn.click(run, inputs=[vocab, enc_in, gr.State("--encode")], outputs=enc_out)
    dec_btn.click(run, inputs=[vocab, dec_in, gr.State("--decode")], outputs=dec_out)


def build_bench_tab():
    for key in ["compute_bench", "bench_thresholds", "gpu_test",
                "attn_consistency_test", "swiglu_gradcheck", "rmsnorm_gradcheck"]:
        build_tool_block(key, open_acc=False)


def build_build_tab():
    with gr.Accordion("CMake 构建", open=True):
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                build_dir = gr.Textbox(value="build", label="构建目录")
                build_type = gr.Dropdown(choices=["Release", "Debug"],
                                         value="Release", label="构建类型")
                cuda_cb = gr.Checkbox(value=False, label="启用 CUDA（NN_ENABLE_CUDA）")
                tests_cb = gr.Checkbox(value=False, label="构建单元测试（NN_BUILD_TESTS）")
                note = gr.Markdown("Vulkan 由 CMake 自动检测（需 Vulkan SDK + glslc）。")
                btn = gr.Button("▶ 配置并构建", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="输出", lines=20, interactive=False)

    def gen(bdir, btype, cuda_v, tests_v):
        bdir = _norm_path(bdir) or "build"
        cfg = ["cmake", "-B", bdir, "-G", "Ninja", f"-DCMAKE_BUILD_TYPE={btype}"]
        if cuda_v:
            cfg.append("-DNN_ENABLE_CUDA=ON")
        if tests_v:
            cfg.append("-DNN_BUILD_TESTS=ON")
        lines: list[str] = []
        for cmd in (cfg, ["cmake", "--build", bdir, "--parallel"]):
            lines.append("$ " + " ".join(cmd))
            yield "\n".join(lines)
            runner = StreamRunner()
            runner.start(cmd, str(PROJECT_ROOT))
            while True:
                done = None
                for ev in runner.drain(0.05):
                    if ev[0] == "line":
                        lines.append(ev[1])
                    elif ev[0] == "progress":
                        if lines:
                            lines[-1] = ev[1]
                        else:
                            lines.append(ev[1])
                    else:
                        done = ev[1]
                if done is not None:
                    if done != 0:
                        lines.append("✘ 失败，停止")
                        yield "\n".join(lines)
                        return
                    break
                yield "\n".join(lines)
                time.sleep(0.05)
        yield "\n".join(lines)

    btn.click(gen, inputs=[build_dir, build_type, cuda_cb, tests_cb], outputs=log)


def build_home_tab():
    exes = available_executables()
    models = scan_models()
    datasets = scan_datasets()
    built = "✔ 已配置" if (BUILD_DIR / "build.ninja").exists() else "✘ 未配置"
    model_list = "\n".join(f"- `{path}`" for path in models)
    ds_list = "\n".join(f"- `{p}`" for p in datasets)
    gr.Markdown(
        f"""
# neuralnet.cpp 控制台

基于 Gradio 的 Web 界面 —— 训练 / 推理 / 分词器 / 基准 / 构建。

**构建状态**：{built} ｜ 可执行文件 **{len(exes)}** 个

**可用模型（{len(models)}）**
{model_list if model_list else "（无）"}

**可用数据集（{len(datasets)}）**
{ds_list if ds_list else "（无）"}
        """
    )


# ══════════════════════════════════════════════════════════════════
# 应用组装
# ══════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="neuralnet.cpp 控制台") as demo:
        with gr.Tabs():
            with gr.Tab("🏠 首页"):
                build_home_tab()
            with gr.Tab("🔢 MNIST 训练"):
                build_mnist_train_tab()
            with gr.Tab("✍️ MNIST 推理"):
                build_mnist_infer_tab()
            with gr.Tab("📝 文本训练"):
                build_text_train_tab()
            with gr.Tab("💬 文本推理"):
                build_text_infer_tab()
            with gr.Tab("🔠 分词器"):
                build_tokenizer_tab()
            with gr.Tab("⚡ 基准与验证"):
                build_bench_tab()
            with gr.Tab("🔨 构建"):
                build_build_tab()
    demo.queue()
    return demo
