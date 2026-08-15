"""CLI 工具注册表：GUI 表单 ↔ 命令行参数的唯一映射源。

每个 Tool 对应一个 build/ 下的可执行文件，参数定义与 CLI 帮助信息一一对应。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .paths import exe_path


@dataclass
class Param:
    key: str
    flag: str                 # 例如 "--epochs"；位置参数为 ""
    kind: str                 # int | float | str | bool | choice | text
    default: Any = None
    label: str = ""
    help: str = ""
    group: str = "通用"
    choices: list[str] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    bool_style: str = "flag"  # flag（仅出现）| true_false | on_off
    browse: str = ""          # "" | file | dir | save
    required: bool = False
    advanced: bool = False
    show_if: tuple[str, list[str]] | None = None  # (依赖参数, 允许值) 动态显隐


@dataclass
class Tool:
    key: str
    exe: str
    title: str
    description: str = ""
    positional: list[Param] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    group_notes: dict[str, str] = field(default_factory=dict)  # 分组标题下的说明
    expanded_groups: list[str] | None = None  # 默认展开的分组；None = 数据/训练


def coerce(kind: str, v: Any) -> Any:
    """把 QSettings 读回的值按参数类型强制转换。"""
    if kind == "bool":
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)
    if kind == "int":
        return int(float(v))
    if kind == "float":
        return float(v)
    return str(v)


def coerce_values(tool: Tool, saved: dict) -> dict:
    """把已保存的表单值按参数类型转换。"""
    by_key = {p.key: p for p in (tool.positional + tool.params)}
    out: dict = {}
    for k, v in saved.items():
        p = by_key.get(k)
        out[k] = coerce(p.kind, v) if p else v
    return out


def build_command(tool: Tool, values: dict[str, Any]) -> list[str]:
    """把表单值组装成完整命令行（含可执行文件绝对路径）。"""
    cmd: list[str] = [str(exe_path(tool.exe))]
    for p in tool.positional:
        v = values.get(p.key, "")
        if v:
            cmd.append(str(v))
    for p in tool.params:
        v = values.get(p.key, p.default)
        if p.kind == "bool":
            if p.bool_style == "flag":
                if v:
                    cmd.append(p.flag)
            elif p.bool_style == "true_false":
                cmd += [p.flag, "true" if v else "false"]
            else:  # on_off
                cmd += [p.flag, "on" if v else "off"]
        elif p.kind == "engine":
            # 计算引擎：cpu / vulkan / cuda，互斥，映射到 --gpu / --cuda
            if str(v) == "vulkan":
                cmd.append("--gpu")
            elif str(v) == "cuda":
                cmd.append("--cuda")
            # cpu → 不传任何 flag
        else:
            if v is None or v == "":
                continue  # 空字符串参数不传递（使用 CLI 默认）
            cmd += [p.flag, str(v)]
    return cmd


# ══════════════════════════════════════════════════════════════════
# 工具定义
# ══════════════════════════════════════════════════════════════════

MNIST_TRAIN = Tool(
    key="mnist_train",
    exe="mnist_train",
    title="MNIST 训练",
    description="手写数字训练，支持 MLP 与 Transformer（ViT）架构。",
    params=[
        # 数据
        Param("dataset", "--dataset", "str", "datasets/mnist_data", label="数据集目录",
              browse="dir", group="数据"),
        Param("max_samples", "--max-samples", "int", -1, label="最大样本数",
              help="-1 = 使用全部", group="数据", min=-1),
        Param("shuffle_steps", "--shuffle-steps", "bool", True, label="每轮打乱 batch 顺序",
              bool_style="true_false", group="数据"),
        # 模型架构（按架构动态显隐）
        Param("arch", "--arch", "choice", "mlp", label="架构",
              choices=["mlp", "transformer"], group="模型架构"),
        Param("resume", "--resume", "str", "", label="恢复训练", browse="file",
              help="从已有模型恢复（自动读取架构）", group="模型架构"),
        Param("save", "--save", "str", "mnist_model.bin", label="模型保存路径",
              browse="save", group="模型架构"),
        Param("layer_dims", "--layer-dims", "str", "784,512,256,128,64,10",
              label="MLP 层维度", help="逗号分隔", group="模型架构", advanced=True,
              show_if=("arch", ["mlp"])),
        Param("d_model", "--d-model", "int", 64, label="d_model", group="模型架构",
              min=1, show_if=("arch", ["transformer"])),
        Param("num_heads", "--num-heads", "int", 4, label="注意力头数", group="模型架构",
              min=1, show_if=("arch", ["transformer"])),
        Param("num_layers", "--num-layers", "int", 2, label="层数", group="模型架构",
              min=1, show_if=("arch", ["transformer"])),
        Param("d_ff", "--d-ff", "int", 128, label="FFN 中间维度", group="模型架构",
              min=1, show_if=("arch", ["transformer"])),
        Param("patch_size", "--patch-size", "int", 7, label="Patch 大小",
              help="28/7=4 → 16 patches", group="模型架构", min=1,
              show_if=("arch", ["transformer"])),
        Param("eval_samples", "--eval-samples", "int", 200, label="评估样本数",
              help="Transformer 评估上限，避免过慢", group="模型架构", min=1,
              advanced=True, show_if=("arch", ["transformer"])),
        # 训练（含学习率）
        Param("epochs", "--epochs", "int", 10, label="训练轮数", group="训练", min=1),
        Param("batch_size", "--batch-size", "int", 64, label="批大小", group="训练", min=1),
        Param("optimizer", "--optimizer", "choice", "adam",
              choices=["sgd", "sgd_momentum", "adam", "adamw", "muon"],
              label="优化器", group="训练"),
        Param("weight_decay", "--weight-decay", "float", 0.01, label="权重衰减 (AdamW)",
              group="训练", min=0.0, step=0.001),
        Param("lr_schedule", "--lr-schedule", "choice", "fixed",
              choices=["fixed", "cosine"], label="学习率调度", group="训练"),
        Param("lr", "--lr", "float", 0.001, label="学习率", group="训练",
              step=0.0005, min=0.0,
              help="cosine 模式为峰值（最高）学习率；fixed 为恒定学习率"),
        Param("warmup_epochs", "--warmup-epochs", "int", 0, label="预热轮数",
              help="线性预热到峰值，仅 cosine", group="训练", min=0,
              show_if=("lr_schedule", ["cosine"])),
        Param("min_lr", "--min-lr", "float", 1e-6, label="最低学习率",
              help="cosine 退火终点", group="训练", min=0.0, step=1e-6,
              show_if=("lr_schedule", ["cosine"])),
        Param("lr_per_epoch", "--lr-per-epoch", "str", "", label="手动逐轮学习率",
              help="逗号分隔，优先级最高（覆盖调度）", group="训练", advanced=True),
        # 后端
        Param("engine", "", "engine", "cpu", label="计算引擎", group="后端",
              choices=["cpu", "vulkan", "cuda"],
              help="cpu=CPU / vulkan=GPU 加速(Vulkan) / cuda=GPU 加速(CUDA)"),
    ],
    group_notes={
        "训练": "cosine：预热到峰值（学习率）后余弦退火到最低学习率；手动逐轮学习率优先级最高。",
    },
)

TEXT_TRAIN = Tool(
    key="text_train",
    exe="text_train",
    title="文本训练（GPT）",
    description="从文本文件训练 GPT 语言模型，支持多种架构与学习率调度。",
    positional=[
        Param("text_file", "", "str", "", label="训练文本文件", required=True,
              browse="file", group="数据", help="必需"),
    ],
    params=[
        # 数据
        Param("test_file", "--test-file", "str", "", label="测试集文件",
              browse="file", help="可选，每轮结束后评估 test loss", group="数据"),
        Param("vocab", "--vocab", "str", "gpt_bpe.json", label="词表 JSON",
              browse="file", group="数据"),
        # 模型架构
        Param("resume", "--resume", "str", "", label="恢复训练", browse="file", group="模型架构"),
        Param("save", "--save", "str", "gpt_model.bin", label="模型保存路径", browse="save", group="模型架构"),
        Param("d_model", "--d-model", "int", 128, label="d_model", group="模型架构", min=1),
        Param("num_heads", "--num-heads", "int", 4, label="注意力头数", group="模型架构", min=1),
        Param("num_layers", "--num-layers", "int", 4, label="层数", group="模型架构", min=1),
        Param("d_ff", "--d-ff", "int", 512, label="FFN 中间维度", group="模型架构", min=1),
        Param("positional_encoding", "--positional-encoding", "choice", "learned",
              choices=["learned", "sinusoidal", "alibi"], label="位置编码", group="模型架构"),
        Param("activation", "--activation", "choice", "gelu",
              choices=["gelu", "swiglu"], label="FFN 激活", group="模型架构"),
        Param("norm", "--norm", "choice", "layernorm",
              choices=["layernorm", "rmsnorm"], label="归一化层", group="模型架构"),
        # 训练（含学习率）
        Param("epochs", "--epochs", "int", 10, label="训练轮数", group="训练", min=1),
        Param("batch_size", "--batch-size", "int", 32, label="批大小", group="训练", min=1),
        Param("accum_steps", "--accum-steps", "int", 1, label="梯度累积步数",
              help="等效放大 batch_size×n", group="训练", min=1),
        Param("seq_len", "--seq-len", "int", 256, label="序列长度", group="训练", min=1),
        Param("stride", "--stride", "int", 0, label="滑动窗口步长",
              help="0 = 与序列长度相同（不重叠）", group="训练", min=0),
        Param("optimizer", "--optimizer", "choice", "adam",
              choices=["sgd", "sgd_momentum", "adam", "adamw", "muon"], label="优化器", group="训练"),
        Param("weight_decay", "--weight-decay", "float", 0.01, label="权重衰减 (AdamW)",
              group="训练", min=0.0, step=0.001),
        Param("max_norm", "--max-norm", "float", 0.0, label="梯度裁剪最大范数",
              help="0 = 不裁剪", group="训练", min=0.0),
        Param("lr_schedule", "--lr-schedule", "choice", "fixed",
              choices=["fixed", "cosine", "step_cosine"], label="学习率调度", group="训练"),
        Param("lr", "--lr", "float", 0.001, label="学习率", group="训练",
              min=0.0, step=0.0005,
              help="cosine/step_cosine 模式为峰值（最高）学习率；fixed 为恒定学习率"),
        Param("warmup_epochs", "--warmup-epochs", "int", 0, label="预热轮数",
              help="线性预热到峰值，仅 cosine", group="训练", min=0,
              show_if=("lr_schedule", ["cosine"])),
        Param("warmup_steps", "--warmup-steps", "int", 0, label="预热步数",
              help="仅 step_cosine", group="训练", min=0,
              show_if=("lr_schedule", ["step_cosine"])),
        Param("min_lr", "--min-lr", "float", 1e-6, label="最低学习率",
              help="cosine 退火终点", group="训练", min=0.0, step=1e-6,
              show_if=("lr_schedule", ["cosine", "step_cosine"])),
        Param("lr_per_epoch", "--lr-per-epoch", "str", "", label="手动逐轮学习率",
              help="逗号分隔，优先级最高（覆盖调度）", group="训练", advanced=True),
        Param("log_interval", "--log-interval", "int", 50, label="进度显示间隔 (step)",
              group="训练", min=1),
        Param("save_interval", "--save-interval", "int", 100, label="检查点保存间隔 (step)",
              group="训练", min=1),
        Param("grad_log", "--grad-log", "bool", False, label="显示梯度统计", group="训练"),
        # 后端
        Param("engine", "", "engine", "cpu", label="计算引擎", group="后端",
              choices=["cpu", "vulkan", "cuda"],
              help="cpu=CPU / vulkan=GPU 加速(Vulkan) / cuda=GPU 加速(CUDA)"),
        Param("tdr_retry", "--tdr-retry", "bool", True, label="GPU 超时自动减小 batch 重试",
              bool_style="on_off", group="后端", advanced=True),
        Param("max_tdr_retries", "--max-tdr-retries", "int", 4, label="最大重试次数",
              help="每次重试 batch 减半", group="后端", min=1, advanced=True),
        Param("flush_interval", "--flush-interval", "int", 0, label="Batch 录制粒度",
              help="每 N 个 block flush 一次，0=不间断", group="后端", min=0, advanced=True),
    ],
    group_notes={
        "训练": "cosine：预热到峰值（学习率）后余弦退火到最低学习率；step_cosine 按步退火；手动逐轮学习率优先级最高。",
    },
)

MNIST_INFER = Tool(
    key="mnist_infer",
    exe="mnist_infer",
    title="MNIST 推理",
    description="对单张图片（CSV 行）或目录批量推理手写数字。",
    positional=[
        Param("input", "", "str", "", label="图片文件或目录", required=True,
              browse="file", help="CSV 图片或包含 CSV 的目录", group="输入"),
    ],
    params=[
        Param("model", "--model", "str", "pretrained/MNIST_MLP.bin", label="模型文件",
              browse="file", group="模型"),
        Param("topk", "--topk", "int", 3, label="显示前 N 个预测", group="推理", min=1, max=10),
        Param("show_pixels", "--show-pixels", "bool", False, label="显示像素矩阵（调试）",
              group="推理"),
        Param("engine", "", "engine", "cpu", label="计算引擎", group="后端",
              choices=["cpu", "vulkan", "cuda"],
              help="cpu=CPU / vulkan=GPU 加速(Vulkan) / cuda=GPU 加速(CUDA)"),
    ],
    expanded_groups=["输入", "模型", "推理", "后端"],
)

TEXT_INFER = Tool(
    key="text_infer",
    exe="text_infer",
    title="文本推理（GPT 生成）",
    description="加载 GPT 模型，给定提示词生成续写文本。",
    params=[
        Param("model", "--model", "str", "gpt_model.bin", label="模型文件",
              browse="file", group="模型"),
        Param("vocab", "--vocab", "str", "bpe_vocab.json", label="词表 JSON",
              browse="file", group="模型", help="仅当模型未内嵌 tokenizer 时使用"),
        Param("max_tokens", "--max-tokens", "int", 200, label="最大生成 token 数",
              group="生成", min=1),
        Param("temperature", "--temperature", "float", 1.0, label="温度", group="生成",
              min=0.0, max=5.0, step=0.1, help="0 = 贪心"),
        Param("engine", "", "engine", "cpu", label="计算引擎", group="后端",
              choices=["cpu", "vulkan", "cuda"],
              help="cpu=CPU / vulkan=GPU 加速(Vulkan) / cuda=GPU 加速(CUDA)"),
    ],
    expanded_groups=["模型", "生成", "后端"],
)

TOKENIZER_TRAIN = Tool(
    key="tokenizer_train",
    exe="tokenizer_train",
    title="分词器训练",
    description="从文本训练 BPE / 字符级 BPE（支持中文）词表。",
    positional=[
        Param("text_file", "", "str", "", label="训练文本文件", required=True,
              browse="file", group="数据", help="必需"),
    ],
    params=[
        Param("tokenizer", "--tokenizer", "choice", "bpe",
              choices=["bpe", "charbpe"], label="分词器类型",
              help="bpe=字节级 / charbpe=字符级（支持中文）", group="分词器"),
        Param("output", "--output", "str", "bpe_vocab.json", label="词表输出路径",
              browse="save", group="分词器"),
        Param("vocab_size", "--vocab-size", "int", 5000, label="目标词表大小",
              group="分词器", min=1),
        Param("min_freq", "--min-freq", "int", 2, label="最小合并频率", group="分词器", min=1),
    ],
    expanded_groups=["数据", "分词器"],
)

TOKENIZER_INFER = Tool(
    key="tokenizer_infer",
    exe="tokenizer_infer",
    title="分词器工具",
    description="编码 / 解码 token。",
    params=[
        Param("vocab", "--vocab", "str", "bpe_vocab.json", label="词表 JSON",
              browse="file", group="词表"),
    ],
    expanded_groups=["词表"],
)

COMPUTE_BENCH = Tool(
    key="compute_bench",
    exe="compute_bench",
    title="计算性能基准",
    description="矩阵/算子性能基准测试（CPU，可选 GPU）。",
    params=[
        Param("size", "--size", "int", 512, label="矩阵维度", group="基准", min=1),
        Param("iters", "--iters", "int", 20, label="迭代次数", group="基准", min=1),
        Param("warmup", "--warmup", "int", 5, label="预热次数", group="基准", min=0),
    ],
    expanded_groups=["基准"],
)

BENCH_THRESHOLDS = Tool(
    key="bench_thresholds",
    exe="bench_thresholds",
    title="并行阈值基准",
    description="测试各算子并行阈值。",
)

GPU_TEST = Tool(
    key="gpu_test",
    exe="gpu_test",
    title="GPU 测试",
    description="GPU 后端（Vulkan）矩阵运算与性能测试。",
    params=[
        Param("size", "--size", "int", 256, label="矩阵维度", group="测试", min=1),
        Param("iters", "--iters", "int", 10, label="性能迭代次数", group="测试", min=1),
    ],
    expanded_groups=["测试"],
)

ATTN_TEST = Tool(key="attn_consistency_test", exe="attn_consistency_test",
                 title="Attention 一致性验证", description="forward 与 forward_step 一致性检查。")
SWIGLU_GRADCHECK = Tool(key="swiglu_gradcheck", exe="swiglu_gradcheck",
                        title="SwiGLU 梯度检查", description="数值梯度验证 split/merge 反向。")
RMSNORM_GRADCHECK = Tool(key="rmsnorm_gradcheck", exe="rmsnorm_gradcheck",
                         title="RMSNorm 梯度检查", description="数值梯度验证 gamma/输入梯度。")

TOOLS: dict[str, Tool] = {
    t.key: t
    for t in [
        MNIST_TRAIN, TEXT_TRAIN, MNIST_INFER, TEXT_INFER,
        TOKENIZER_TRAIN, TOKENIZER_INFER,
        COMPUTE_BENCH, BENCH_THRESHOLDS, GPU_TEST,
        ATTN_TEST, SWIGLU_GRADCHECK, RMSNORM_GRADCHECK,
    ]
}
