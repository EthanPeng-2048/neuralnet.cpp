# ── text_infer.py — GPT 文本生成推理 (PyTorch CUDA) ────────────────────────
#
# 复制自 C++ 项目 src/text_infer.cpp。
# 数据流：
#   加载模型 (含嵌入 tokenizer) → tokenizer.encode(prompt) → GPTModel.generate
#   → tokenizer.decode(generated) → 输出文本
#
# 与 C++ 一致：
#   - prompt 前自动加 BOS（使推理输入格式与训练时一致）
#   - min_new_tokens = max_new_tokens / 2，避免模型一上来就输出 EOS
# ─────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time

import torch

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import TokenizerAdapter, load_tokenizer_from_file, load_tokenizer_from_string
from model import GPTModel


def parse_args():
    parser = argparse.ArgumentParser(description="GPT 文本生成推理程序 (PyTorch CUDA)")
    parser.add_argument("--model", default="gpt_model_torch.pt", help="模型文件路径")
    parser.add_argument(
        "--vocab",
        default="gpt_bpe.json",
        help="词表 JSON 路径（仅当模型未嵌入 tokenizer 时使用）",
    )
    parser.add_argument("--prompt", default="Hello", help="输入提示文本")
    parser.add_argument("--interactive", action="store_true", help="交互式生成模式")
    parser.add_argument("--max-tokens", type=int, default=200, help="最大生成 token 数")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="温度参数 (0=贪心, 1.0=贪心)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu / cuda:0（默认 cuda，不可用自动回退 cpu）",
    )
    parser.add_argument("--show-tokens", action="store_true", help="显示 token ID (调试用)")
    return parser.parse_args()


def load_model_and_tokenizer(cfg):
    """加载模型 + tokenizer（优先从模型嵌入，回退到外部 JSON）。"""
    if not os.path.exists(cfg.model):
        print(f"模型文件不存在: {cfg.model}", file=sys.stderr)
        sys.exit(1)

    # 设备选择
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    ckpt = torch.load(cfg.model, map_location=device, weights_only=False)
    config = ckpt["config"]

    print(
        f"模型规格: vocab={config['vocab_size']}"
        f" d_model={config['d_model']}"
        f" heads={config['num_heads']}"
        f" layers={config['num_layers']}"
        f" d_ff={config['d_ff']}"
        f" seq_len={config['seq_len']}"
    )

    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        seq_len=config["seq_len"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"加载模型: {cfg.model}")

    # 加载 tokenizer
    tokenizer: TokenizerAdapter | None = None
    embedded_json = ckpt.get("tokenizer_json", "")
    if embedded_json:
        tokenizer = load_tokenizer_from_string(embedded_json)
        if tokenizer is None:
            print("解析嵌入 tokenizer 失败（非 tokenizers 库标准格式）", file=sys.stderr)
            sys.exit(1)
        print("已从模型文件加载嵌入 tokenizer")
    else:
        tokenizer = load_tokenizer_from_file(cfg.vocab)
        if tokenizer is None:
            print(
                f"加载词表失败（非 tokenizers 库标准格式）: {cfg.vocab}\n"
                "请使用 --vocab 指定词表路径，或使用嵌入 tokenizer 的模型",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"已从外部文件加载 tokenizer: {cfg.vocab}")

    print(f"词表大小: {tokenizer.vocab_size()}\n")
    return model, tokenizer, device


def generate_text(
    model: GPTModel,
    tokenizer: TokenizerAdapter,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int,
):
    """调用 model.generate（与 C++ generate_text 对应）。"""
    # min_new_tokens = max_new_tokens / 2，避免模型一上来就输出 EOS
    min_new = max_new_tokens // 2
    return model.generate(
        prompt=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
        min_new_tokens=min_new,
    )


def interactive_mode(model: GPTModel, tokenizer: TokenizerAdapter, cfg):
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    print("GPT 交互式生成 (输入 'quit' 退出)\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt in ("quit", "exit"):
            break
        if not prompt:
            prompt = " "

        prompt_tokens = tokenizer.encode(prompt)
        # 添加 BOS 前缀，使推理输入格式与训练时一致
        if bos_id != TokenizerAdapter.npos:
            prompt_tokens.insert(0, bos_id)

        print("生成中...")
        generated = generate_text(
            model, tokenizer, prompt_tokens, cfg.max_tokens, cfg.temperature, eos_id
        )
        print(f"\n{prompt}{tokenizer.decode(generated)}\n")

        if cfg.show_tokens:
            print(f"Tokens: {prompt_tokens} -> {generated}\n")


def main():
    cfg = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(cfg)

    # ── 交互模式 ─────────────────────────────────────────────
    if cfg.interactive:
        interactive_mode(model, tokenizer, cfg)
        return

    # ── 单次生成 ─────────────────────────────────────────────
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    prompt_tokens = tokenizer.encode(cfg.prompt)
    # 添加 BOS 前缀，使推理输入格式与训练时一致
    if bos_id != TokenizerAdapter.npos:
        prompt_tokens.insert(0, bos_id)

    print(f'提示: "{cfg.prompt}"')
    print(f"Prompt tokens: {len(prompt_tokens)} 个")
    print(f"生成 {cfg.max_tokens} 个 token (temperature={cfg.temperature})")
    print("----------------------------------------")

    t_start = time.time()
    generated = generate_text(
        model, tokenizer, prompt_tokens, cfg.max_tokens, cfg.temperature, eos_id
    )
    t_end = time.time()
    gen_sec = t_end - t_start

    print(f"{cfg.prompt}{tokenizer.decode(generated)}")

    if cfg.show_tokens:
        print(f"\n[Generated tokens: {generated}]")

    print("----------------------------------------")
    tokens_per_sec = len(generated) / gen_sec if gen_sec > 0 else 0
    print(
        f"生成 {len(generated)} tokens, 耗时 {gen_sec:.1f}s"
        f" ({tokens_per_sec:.0f} tokens/s)"
    )


if __name__ == "__main__":
    main()
