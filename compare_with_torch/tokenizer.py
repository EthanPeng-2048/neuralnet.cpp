# ── tokenizer.py — 分词器 (基于 tokenizers 库) ─────────────────────────────
#
# 直接使用 HuggingFace tokenizers 库，不再手写 BPE 算法。
# 仅提供统一适配器，包装 tokenizers.Tokenizer 暴露与 C++ 项目相近的接口。
#
# 接口：
#   encode(text) -> list[int]
#   decode(ids)  -> str
#   vocab_size() -> int
#   bos_id()     -> int
#   eos_id()     -> int
#
# 加载/训练：
#   load_tokenizer_from_file(path)  — 加载 tokenizers 库标准 JSON
#   train_bpe(text_path, vocab_size, output_path) — 训练并保存
#   get_or_train_tokenizer(...)     — 优先加载，不存在则训练
# ─────────────────────────────────────────────────────────────────────────

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# 特殊 token（与 C++ 项目保持一致的命名）
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID = 0
UNK_ID = 1
BOS_NAME = "<bos>"
EOS_NAME = "<eos>"


class TokenizerAdapter:
    """包装 tokenizers.Tokenizer，提供与 C++ 项目 Tokenizer 一致的接口。"""

    npos: int = -1

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def bos_id(self) -> int:
        tid = self._tok.token_to_id(BOS_NAME)
        return tid if tid is not None else self.npos

    def eos_id(self) -> int:
        tid = self._tok.token_to_id(EOS_NAME)
        return tid if tid is not None else self.npos


def load_tokenizer_from_string(json_content: str) -> Optional[TokenizerAdapter]:
    """从 tokenizers 库标准 JSON 字符串加载。"""
    try:
        tok = Tokenizer.from_str(json_content)
    except Exception:
        return None
    return TokenizerAdapter(tok)


def load_tokenizer_from_file(path: str) -> Optional[TokenizerAdapter]:
    """从 tokenizers 库标准 JSON 文件加载。"""
    try:
        tok = Tokenizer.from_file(path)
    except Exception:
        return None
    return TokenizerAdapter(tok)


def train_bpe(
    text_path: str,
    vocab_size: int,
    output_path: Optional[str] = None,
) -> TokenizerAdapter:
    """训练一个 ByteLevel BPE 分词器。

    Args:
        text_path:    训练文本路径
        vocab_size:   目标词表大小
        output_path:  若提供，则保存为 tokenizers 库标准 JSON

    Returns:
        TokenizerAdapter 实例
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train(files=[text_path], trainer=trainer)

    if output_path:
        tokenizer.save(output_path)

    return TokenizerAdapter(tokenizer)


def get_or_train_tokenizer(
    vocab_path: str,
    text_path: str,
    vocab_size: int,
) -> TokenizerAdapter:
    """优先从 vocab_path 加载；失败则在 text_path 上现场训练并保存。"""
    tok = load_tokenizer_from_file(vocab_path)
    if tok is not None:
        print(f"已加载 tokenizer: {vocab_path} (vocab_size={tok.vocab_size()})")
        return tok

    print(f"无法加载 {vocab_path}，开始训练新 BPE (vocab_size={vocab_size})...")
    tok = train_bpe(text_path, vocab_size, vocab_path)
    print(f"已训练并保存 tokenizer 到 {vocab_path} (vocab_size={tok.vocab_size()})")
    return tok
