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

import json
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


class CharBPEAdapter:
    """加载 C++ 项目 CharBPE 格式 tokenizer (charbpe_fast.json)。

    JSON 格式:
      "type": "char_bpe_tokenizer"
      "vocab": { "id_str": "hex_bytes" | "special_name" }
        - id 0-3: 特殊 token (纯文本名: <pad>/<unk>/<bos>/<eos>)
        - id 4-259: 256 个单字节 token (hex 编码: "00"~"ff")
        - id 260+: 多字节字符 / 合并 token (hex 编码的 UTF-8 字节串)
      "merges": [[id_a, id_b, new_id], ...]

    编码算法 (与 C++ CharBPETokenizer::encode 一致):
      1. pre_tokenize: 按空白 + CJK 标点切分成 chunks
      2. 每 chunk: 字符 → base token id 序列
         - ASCII (byte < 0x80): id = 4 + byte
         - 非 ASCII: UTF-8 解码得到字符, 查 char_to_id_
      3. 按 merge priority 合并相邻 id 对
    """

    npos: int = -1
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    BYTE_BASE = 4

    def __init__(self, vocab: dict, merges: list):
        # vocab: {id_int: token_bytes}
        # merges: [[id_a, id_b, new_id], ...]
        self._vocab = vocab
        self._merges = merges
        self._vocab_size = len(vocab)

        # char_to_id: 多字节字符 token → id (BYTE_BASE 及以上)
        self._char_to_id = {}
        for tid, tok in vocab.items():
            if tid >= self.BYTE_BASE:
                self._char_to_id[tok] = tid

        # merge_map: (id_a, id_b) → (new_id, priority)
        self._merge_map = {}
        for priority, m in enumerate(merges):
            id_a, id_b, new_id = m[0], m[1], m[2]
            self._merge_map[(id_a, id_b)] = (new_id, priority)

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        chunks = self._pre_tokenize(data)
        all_ids: list[int] = []
        for chunk in chunks:
            ids = self._chars_to_ids(chunk)
            ids = self._apply_merges(ids)
            all_ids.extend(ids)
        return all_ids

    def decode(self, ids: list[int]) -> str:
        raw = b""
        for tid in ids:
            if tid >= self.BYTE_BASE and tid < self._vocab_size:
                # 防御：稀疏/缺失 ID 时跳过而非 KeyError
                tok = self._vocab.get(tid) if isinstance(self._vocab, dict) else None
                if tok is None:
                    continue
                raw += tok
        return raw.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return self._vocab_size

    def bos_id(self) -> int:
        return self.BOS_ID

    def eos_id(self) -> int:
        return self.EOS_ID

    # ── pre_tokenize: 按空白 + CJK 切分 (与 C++ 一致) ──────────

    @staticmethod
    def _is_space(b: int) -> bool:
        return b in (0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B)

    @staticmethod
    def _is_alnum_ascii(b: int) -> bool:
        return (0x30 <= b <= 0x39) or (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A)

    @staticmethod
    def _decode_utf8(data: bytes, pos: int) -> tuple[int, int]:
        """返回 (码点, 字节长度)。"""
        b0 = data[pos]
        if b0 < 0x80:
            return b0, 1
        if pos + 1 >= len(data):
            return 0xFFFD, 1
        if (b0 & 0xE0) == 0xC0:
            cp = ((b0 & 0x1F) << 6) | (data[pos + 1] & 0x3F)
            return cp, 2
        if pos + 2 >= len(data):
            return 0xFFFD, 1
        if (b0 & 0xF0) == 0xE0:
            cp = ((b0 & 0x0F) << 12) | ((data[pos + 1] & 0x3F) << 6) | (data[pos + 2] & 0x3F)
            return cp, 3
        if pos + 3 >= len(data):
            return 0xFFFD, 1
        if (b0 & 0xF8) == 0xF0:
            cp = ((b0 & 0x07) << 18) | ((data[pos + 1] & 0x3F) << 12) | ((data[pos + 2] & 0x3F) << 6) | (data[pos + 3] & 0x3F)
            return cp, 4
        return 0xFFFD, 1

    @staticmethod
    def _is_cjk(cp: int) -> bool:
        return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF)

    @staticmethod
    def _is_cjk_punct(cp: int) -> bool:
        return (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF)

    def _pre_tokenize(self, data: bytes) -> list[bytes]:
        chunks: list[bytes] = []
        i = 0
        n = len(data)
        while i < n:
            # 收集前导空白 (附加到下一个词)
            start = i
            while i < n and self._is_space(data[i]):
                i += 1
            if i >= n:
                break
            prefix = data[start:i]

            cp, clen = self._decode_utf8(data, i)
            if self._is_cjk(cp):
                word = prefix
                while i < n:
                    cp, clen = self._decode_utf8(data, i)
                    if not self._is_cjk(cp) or self._is_cjk_punct(cp):
                        break
                    word += data[i:i + clen]
                    i += clen
                if word:
                    chunks.append(word)
            elif self._is_alnum_ascii(data[i]):
                word = prefix
                while i < n:
                    if not self._is_alnum_ascii(data[i]):
                        break
                    word += bytes([data[i]])
                    i += 1
                if word:
                    chunks.append(word)
            else:
                word = prefix + data[i:i + clen]
                i += clen
                if word:
                    chunks.append(word)
        return chunks

    def _chars_to_ids(self, chunk: bytes) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(chunk)
        while i < n:
            b = chunk[i]
            if b < 0x80:
                ids.append(self.BYTE_BASE + b)
                i += 1
            else:
                _, clen = self._decode_utf8(chunk, i)
                ch = chunk[i:i + clen]
                ids.append(self._char_to_id.get(ch, self.UNK_ID))
                i += clen
        return ids

    def _apply_merges(self, ids: list[int]) -> list[int]:
        if len(ids) <= 1:
            return ids
        ids = list(ids)
        while True:
            best_prio = None
            best_pos = -1
            best_new = -1
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                hit = self._merge_map.get(pair)
                if hit is not None:
                    new_id, prio = hit
                    if best_prio is None or prio < best_prio:
                        best_prio = prio
                        best_pos = j
                        best_new = new_id
            if best_pos == -1:
                break
            ids[best_pos:best_pos + 2] = [best_new]
        return ids


def load_charbpe_from_file(path: str) -> Optional[CharBPEAdapter]:
    """加载 C++ CharBPE 格式的 JSON 文件 (charbpe_fast.json)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None
    return load_charbpe_from_string(content)


def load_charbpe_from_string(json_content: str) -> Optional[CharBPEAdapter]:
    """从 JSON 字符串加载 C++ CharBPE tokenizer。"""
    try:
        obj = json.loads(json_content)
    except json.JSONDecodeError:
        return None

    if obj.get("type") != "char_bpe_tokenizer":
        return None

    # 解析 vocab: { "id_str": "hex_or_name" } → { id_int: bytes }
    raw_vocab = obj.get("vocab", {})
    vocab: dict[int, bytes] = {}
    for id_str, val in raw_vocab.items():
        tid = int(id_str)
        if tid < 4:
            # 特殊 token: 纯文本名, 不参与编码
            vocab[tid] = val.encode("utf-8")
        else:
            # hex 编码的 UTF-8 字节串
            vocab[tid] = bytes.fromhex(val)

    # 解析 merges: [[id_a, id_b, new_id], ...]
    merges = obj.get("merges", [])

    return CharBPEAdapter(vocab, merges)


def get_or_train_tokenizer(
    vocab_path: str,
    text_path: str,
    vocab_size: int,
) -> "TokenizerAdapter | CharBPEAdapter":
    """优先从 vocab_path 加载；失败则在 text_path 上现场训练并保存。

    自动检测两种格式:
      1. C++ CharBPE 格式 (type=char_bpe_tokenizer) → CharBPEAdapter
      2. HuggingFace tokenizers 库标准格式 → TokenizerAdapter
    """
    # 先尝试 C++ CharBPE 格式
    tok = load_charbpe_from_file(vocab_path)
    if tok is not None:
        print(f"已加载 CharBPE tokenizer: {vocab_path} (vocab_size={tok.vocab_size()})")
        return tok

    # 再尝试 HuggingFace tokenizers 格式
    tok = load_tokenizer_from_file(vocab_path)
    if tok is not None:
        print(f"已加载 tokenizer: {vocab_path} (vocab_size={tok.vocab_size()})")
        return tok

    print(f"无法加载 {vocab_path}，开始训练新 BPE (vocab_size={vocab_size})...")
    tok = train_bpe(text_path, vocab_size, vocab_path)
    print(f"已训练并保存 tokenizer 到 {vocab_path} (vocab_size={tok.vocab_size()})")
    return tok
