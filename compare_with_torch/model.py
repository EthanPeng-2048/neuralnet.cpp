# ── model.py — GPT 模型 (PyTorch CUDA) ──────────────────────────────────────
#
# 复制自 C++ 项目 include/neuralnet.cpp/compute_layer.hpp 中的 GPTModel。
# 架构完全一致：
#   GPTModel = TokenEmb + PosEmb + N × GPTBlock + LayerNorm + LM Head
#   GPTBlock (Pre-Norm):
#     x = x + CausalSelfAttn(LN1(x))
#     x = x + FFN(LN2(x))
#   CausalSelfAttention: 独立 Q/K/V/O 投影 (非合并 QKV), scale = 1/sqrt(d_k)
#   GeLU: QuickGeLU = x * sigmoid(1.702 * x)
#   FeedForward: Linear(d_model→d_ff) → GeLU → Linear(d_ff→d_model)
#
# 初始化（与 C++ 一致）：
#   Linear 权重: Xavier 均匀分布  limit = sqrt(6/(in+out))
#   Linear 偏置: 0
#   token_emb / pos_emb: N(0, 0.02)
#   LayerNorm: gamma=1, beta=0
# ─────────────────────────────────────────────────────────────────────────

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGeLU(nn.Module):
    """QuickGeLU: x * sigmoid(β * x), β = 1.702

    与 C++ GeLU 层完全一致（非标准 nn.GELU）。
    """

    BETA = 1.702

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.BETA * x)


class CausalSelfAttention(nn.Module):
    """因果多头自注意力（复制自 C++ CausalSelfAttention）。

    - 独立 Q/K/V/O 投影（不使用合并 QKV）
    - scale = 1 / sqrt(d_k), d_k = d_model / num_heads
    - 因果掩码: j <= i 允许, 否则 -inf
    """

    def __init__(self, d_model: int, num_heads: int, max_len: int = 1024):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # 四个独立投影（与 C++ 一致，非合并 QKV）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 因果掩码: 上三角（不含对角线）为 -inf
        mask = torch.triu(
            torch.full((max_len, max_len), float("-inf")), diagonal=1
        )
        self.register_buffer("mask", mask, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for layer in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        B, T, C = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # (batch, num_heads, seq, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # scores: (batch, num_heads, seq, seq) — 与 C++ batched_matmul(Q,K,H,true,false) 等价
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 施加因果掩码（C++ 在 scale 后 add mask，顺序一致）
        scores = scores + self.mask[:T, :T]

        # 行级 softmax（C++ Softmax 按 row 归约）
        attn = F.softmax(scores, dim=-1)

        # (batch, num_heads, seq, d_k)
        out = torch.matmul(attn, V)

        # 合并头: (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.w_o(out)


class FeedForward(nn.Module):
    """FFN(x) = Linear2(GeLU(Linear1(x)))（复制自 C++ FeedForward）。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = QuickGeLU()
        self._init_weights()

    def _init_weights(self):
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class GPTBlock(nn.Module):
    """Pre-Norm 解码器块（复制自 C++ GPTBlock）。

        x = x + CausalSelfAttn(LN1(x))
        x = x + FFN(LN2(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_len: int = 1024):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """Decoder-only Transformer 语言模型（复制自 C++ GPTModel）。

    组件: TokenEmb + PosEmb + N × GPTBlock + LayerNorm + LM Head
    输入: (batch, seq) LongTensor — token ID
    输出: (batch, seq, vocab_size) — 每个位置的 logits

    与 C++ 的差异仅在内存布局：
      C++:   (seq, batch) → (vocab, seq*batch)  [列优先]
      Torch: (batch, seq) → (batch, seq, vocab) [行优先]
    数学语义完全等价。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        seq_len: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers

        # 可学习嵌入（C++ 用 N(0, 0.02) 初始化）
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                GPTBlock(d_model, num_heads, d_ff, max_len=seq_len)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        # LM Head: Linear(d_model → vocab_size)，不与 token_emb 共享权重（与 C++ 一致）
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # 嵌入: N(0, 0.02)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        # LM Head: Xavier 均匀 + 零偏置
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播。

        Args:
            tokens:  (batch, seq) LongTensor
            targets: (batch, seq) LongTensor 或 None。若提供则计算交叉熵损失。

        Returns:
            (logits, loss)
            logits: (batch, seq, vocab_size)
            loss:   标量或 None
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # (1, T)

        # Token 嵌入 + 位置嵌入
        x = self.token_emb(tokens) + self.pos_emb(positions)  # (B, T, d_model)

        # Transformer 块
        for blk in self.blocks:
            x = blk(x)

        # 最终 LayerNorm + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 与 C++ CrossEntropyLoss 一致：
            #   对每个位置独立 softmax + cross entropy，求平均（不忽略 PAD）
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction="mean",
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        min_new_tokens: int = 0,
    ) -> list[int]:
        """自回归采样生成（复制自 C++ GPTModel::generate）。

        Args:
            prompt:          prompt token ID 列表
            max_new_tokens:  最大生成 token 数
            temperature:     0 = 贪心, >0 且 !=1.0 = 采样, 1.0 = 贪心（与 C++ 一致）
            eos_token_id:    遇到此 token 停止（min_new_tokens 之后才检查）
            min_new_tokens:  前这么多个 token 内不检查 EOS

        Returns:
            生成的 token ID 列表（不含 prompt，含触发停止的 EOS 之前的 token）
        """
        device = next(self.parameters()).device
        context = list(prompt)
        generated: list[int] = []

        for step in range(max_new_tokens):
            # 滑动窗口：超过 seq_len 时只取最后 seq_len 个
            start = 0
            if len(context) > self.seq_len:
                start = len(context) - self.seq_len

            cur_len = len(context) - start
            x = torch.tensor(
                [context[start:]], dtype=torch.long, device=device
            )  # (1, cur_len)

            logits, _ = self.forward(x)
            last_logits = logits[0, -1, :]  # (vocab,)

            # temperature（仅当 >0 且 !=1.0 时应用，与 C++ 一致）
            if temperature > 0.0 and temperature != 1.0:
                last_logits = last_logits / temperature

            # 数值稳定 softmax
            probs = F.softmax(last_logits, dim=-1)

            # 采样 or 贪心（与 C++ 分支一致）
            if temperature > 0.0 and temperature != 1.0:
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(probs).item()

            context.append(next_token)

            # EOS 检查（min_new_tokens 之后才允许停止）
            if step >= min_new_tokens and eos_token_id is not None and next_token == eos_token_id:
                break

            generated.append(next_token)

        return generated
