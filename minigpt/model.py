"""Pre-LN GPT: embeddings + causal attention + MLP, with tied output weights."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 10
    n_embd: int = 640
    dropout: float = 0.1
    attention: str = "sdpa"

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.attention not in ("sdpa", "manual"):
            raise ValueError("attention must be sdpa or manual")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")


class CausalAttention(nn.Module):
    """将各位置投影为 Q/K/V，再按因果约束汇总前缀信息。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        head_size = hidden_size // self.cfg.n_head
        # (B, T, d) -> (B, heads, T, head_size)，每个头独立计算位置关系。
        query, key, value = [
            tensor.view(batch_size, sequence_length, self.cfg.n_head, head_size).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        dropout = self.cfg.dropout if self.training else 0.0
        if self.cfg.attention == "sdpa":
            # SDPA 完成注意力运算，投影层与 Transformer 块由本项目实现。
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, is_causal=True
            )
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(head_size)
            # 第 i 行只读取第 0 到 i 列，防止下一 token 的信息泄漏。
            mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            probabilities = F.dropout(scores.softmax(dim=-1), p=dropout, training=self.training)
            attended = probabilities @ value
        # 拼接各头后恢复 (B, T, d)，再作输出投影。
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        return self.resid_dropout(self.proj(attended))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # Pre-LN 在分支计算之前归一化，残差路径保留原始表示。
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 共享同一个 Parameter，使查表与词表输出共同更新这组权重。
        self.head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                # 缩小残差分支的初始输出，控制多层相加后的幅度。
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        _, t = ids.shape
        if t > self.config.block_size:
            raise ValueError("Sequence exceeds block_size")
        positions = torch.arange(t, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=64, temperature=0.0, top_k=40, eos_id=None, forbidden_ids=()):
        """逐 token 生成并返回完整序列，结束后恢复调用前的训练模式。"""
        if ids.size(0) != 1:
            raise ValueError("Reference decoding supports batch_size=1")
        if temperature < 0 or max_new_tokens < 0:
            raise ValueError("Invalid decoding settings")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # 每轮只使用最近的上下文，最后一个位置预测待追加的 token。
                logits = self(ids[:, -self.config.block_size:])[:, -1, :].float()
                if forbidden_ids:
                    logits[:, list(forbidden_ids)] = -float("inf")
                if temperature == 0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, -float("inf"))
                    nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
                ids = torch.cat((ids, nxt), dim=1)
                if eos_id is not None and nxt.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return ids

    def config_dict(self):
        return asdict(self.config)
