"""实现带因果注意力、Pre-LN 和输入输出权重共享的 GPT。"""
# 导入本模块使用的标准库工具。
import math
from dataclasses import asdict, dataclass

# 导入张量、神经网络层和注意力计算接口。
import torch
from torch import nn
from torch.nn import functional as F


# 集中保存模型维度、上下文长度和注意力实现设置。
@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 10
    n_embd: int = 640
    dropout: float = 0.1
    attention: str = "sdpa"

    # 检查维度、注意力头划分和 dropout 参数是否合法。
    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.attention not in ("sdpa", "manual"):
            raise ValueError("attention must be sdpa or manual")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")


# 实现带因果遮罩的多头自注意力。
class CausalAttention(nn.Module):
    """将各位置投影为 Q/K/V，再按因果约束汇总前缀信息。"""

    # 创建联合 QKV 投影、输出投影和分支 dropout。
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    # 拆分注意力头，汇总可见前缀后恢复原隐藏维度。
    def forward(self, x):
        # 输入形状为 B×T×d，联合投影后沿最后一维拆分 Q、K、V。
        batch_size, sequence_length, hidden_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        head_size = hidden_size // self.cfg.n_head
        # (B, T, d) -> (B, heads, T, head_size)，每个头独立计算位置关系。
        query, key, value = [
            tensor.view(batch_size, sequence_length, self.cfg.n_head, head_size).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        # SDPA 的 dropout 需显式传入，评估状态下设为零。
        dropout = self.cfg.dropout if self.training else 0.0
        # 按配置选择融合注意力或便于核对公式的手写计算。
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


# 组合 Pre-LN 注意力和前馈网络的两个残差分支。
class Block(nn.Module):
    # 建立两个归一化层和四倍隐藏维度的前馈网络。
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

    # 依次执行归一化、子层变换和残差相加。
    def forward(self, x):
        # Pre-LN 在分支计算之前归一化，残差路径保留原始表示。
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# 从 token ID 计算词表得分，并提供自回归生成接口。
class GPT(nn.Module):
    # 构建嵌入、Transformer 块和共享输出层，并初始化参数。
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        # 分别建立词表向量与位置向量，二者共享相同隐藏维度。
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

    # 为线性层与嵌入矩阵设置初始权重和偏置。
    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # 将词嵌入与位置嵌入相加，输出每个位置的词表得分。
    def forward(self, ids):
        _, t = ids.shape
        # 检查输入长度后，依次执行嵌入、各层变换和词表投影。
        if t > self.config.block_size:
            raise ValueError("Sequence exceeds block_size")
        positions = torch.arange(t, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    # 按独立参数统计规模，共享权重由 PyTorch 自动去重。
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    # 逐个选择新 token，遇到 EOS 或长度上限时结束。
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=64, temperature=0.0, top_k=40, eos_id=None, forbidden_ids=()):
        """逐 token 生成并返回完整序列，结束后恢复调用前的训练模式。"""
        if ids.size(0) != 1:
            raise ValueError("Reference decoding supports batch_size=1")
        if temperature < 0 or max_new_tokens < 0:
            raise ValueError("Invalid decoding settings")
        # 保存调用前模式，整个生成过程关闭 dropout 并在结束时恢复。
        was_training = self.training
        self.eval()
        # 通过 finally 恢复进入生成接口前的训练模式。
        try:
            # 以 max_new_tokens 限制新增 token 的数量。
            for _ in range(max_new_tokens):
                # 每轮只使用最近的上下文，最后一个位置预测待追加的 token。
                logits = self(ids[:, -self.config.block_size:])[:, -1, :].float()
                # 将禁止生成的特殊 token 得分置为负无穷。
                if forbidden_ids:
                    logits[:, list(forbidden_ids)] = -float("inf")
                # 零温度直接取最大得分，其余情况执行温度与 top-k 采样。
                if temperature == 0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, -float("inf"))
                    nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
                # 把新 token 追加到前缀，EOS 也计入生成长度。
                ids = torch.cat((ids, nxt), dim=1)
                if eos_id is not None and nxt.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return ids

    # 将模型配置转换为可写入检查点的普通字典。
    def config_dict(self):
        return asdict(self.config)
