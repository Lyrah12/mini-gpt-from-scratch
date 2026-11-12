"""在训练集学习字节级 BPE，并统一标题任务的提示与标签格式。"""
# 导入本模块使用的标准库工具。
from pathlib import Path

# 使用 Tokenizers 构建字节级 BPE 及其训练与解码规则。
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# 固定特殊 token 编号与标题提示，训练和推理共用同一格式。
SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>"]
PAD, BOS, EOS = 0, 1, 2
PREFIX = "请根据以下新闻正文生成一个简洁、准确的中文标题。\n正文："
SUFFIX = "\n标题："
TITLE_TOKENS = 64  # 包含 EOS，数据准备和生成评测共用这一上限。


# 从训练文本学习字节级 BPE，并保存词表与合并规则。
def train_tokenizer(texts, path, vocab_size=16000):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    # 将全部字节放入初始字母表，并固定特殊 token 的排列顺序。
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    for i, token in enumerate(SPECIALS):
        assert tok.token_to_id(token) == i
    return tok


# 加载已保存的分词器，核对三个特殊 token 的固定编号。
def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    if [tok.token_to_id(t) for t in SPECIALS] != [PAD, BOS, EOS]:
        raise ValueError("Incompatible special token IDs")
    return tok


# 先为标题和 EOS 留出位置，再截取可见新闻前缀。
def title_prompt(tok, body, block_size, title_tokens=TITLE_TOKENS):
    """统一训练与推理的新闻截断规则，为标题及 EOS 预留上下文。"""
    prefix, suffix = tok.encode(PREFIX).ids, tok.encode(SUFFIX).ids
    # 总长度扣除提示、BOS 及标题预留位置，得到新闻 token 上限。
    budget = block_size - title_tokens - len(prefix) - len(suffix) - 1
    if budget < 1:
        raise ValueError("block_size too small for the instruction and target reserve")
    return [BOS] + prefix + tok.encode(body).ids[:budget] + suffix


# 拼接提示、标题与 EOS，只在标题目标上保留监督标签。
def sft_sequence(tok, row, block_size):
    """构造尚未移位的序列，仅标题与 EOS 作为有效监督目标。"""
    prompt = title_prompt(tok, row["content"], block_size)
    # 标题最多保留 63 个 token，再加入 EOS 作为最后一个监督目标。
    answer = tok.encode(row["title"]).ids[:TITLE_TOKENS - 1] + [EOS]
    ids = prompt + answer
    labels = [-100] * len(prompt) + answer
    # 多保留一个位置，供 Batches.sample 构造 ids[:-1] 与 labels[1:]。
    pad = block_size + 1 - len(ids)
    assert pad >= 0
    return ids + [PAD] * pad, labels + [-100] * pad
