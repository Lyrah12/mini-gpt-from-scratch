"""Byte-level BPE learned only from the training partition, with no UNK bytes."""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>"]
PAD, BOS, EOS = 0, 1, 2
PREFIX = "请根据以下新闻正文生成一个简洁、准确的中文标题。\n正文："
SUFFIX = "\n标题："
TITLE_TOKENS = 64  # Includes EOS; shared between preparation and evaluation.


def train_tokenizer(texts, path, vocab_size=16000):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    for i, token in enumerate(SPECIALS):
        assert tok.token_to_id(token) == i
    return tok


def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    if [tok.token_to_id(t) for t in SPECIALS] != [PAD, BOS, EOS]:
        raise ValueError("Incompatible special token IDs")
    return tok


def title_prompt(tok, body, block_size, title_tokens=TITLE_TOKENS):
    """统一训练与推理的新闻截断规则，为标题及 EOS 预留上下文。"""
    prefix, suffix = tok.encode(PREFIX).ids, tok.encode(SUFFIX).ids
    budget = block_size - title_tokens - len(prefix) - len(suffix) - 1
    if budget < 1:
        raise ValueError("block_size too small for the instruction and target reserve")
    return [BOS] + prefix + tok.encode(body).ids[:budget] + suffix


def sft_sequence(tok, row, block_size):
    """构造尚未移位的序列，仅标题与 EOS 作为有效监督目标。"""
    prompt = title_prompt(tok, row["content"], block_size)
    answer = tok.encode(row["title"]).ids[:TITLE_TOKENS - 1] + [EOS]
    ids = prompt + answer
    labels = [-100] * len(prompt) + answer
    # 多保留一个位置，供 Batches.sample 构造 ids[:-1] 与 labels[1:]。
    pad = block_size + 1 - len(ids)
    assert pad >= 0
    return ids + [PAD] * pad, labels + [-100] * pad
