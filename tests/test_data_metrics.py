"""测试新闻清洗、分词、微调标签和字符级评价指标。"""
# 导入测试断言、数组和张量工具，检查数据构造与指标。
import pytest
import numpy as np
import torch

# 复用项目内的数据、模型或实验工具。
from minigpt.metrics import rouge
from minigpt.data import Batches
from minigpt.prepare import canonical, clean, fingerprint, split_name
from minigpt.tokenizer import EOS, TITLE_TOKENS, sft_sequence, title_prompt, train_tokenizer


# 检查规范文本相同，且相同文档组总是映射到同一分区。
def test_clean_and_partition_same_normalized_body():
    a = clean("<p>中文 新闻 ＡＢＣ１２３！</p>")
    b = clean("中文新闻 ABC123")
    assert canonical(a) == canonical(b)
    group = fingerprint(canonical(a)[:256])
    assert split_name(group, 42) == split_name(group, 42)


# 检查 BPE 编解码往返，以及只对标题和 EOS 计入损失。
def test_byte_bpe_round_trip_and_response_only_mask(tmp_path):
    tok = train_tokenizer(["中文新闻正文与标题。今天学校开展活动。"] * 30, tmp_path / "tok.json", 512)
    text = "未见汉字龘、emoji🧪与English 123\n"
    assert tok.decode(tok.encode(text).ids) == text
    # 构造超过窗口长度的新闻，检查截断、填充和标题标签。
    row = {"content": "学校今天举办科技展览。" * 60, "title": "学校举办科技展览"}
    prompt = title_prompt(tok, row["content"], 256)
    ids, labels = sft_sequence(tok, row, 256)
    assert len(ids) == len(labels) == 257
    assert labels[:len(prompt)] == [-100] * len(prompt)
    assert labels[len(prompt)] == tok.encode(row["title"]).ids[0]
    assert labels[len(prompt) + len(tok.encode(row["title"]).ids)] == EOS
    assert len(prompt) <= 256 - TITLE_TOKENS
    # 标签移位后，提示末尾应预测第一个标题 token。
    x, y = torch.tensor(ids[:-1]), torch.tensor(labels[1:])
    assert y[len(prompt) - 1] == ids[len(prompt)]
    # 对随机预测反向传播，检查提示位置的梯度被损失遮罩忽略。
    logits = torch.randn(1, 256, tok.get_vocab_size(), requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y, ignore_index=-100)
    loss.backward()
    assert logits.grad[0, :len(prompt) - 1].abs().sum() == 0
    assert logits.grad[0, len(prompt) - 1].abs().sum() > 0


# 用已知匹配关系检查字符级 ROUGE 的数值。
def test_character_rouge_known_cases():
    assert rouge("学校，开学！", "学校开学") == {"rouge1": 1., "rouge2": 1., "rougeL": 1.}
    assert rouge("", "学校") == {"rouge1": 0., "rouge2": 0., "rougeL": 0.}
    score = rouge("甲乙丙", "甲乙丁")
    assert score["rouge1"] == pytest.approx(2 / 3)
    assert score["rouge2"] == pytest.approx(0.5)
    assert score["rougeL"] == pytest.approx(2 / 3)


# 检查批次标签移位，并验证修改批次不会改写底层数组。
def test_sft_batch_shift_and_arrays_are_independent(tmp_path):
    ids = np.array([[1, 4, 5, 2, 0]], dtype=np.int64)
    labels = np.array([[-100, -100, 5, 2, -100]], dtype=np.int64)
    # 创建最小微调数组，以同一格式分别模拟训练和验证分区。
    for split in ("train", "val"):
        np.save(tmp_path / f"sft_{split}_ids.npy", ids)
        np.save(tmp_path / f"sft_{split}_labels.npy", labels)
    batches = Batches(tmp_path, "sft", block_size=4, batch_size=1)
    x, y = batches.sample("train", torch.Generator().manual_seed(42))
    assert x.dtype == y.dtype == torch.int64
    assert x.tolist() == [[1, 4, 5, 2]]
    assert y.tolist() == [[-100, 5, 2, -100]]
    # 改写抽出的批次，再确认磁盘映射中的原数组仍与输入一致。
    x[0, 0], y[0, 0] = 9, 9
    np.testing.assert_array_equal(batches.arrays["train"][0], ids)
    np.testing.assert_array_equal(batches.arrays["train"][1], labels)
