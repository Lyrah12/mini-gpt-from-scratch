import torch

from minigpt.model import GPT, ModelConfig
from minigpt.tokenizer import BOS, EOS, PAD
from scripts.prepare_refinement import combine_article
from scripts.title_decoding import constrain, decode


def test_unconstrained_decoder_reproduces_reference_and_restores_mode():
    torch.manual_seed(11)
    model = GPT(ModelConfig(vocab_size=32, block_size=16, n_layer=1,
                           n_head=2, n_embd=16, dropout=0.1)).train()
    prompt = [BOS, 7, 8, 9]
    expected = model.generate(torch.tensor([prompt]), 5, 0, eos_id=EOS, forbidden_ids=(PAD, BOS))
    actual = decode(model, prompt, 'greedy', max_new_tokens=5)
    assert actual == expected[0, len(prompt):].tolist()
    assert model.training


def test_ngram_constraint_blocks_only_matching_generated_prefix():
    logits = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    actual = constrain(logits, [5, 6, 7, 5, 6], no_repeat_ngram=3)
    assert torch.isneginf(actual[0, 7])
    assert actual[0, 8] == logits[0, 8]
    assert torch.isneginf(actual[0, PAD]) and torch.isneginf(actual[0, BOS])
    assert torch.isfinite(actual[0, EOS])
    assert torch.isfinite(logits).all()


def test_repetition_penalty_handles_both_logit_signs():
    logits = torch.tensor([[0., 0., 0., 4., -4., 3.]])
    actual = constrain(logits, [3, 4, 3], repetition_penalty=2.)
    assert actual[0, 3].item() == 2.
    assert actual[0, 4].item() == -8.
    assert actual[0, 5].item() == 3.


def test_source_lead_is_not_duplicated_and_no_headline_is_injected():
    assert combine_article('导语。', '导语。正文。') == '导语。正文。'
    assert combine_article('', '正文。') == '正文。'
    assert combine_article('<p>新闻导语。</p>', '正文。') == '新闻导语。 正文。'
