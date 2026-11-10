import copy

import pytest
import torch

from minigpt.model import GPT, ModelConfig
from minigpt.train import validation


@pytest.mark.parametrize("attention", ["manual", "sdpa"])
def test_future_tokens_do_not_change_prefix_logits(attention):
    torch.manual_seed(3)
    model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                            n_embd=32, dropout=0, attention=attention)).eval()
    x = torch.randint(0, 64, (2, 12))
    changed = x.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 64
    with torch.no_grad():
        a, b = model(x), model(changed)
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=0, atol=0)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_fused_attention_matches_manual_forward_and_gradients():
    torch.manual_seed(4)
    model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                            n_embd=32, dropout=0, attention="manual"))
    fused = copy.deepcopy(model)
    for block in fused.blocks:
        block.attn.cfg = copy.copy(block.attn.cfg)
        block.attn.cfg.attention = "sdpa"
    x = torch.randint(0, 64, (2, 9))
    y1, y2 = model(x), fused(x)
    torch.testing.assert_close(y1, y2, rtol=2e-4, atol=2e-6)
    y1.square().mean().backward()
    y2.square().mean().backward()
    for a, b in zip(model.parameters(), fused.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-6)
    assert model.head.weight is model.token_embedding.weight


def test_tiny_model_can_overfit_next_token_task():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = GPT(ModelConfig(vocab_size=16, block_size=12, n_layer=1, n_head=2, n_embd=32, dropout=0))
    x = torch.arange(12).unsqueeze(0)
    y = (x + 1) % 16
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 20


@pytest.mark.parametrize("training", [True, False])
def test_validation_restores_callers_mode(training):
    class FixedBatch:
        def sample(self, split, generator):
            return torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])

    model = GPT(ModelConfig(vocab_size=16, block_size=4, n_layer=1,
                            n_head=2, n_embd=16, dropout=0.1)).train(training)
    loss, count = validation(model, FixedBatch(), 1, 42, "cpu", "fp32")
    assert count == 3 and loss > 0
    assert model.training == training


def test_validation_restores_mode_after_sampling_failure():
    class BrokenBatch:
        def sample(self, split, generator):
            raise RuntimeError("sampling failed")

    model = GPT(ModelConfig(vocab_size=16, block_size=4, n_layer=1,
                            n_head=2, n_embd=16)).train()
    with pytest.raises(RuntimeError, match="sampling failed"):
        validation(model, BrokenBatch(), 1, 42, "cpu", "fp32")
    assert model.training
