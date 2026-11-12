"""测试因果注意力、梯度一致性、小任务拟合和验证模式恢复。"""
# 导入本模块使用的标准库工具。
import copy

# 使用 pytest 参数化测试，并用 PyTorch 比较输出和梯度。
import pytest
import torch

# 复用项目内的数据、模型或实验工具。
from minigpt.model import GPT, ModelConfig
from minigpt.train import validation


# 修改未来 token，检查两种注意力都保持前缀输出一致。
@pytest.mark.parametrize("attention", ["manual", "sdpa"])
def test_future_tokens_do_not_change_prefix_logits(attention):
    torch.manual_seed(3)
    model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                            n_embd=32, dropout=0, attention=attention)).eval()
    x = torch.randint(0, 64, (2, 12))
    # 只修改后半段输入，前半段输出应满足因果性。
    changed = x.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 64
    # 使用同一模型分别计算原序列和修改未来位置后的序列。
    with torch.no_grad():
        a, b = model(x), model(changed)
    # 允许受修改影响的位置变化，同时要求前缀完全一致。
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=0, atol=0)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


# 比较手写注意力与 SDPA 的前向结果、梯度和权重共享。
def test_fused_attention_matches_manual_forward_and_gradients():
    torch.manual_seed(4)
    model = GPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                            n_embd=32, dropout=0, attention="manual"))
    fused = copy.deepcopy(model)
    # 复制注意力配置后切换实现，保持两组模型参数一致。
    for block in fused.blocks:
        block.attn.cfg = copy.copy(block.attn.cfg)
        block.attn.cfg.attention = "sdpa"
    # 对两种注意力实现使用完全相同的输入和初始参数。
    x = torch.randint(0, 64, (2, 9))
    y1, y2 = model(x), fused(x)
    torch.testing.assert_close(y1, y2, rtol=2e-4, atol=2e-6)
    # 对同一形式的标量目标反向传播，再逐个比较参数梯度。
    y1.square().mean().backward()
    y2.square().mean().backward()
    for a, b in zip(model.parameters(), fused.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-6)
    assert model.head.weight is model.token_embedding.weight


# 训练极小模型拟合已知后继 token，检查损失明显下降。
def test_tiny_model_can_overfit_next_token_task():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = GPT(ModelConfig(vocab_size=16, block_size=12, n_layer=1, n_head=2, n_embd=32, dropout=0))
    x = torch.arange(12).unsqueeze(0)
    y = (x + 1) % 16
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    # 在固定短序列上多次更新，检查训练能够拟合简单对应关系。
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 拟合成功时，末次损失应足够低且显著小于初始损失。
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 20


# 分别在训练与评估状态调用验证，检查原模式得到恢复。
@pytest.mark.parametrize("training", [True, False])
def test_validation_restores_callers_mode(training):
    # 提供固定输入和标签，使模式恢复测试与数据采样无关。
    class FixedBatch:
        # 返回同一对短序列输入和下一 token 标签。
        def sample(self, split, generator):
            return torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])

    model = GPT(ModelConfig(vocab_size=16, block_size=4, n_layer=1,
                            n_head=2, n_embd=16, dropout=0.1)).train(training)
    loss, count = validation(model, FixedBatch(), 1, 42, "cpu", "fp32")
    assert count == 3 and loss > 0
    assert model.training == training


# 故意使采样报错，检查异常退出时仍恢复模型模式。
def test_validation_restores_mode_after_sampling_failure():
    # 模拟批次采样阶段抛出异常的数据源。
    class BrokenBatch:
        # 触发预设异常，覆盖验证流程的清理分支。
        def sample(self, split, generator):
            raise RuntimeError("sampling failed")

    model = GPT(ModelConfig(vocab_size=16, block_size=4, n_layer=1,
                            n_head=2, n_embd=16)).train()
    with pytest.raises(RuntimeError, match="sampling failed"):
        validation(model, BrokenBatch(), 1, 42, "cpu", "fp32")
    assert model.training
