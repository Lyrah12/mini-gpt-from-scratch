"""测试标题解码、摘要拼接和可选报告导出流程。"""
# 导入本模块使用的标准库工具。
import sys

# 导入参数化测试和张量工具，检查解码与流水线行为。
import pytest
import torch

# 复用项目内的数据、模型或实验工具。
from minigpt.model import GPT, ModelConfig
from minigpt.tokenizer import BOS, EOS, PAD
from scripts.prepare_refinement import combine_article
from scripts.title_decoding import constrain, decode


# 检查无附加约束时的解码结果及调用前模式恢复。
def test_unconstrained_decoder_reproduces_reference_and_restores_mode():
    torch.manual_seed(11)
    model = GPT(ModelConfig(vocab_size=32, block_size=16, n_layer=1,
                           n_head=2, n_embd=16, dropout=0.1)).train()
    prompt = [BOS, 7, 8, 9]
    expected = model.generate(torch.tensor([prompt]), 5, 0, eos_id=EOS, forbidden_ids=(PAD, BOS))
    actual = decode(model, prompt, 'greedy', max_new_tokens=5)
    assert actual == expected[0, len(prompt):].tolist()
    assert model.training


# 检查三元组约束仅屏蔽会补成重复片段的 token。
def test_ngram_constraint_blocks_only_matching_generated_prefix():
    logits = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    actual = constrain(logits, [5, 6, 7, 5, 6], no_repeat_ngram=3)
    assert torch.isneginf(actual[0, 7])
    assert actual[0, 8] == logits[0, 8]
    assert torch.isneginf(actual[0, PAD]) and torch.isneginf(actual[0, BOS])
    assert torch.isfinite(actual[0, EOS])
    assert torch.isfinite(logits).all()


# 检查正负 logit 的重复惩罚方向均降低重复候选得分。
def test_repetition_penalty_handles_both_logit_signs():
    logits = torch.tensor([[0., 0., 0., 4., -4., 3.]])
    actual = constrain(logits, [3, 4, 3], repetition_penalty=2.)
    assert actual[0, 3].item() == 2.
    assert actual[0, 4].item() == -8.
    assert actual[0, 5].item() == 3.


# 检查摘要只拼接一次，空摘要时直接使用正文。
def test_source_lead_is_not_duplicated_and_no_headline_is_injected():
    assert combine_article('导语。', '导语。正文。') == '导语。正文。'
    assert combine_article('', '正文。') == '正文。'
    assert combine_article('<p>新闻导语。</p>', '正文。') == '新闻导语。 正文。'


# 分别检查有无本地报告时，补充实验都能保存并正常结束。
@pytest.mark.parametrize('report_present', [False, True])
def test_refinement_pipeline_completes_with_optional_report(tmp_path, monkeypatch, report_present):
    from minigpt.common import read_json, write_json
    from scripts import export_refinement, run_refinement

    monkeypatch.chdir(tmp_path)
    original = tmp_path / 'original'
    run = tmp_path / 'refinement'
    data = tmp_path / 'data'
    write_json(original / 'sft/run.json', {'config': {'stage': 'sft'}})
    write_json(original / 'eval_val_sft/metrics.json', {
        'selection_hash': 'same-documents', 'metrics': {'rougeL': 0.3}})
    write_json(original / 'eval_test_sft/metrics.json', {})
    write_json(original / 'eval_test_lead/metrics.json', {})
    write_json(run / 'sft/status.json', {'complete': True})
    write_json(run / 'sft/run.json', {})
    write_json(data / 'manifest.json', {})
    # 只在对应测试分支创建报告，另一分支模拟纯源码仓库。
    if report_present:
        report = tmp_path / 'report'
        report.mkdir()
        (report / 'main.tex').write_text('local report', encoding='utf-8')

    # 返回固定评测数据，使流程测试覆盖状态保存而无需实际训练。
    def evaluate_fixture(*args, **kwargs):
        return {
            'identity': {'selection_hash': 'same-documents', 'reference_hash': 'same-titles'},
            'metrics': {'rougeL': 0.3}, 'raw_repetition_flags': 0,
        }

    # 替换实际训练、评测和报告导出，只检查流水线是否正确完成。
    exported = []
    monkeypatch.setattr(run_refinement, 'prepare', lambda *args: None)
    monkeypatch.setattr(run_refinement, 'evaluate', evaluate_fixture)
    monkeypatch.setattr(export_refinement, 'export', exported.append)
    monkeypatch.setattr(sys, 'argv', ['run_refinement', '--runs', str(run),
                                     '--base', str(data), '--data', str(data),
                                     '--original', str(original)])
    # 运行缓存完毕的测试流程，核对完成状态、汇总文件和运行锁。
    run_refinement.main()

    assert read_json(run / 'pipeline.json')['complete']
    assert (run / 'summary.json').is_file()
    assert not (run / 'pipeline.lock').exists()
    assert exported == ([run] if report_present else [])
