"""复查固定验证样本的生成结果和新闻输入对损失的影响。"""
# 导入本模块使用的标准库工具。
import argparse
import json
from pathlib import Path
import sys


# 加载固定验证预测，复查生成结果与输入敏感性。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    sys.path.insert(0, str(root))
    import torch
    from minigpt.evaluate import load_model
    from minigpt.tokenizer import load_tokenizer, title_prompt, sft_sequence, EOS, PAD, BOS
    from minigpt.common import sha256
    runs, data = root / 'runs/local_4060', root / 'data/processed_xlsum'
    torch.set_num_threads(4)
    rows = [json.loads(l) for l in (runs / 'eval_val_sft/predictions.jsonl').read_text(encoding='utf8').splitlines()]
    model, state = load_model(runs / 'sft/best.pt', data / 'tokenizer.json', 'cuda')
    tok = load_tokenizer(data / 'tokenizer.json')
    result = {'scope': 'diagnostics only; fixed validation subset; no tuning',
        'device': torch.cuda.get_device_name(), 'precision': str(next(model.parameters()).dtype),
        'checkpoint_step': state['step'], 'checkpoint_sha256': sha256(runs / 'sft/best.pt'),
        'replayed_validation': []}
    # 关闭梯度后重放已有验证新闻，并比较完整生成文本。
    with torch.inference_mode():
        # 按固定顺序重放前八条验证新闻，比较完整输出和 EOS 结束情况。
        for row in rows[:8]:
            prompt = title_prompt(tok, row['content'], model.config.block_size)
            x = torch.tensor([prompt], device='cuda')
            y = model.generate(x, 64, 0, eos_id=EOS, forbidden_ids=(PAD, BOS))
            pred = tok.decode(y[0, len(prompt):].tolist(), skip_special_tokens=True).strip()
            result['replayed_validation'].append({'id': row['id'], 'stored': row['prediction'],
                'replayed': pred, 'exact_match': pred == row['prediction'],
                'generated_tokens': y.size(1) - len(prompt), 'ended_with_eos': y[0, -1].item() == EOS})

        # 按有效标题 token 汇总给定新闻标题对的平均交叉熵。
        def nll(records):
            total, tokens = 0., 0
            # 每批构造最多四条新闻，只累计有效标题和 EOS 的交叉熵。
            for begin in range(0, len(records), 4):
                batch = [sft_sequence(tok, r, 512) for r in records[begin:begin+4]]
                x = torch.tensor([a[:-1] for a, b in batch], device='cuda')
                y = torch.tensor([b[1:] for a, b in batch], device='cuda')
                logits = model(x)
                total += torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
                    y.flatten(), ignore_index=-100, reduction='sum').item()
                tokens += (y != -100).sum().item()
            return {'loss': total / tokens, 'target_tokens': tokens, 'examples': len(records)}

        # 将新闻错位配对，再与原配对损失比较输入敏感性。
        val = [{'content': r['content'], 'title': r['reference']} for r in rows]
        shuffled = [{'content': val[(i+1) % len(val)]['content'], 'title': r['title']} for i, r in enumerate(val)]
        result['correct_source_nll'] = nll(val)
        result['shuffled_source_nll'] = nll(shuffled)
        result['shuffled_minus_correct_loss'] = result['shuffled_source_nll']['loss'] - result['correct_source_nll']['loss']
    result['warning'] = 'Shuffling changes visible text and potentially positions; a diagnostic of source sensitivity, not factuality or causal attribution.'
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


# 直接运行模块时，从命令行入口启动。
if __name__ == '__main__':
    main()
