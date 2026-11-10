"""One bounded SFT comparison; select all decoding settings on validation only."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from minigpt.common import read_json, sha256, write_json
from scripts.evaluate_refinement import evaluate
from scripts.prepare_refinement import prepare
from scripts.title_decoding import MODES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default='runs/refinement_20261104')
    ap.add_argument('--base', default='data/processed_xlsum')
    ap.add_argument('--data', default='data/processed_xlsum_lead')
    ap.add_argument('--original', default='runs/local_4060')
    a = ap.parse_args()
    root, original = Path(a.runs), Path(a.original)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'pipeline.lock'
    with lock.open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid()}, stream)
    state = {'pid': os.getpid(), 'complete': False}

    def update(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(root / 'pipeline.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        policy = {'seed': 42, 'validation_n': 200, 'test_n': 500, 'modes': MODES,
            'selection': 'per model/input: maximum validation ROUGE-L; ties: fewer raw repetition flags, then declared mode order',
            'input_cases': ['original_sft_text', 'original_sft_lead', 'refined_sft_lead'],
            'training': 'one 1500-step SFT from original pretrain/best.pt; unchanged original SFT config',
            'constraints': 'No pretrained external weights; original tokenizer and split IDs retained; no extra pretraining',
            'final_test': 'evaluate selected decoding once per case plus enriched lead32; test results do not change selection'}
        if (root / 'policy.json').exists() and read_json(root / 'policy.json') != policy:
            raise ValueError('Experiment policy changed')
        write_json(root / 'policy.json', policy)
        update('prepare_lead_data')
        prepare(a.base, 'data/raw/xlsum/chinese_simplified_XLSum_v2.0.tar.bz2', a.data)
        configs = original / 'sft/run.json'
        config = read_json(configs)['config']
        if (root / 'sft_config.json').exists() and read_json(root / 'sft_config.json') != config:
            raise ValueError('Saved SFT config changed')
        write_json(root / 'sft_config.json', config)
        cases = {'original_sft_text': (a.base, original / 'sft/best.pt'),
                 'original_sft_lead': (a.data, original / 'sft/best.pt'),
                 'refined_sft_lead': (a.data, root / 'sft/best.pt')}
        validation = {}
        for name in ('original_sft_text', 'original_sft_lead'):
            for mode in MODES:
                update('validation_' + name + '_' + mode)
                validation[name + '/' + mode] = evaluate(*[cases[name][0], root / f'val_{name}_{mode}', cases[name][1]], mode=mode)
        original_val = read_json(original / 'eval_val_sft/metrics.json')
        replay = validation['original_sft_text/greedy']
        if replay['identity']['selection_hash'] != original_val['selection_hash'] or any(abs(replay['metrics'][k] - original_val['metrics'][k]) > 1e-12 for k in replay['metrics']):
            raise ValueError('Greedy wrapper did not reproduce original validation result')
        update('sft_lead_1500')
        output = root / 'sft'
        if not (output / 'status.json').exists() or not read_json(output / 'status.json')['complete']:
            cmd = [sys.executable, '-B', '-X', 'utf8', '-u', '-m', 'minigpt.train', '--config', str(root / 'sft_config.json'),
                '--data', a.data, '--output', str(output), '--device', 'cuda', '--stop-file', str(root / 'pause.request')]
            cmd += ['--resume', str(output / 'last.pt')] if (output / 'last.pt').exists() else ['--init', str(original / 'pretrain/best.pt')]
            with (root / 'training.log').open('a', encoding='utf8') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not read_json(output / 'status.json')['complete']:
            update('paused')
            return
        for mode in MODES:
            update('validation_refined_' + mode)
            validation['refined_sft_lead/' + mode] = evaluate(a.data, root / f'val_refined_sft_lead_{mode}', output / 'best.pt', mode=mode)
        validation['lead32_lead'] = evaluate(a.data, root / 'val_lead32_lead')
        selection = {}
        for name in cases:
            mode = min(MODES, key=lambda mode: (-validation[name + '/' + mode]['metrics']['rougeL'],
                       validation[name + '/' + mode]['raw_repetition_flags'], list(MODES).index(mode)))
            selection[name] = {'mode': mode, 'validation': validation[name + '/' + mode]}
        decision = {'policy_sha256': sha256(root / 'policy.json'), 'selection': selection,
                    'frozen_before_test_utc': datetime.now(timezone.utc).isoformat()}
        if (root / 'selection.json').exists():
            saved = read_json(root / 'selection.json')
            if saved['selection'] != selection or saved['policy_sha256'] != decision['policy_sha256']:
                raise ValueError('Frozen selection changed')
            decision = saved
        else:
            write_json(root / 'selection.json', decision)
        tests = {}
        for name, (data, checkpoint) in cases.items():
            update('test_' + name)
            tests[name] = evaluate(data, root / f'test_{name}', checkpoint, split='test', limit=500, mode=selection[name]['mode'])
        tests['lead32_lead'] = evaluate(a.data, root / 'test_lead32_lead', split='test', limit=500)
        if len({v['identity']['selection_hash'] for v in tests.values()}) != 1 or len({v['identity']['reference_hash'] for v in tests.values()}) != 1:
            raise ValueError('Test comparison changed documents or reference titles')
        summary = {'policy': policy, 'selection': decision, 'validation': validation, 'test': tests,
            'data': read_json(Path(a.data) / 'manifest.json'), 'training_status': read_json(output / 'status.json'),
            'training_run': read_json(output / 'run.json'),
            'original_test_sft_greedy': read_json(original / 'eval_test_sft/metrics.json'),
            'original_test_lead32': read_json(original / 'eval_test_lead/metrics.json')}
        write_json(root / 'summary.json', summary)
        update('export_refinement')
        from scripts.export_refinement import export
        export(root)
        update('completed', complete=True)
    except Exception as error:
        update('failed', error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
