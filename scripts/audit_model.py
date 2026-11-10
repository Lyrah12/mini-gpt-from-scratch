"""Read-only audit of frozen homework experiments; write a separate JSON artifact."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    sys.path.insert(0, str(root))
    from minigpt.common import sha256, source_fingerprint
    from minigpt.metrics import characters, rouge
    from minigpt.prepare import canonical, fingerprint
    from minigpt.tokenizer import (BOS, EOS, PAD, PREFIX, SUFFIX, TITLE_TOKENS,
                                  load_tokenizer, sft_sequence)
    data = root / 'data/processed_xlsum'
    runs = root / 'runs/local_4060'
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    tok = load_tokenizer(data / 'tokenizer.json')
    report = {'schema': 1, 'scope': 'read-only diagnostic; no training or tuning',
              'source_sha256': source_fingerprint(), 'checks': {}, 'data': {},
              'training': {}, 'evaluations': {}, 'validation_examples': {}}
    checks = report['checks']
    checks['artifact_sha256'] = {name: sha256(data / name) == digest
                                for name, digest in manifest['files'].items()}
    checks['tokenizer_sha256'] = sha256(data / 'tokenizer.json') == manifest['tokenizer_sha256']
    sets = {}
    budget = manifest['block_size'] - TITLE_TOKENS - len(tok.encode(PREFIX).ids) - len(tok.encode(SUFFIX).ids) - 1
    report['body_token_budget'] = budget
    val_by_id = {}

    def summary(values):
        return {k: float(v) for k, v in zip(('min', 'p50', 'p90', 'max'), np.percentile(values, [0, 50, 90, 100]))}

    for split in ('train', 'val', 'test'):
        rows = [json.loads(line) for line in (data / f'{split}.jsonl').read_text(encoding='utf8').splitlines()]
        if split == 'val':
            val_by_id = {r['id']: r for r in rows}
        ids_set, groups, bodies, body_lengths, title_lengths, visible_chars = set(), set(), set(), [], [], []
        truncated = bad_roundtrip = bad_special = mismatched = 0
        visible_recall, full_recall = [], []
        arrays = (np.load(data / f'sft_{split}_ids.npy', mmap_mode='r'),
                  np.load(data / f'sft_{split}_labels.npy', mmap_mode='r')) if split != 'test' else None
        stream_hash = hashlib.sha256()
        for i, row in enumerate(rows):
            body_ids = tok.encode(row['content']).ids
            title_ids = tok.encode(row['title']).ids
            visible = tok.decode(body_ids[:budget])
            body_lengths.append(len(body_ids))
            title_lengths.append(len(title_ids))
            visible_chars.append(len(visible))
            truncated += len(body_ids) > budget
            bad_roundtrip += tok.decode(body_ids) != row['content'] or tok.decode(title_ids) != row['title']
            bad_special += any(t in (PAD, BOS, EOS) for t in body_ids + title_ids)
            ids_set.add(row['id'])
            groups.add(row['group'])
            bodies.add(fingerprint(canonical(row['content'])))
            ref = Counter(characters(row['title']))
            visible_recall.append(sum((ref & Counter(characters(visible))).values()) / max(1, sum(ref.values())))
            full_recall.append(sum((ref & Counter(characters(row['content']))).values()) / max(1, sum(ref.values())))
            if arrays is not None:
                x, y = sft_sequence(tok, row, manifest['block_size'])
                mismatched += not (np.array_equal(x, arrays[0][i]) and np.array_equal(y, arrays[1][i]))
                stream_hash.update(np.asarray([BOS] + tok.encode(row['title'] + '\n' + row['content']).ids + [EOS], dtype='<u4').tobytes())
        sets[split] = (ids_set, groups, bodies)
        report['data'][split] = {'n': len(rows), 'body_tokens': summary(body_lengths),
            'title_tokens': summary(title_lengths), 'visible_body_characters': summary(visible_chars),
            'body_truncated_n': truncated, 'body_truncated_rate': truncated / len(rows),
            'title_truncated_n': sum(t > TITLE_TOKENS - 1 for t in title_lengths),
            'tokenizer_roundtrip_mismatch_n': bad_roundtrip, 'literal_special_token_n': bad_special,
            'prepared_sft_row_mismatch_n': mismatched if arrays is not None else None,
            'pretrain_stream_matches_reconstruction': stream_hash.hexdigest() == manifest['files'][f'{split}.bin'] if arrays is not None else None,
            'reference_character_recall_visible_mean': float(np.mean(visible_recall)),
            'reference_character_recall_full_mean': float(np.mean(full_recall)),
            'reference_character_recall_note': 'Lexical diagnostic only; not semantic evidence coverage.'}
        print(f'Checked {split}: {len(rows)} records', flush=True)
    checks['cross_split_overlap'] = {
        a + '_' + b: dict(zip(('id', 'lead_group', 'canonical_body'), (len(x & y) for x, y in zip(sets[a], sets[b]))))
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    for stage in ('pretrain', 'sft', 'scratch'):
        run = json.loads((runs / stage / 'run.json').read_text(encoding='utf8'))
        status = json.loads((runs / stage / 'status.json').read_text(encoding='utf8'))
        logs = [json.loads(line) for line in (runs / stage / 'metrics.jsonl').read_text().splitlines()]
        vals = [r for r in logs if r['event'] == 'val']
        trains = [r for r in logs if r['event'] == 'train']
        tc = run['config']['train']
        generator = __import__('torch').Generator().manual_seed(run['config']['seed'] + 1)
        if stage != 'pretrain':
            draws = __import__('torch').randint(manifest['counts']['train'], (tc['max_steps'] * tc['batch_size'] * tc['grad_accum'],), generator=generator).numpy()
            coverage = len(np.unique(draws)) / manifest['counts']['train']
        else:
            coverage = None
        report['training'][stage] = {'complete': status['complete'], 'step': status['step'],
            'best_validation_step': min(vals, key=lambda r: r['loss'])['step'],
            'validation': [{'step': r['step'], 'loss': r['loss'], 'target_tokens': r['target_tokens']} for r in vals],
            'last_20_logged_training_loss_mean': float(np.mean([r['loss'] for r in trains[-20:]])),
            'source_matches_run': report['source_sha256'] == run['environment']['source_sha256'],
            'manifest_matches_run': sha256(data / 'manifest.json') == run['data_manifest_sha256'],
            'sft_training_unique_example_coverage_reconstructed': coverage,
            'sft_validation_sampled_examples': tc['batch_size'] * tc['eval_batches'] if stage != 'pretrain' else None,
            'status': status}
    for split in ('val', 'test'):
        selection_ids = []
        for method in ('lead', 'pretrain', 'sft', 'scratch'):
            folder = runs / f'eval_{split}_{method}'
            rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf8').splitlines()]
            m = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
            scores = [rouge(r['prediction'], r['reference']) for r in rows]
            means = {k: float(np.mean([s[k] for s in scores])) for k in ('rouge1', 'rouge2', 'rougeL')}
            repeated, numeric, replacement, raw_repeated = [], [], [], []
            for r in rows:
                p = ''.join(characters(r['prediction']))
                counts = Counter(p[i:i+4] for i in range(len(p)-3))
                if counts and max(counts.values()) >= 3:
                    repeated.append(r['id'])
                raw_counts = Counter(r['prediction'][i:i+4] for i in range(len(r['prediction'])-3))
                if raw_counts and max(raw_counts.values()) >= 3:
                    raw_repeated.append(r['id'])
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', r['prediction'])
                if any(n not in r['content'] for n in numbers):
                    numeric.append(r['id'])
                if '\ufffd' in r['prediction']:
                    replacement.append(r['id'])
            ids = [r['id'] for r in rows]
            selection_ids.append(ids)
            report['evaluations'][f'{split}_{method}'] = {'n': len(rows), 'metrics_recomputed': means,
                'stored_metrics_match': all(abs(means[k] - m['metrics'][k]) < 1e-12 for k in means),
                'row_scores_match': all(all(abs(s[k]-r['scores'][k]) < 1e-12 for k in s) for s, r in zip(scores, rows)),
                'empty_n': sum(not characters(r['prediction']) for r in rows),
                'normalized_fourgram_at_least_three_times_n': len(repeated),
                'raw_fourgram_at_least_three_times_n': len(raw_repeated), 'numeric_not_in_full_body_n': len(numeric),
                'unicode_replacement_n': len(replacement), 'heuristic_note': 'Flags are not factual error labels.'}
            if split == 'val' and method == 'sft':
                chosen = list(dict.fromkeys(ids[:8] + repeated[:4] + numeric[:4]))
                byid = {r['id']: r for r in rows}
                for key in chosen:
                    r = byid[key]
                    source = val_by_id[key]
                    visible = tok.decode(tok.encode(source['content']).ids[:budget])
                    report['validation_examples'][key] = {**r, 'visible_body': visible,
                        'body_truncated': len(tok.encode(source['content']).ids) > budget,
                        'normalized_repetition_flag': key in repeated, 'numeric_flag': key in numeric}
        checks[f'{split}_identical_comparison_subsets'] = all(ids == selection_ids[0] for ids in selection_ids)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('Audit artifact:', out, flush=True)


if __name__ == '__main__':
    main()
