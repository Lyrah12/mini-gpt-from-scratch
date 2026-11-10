"""Evaluate fixed original IDs, with explicit data/checkpoint/decoding provenance."""
import json
from pathlib import Path
import random
import re
import time

import torch

from minigpt.common import environment, jsonl, object_hash, read_json, sha256, write_json
from minigpt.evaluate import load_model
from minigpt.metrics import characters, rouge
from minigpt.tokenizer import EOS, PREFIX, SUFFIX, load_tokenizer, title_prompt
from scripts.title_decoding import MODES, decode, repetition_flag


def select_rows(data, split, limit, seed=42):
    rng, rows = random.Random(seed), []
    for n, row in enumerate(jsonl(Path(data) / f'{split}.jsonl'), 1):
        if len(rows) < limit:
            rows.append(row)
        else:
            j = rng.randrange(n)
            if j < limit:
                rows[j] = row
    return sorted(rows, key=lambda r: r['id'])


def evaluate(data, output, checkpoint=None, split='val', limit=200, mode='greedy', seed=42):
    torch.set_num_threads(4)
    data, out = Path(data), Path(output)
    manifest = read_json(data / 'manifest.json')
    if sha256(data / f'{split}.jsonl') != manifest['files'][f'{split}.jsonl']:
        raise ValueError('Evaluation partition changed')
    if sha256(data / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    rows = select_rows(data, split, limit, seed)
    if not rows:
        raise ValueError('Empty evaluation partition')
    identity = {'split': split, 'limit': limit, 'seed': seed,
        'selection_hash': object_hash([r['id'] for r in rows]),
        'reference_hash': object_hash([(r['id'], r['title']) for r in rows]),
        'data_manifest_sha256': sha256(data / 'manifest.json'),
        'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
        'mode': mode if checkpoint else 'lead32',
        'decoding_settings': MODES[mode] if checkpoint else None,
        'decoder_sha256': sha256(Path(__file__).with_name('title_decoding.py')),
        'evaluator_sha256': sha256(__file__)}
    if (out / 'metrics.json').exists():
        cached = read_json(out / 'metrics.json')
        if cached['identity'] != identity or sha256(out / 'predictions.jsonl') != cached['predictions_sha256']:
            raise ValueError('Cached evaluation has changed')
        return cached
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists() and read_json(out / 'identity.json') != identity:
        raise ValueError('Output directory belongs to another evaluation')
    write_json(out / 'identity.json', identity)
    tok = load_tokenizer(data / 'tokenizer.json')
    model, state = load_model(checkpoint, data / 'tokenizer.json', 'cuda') if checkpoint else (None, None)
    block_size = model.config.block_size if model else manifest['block_size']
    sums = {k: 0. for k in ('rouge1', 'rouge2', 'rougeL')}
    empty = repeated = numbers = eos = 0
    lengths = []
    started = time.perf_counter()
    tmp = out / 'predictions.jsonl.tmp'
    with tmp.open('w', encoding='utf8') as stream:
        for row in rows:
            prompt = title_prompt(tok, row['content'], block_size)
            if model is not None:
                generated = decode(model, prompt, mode)
                prediction = tok.decode(generated, skip_special_tokens=True).strip()
                eos += bool(generated and generated[-1] == EOS)
                lengths.append(len(generated))
            else:
                begin = 1 + len(tok.encode(PREFIX).ids)
                end = len(prompt) - len(tok.encode(SUFFIX).ids)
                visible = tok.decode(prompt[begin:end])
                prediction = re.split(r'[。！？!?]', visible, maxsplit=1)[0][:32]
            scores = rouge(prediction, row['title'])
            for key in sums:
                sums[key] += scores[key]
            empty += not bool(characters(prediction))
            repeated += repetition_flag(prediction)
            numbers += any(n not in row['content'] for n in re.findall(r'[0-9]+(?:\.[0-9]+)?', prediction))
            stream.write(json.dumps({'id': row['id'], 'content': row['content'], 'reference': row['title'],
                'prediction': prediction, 'scores': scores}, ensure_ascii=False) + '\n')
    tmp.replace(out / 'predictions.jsonl')
    result = {'identity': identity, 'n': len(rows), 'metrics': {k: v / len(rows) for k, v in sums.items()},
        'metric': 'macro character ROUGE F1; Unicode L/N, lowercase, punctuation/whitespace removed',
        'empty_n': empty, 'raw_repetition_flags': repeated, 'numeric_flags': numbers,
        'flags_note': 'Heuristics, not factual error rates; source strings differ between input variants.',
        'ended_with_eos_n': eos if model else None,
        'generated_tokens_mean': sum(lengths) / len(lengths) if lengths else None,
        'predictions_sha256': sha256(out / 'predictions.jsonl'),
        'checkpoint_step': state['step'] if state else None,
        'seconds': time.perf_counter() - started, 'environment': environment()}
    write_json(out / 'metrics.json', result)
    print(f'{out.name}: ROUGE-L={100*result["metrics"]["rougeL"]:.3f}, repetition={repeated}/{len(rows)}', flush=True)
    return result
