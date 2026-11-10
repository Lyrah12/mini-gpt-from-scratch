"""Inspect original XL-Sum summaries omitted from the text-to-title input."""
import argparse
import json
from pathlib import Path
import sys
import tarfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    sys.path.insert(0, str(root))
    from minigpt.prepare import clean, canonical, fingerprint
    from minigpt.metrics import characters
    from minigpt.tokenizer import load_tokenizer, PREFIX, SUFFIX, TITLE_TOKENS
    from collections import Counter
    import numpy as np
    rows = [json.loads(l) for l in (root / 'runs/local_4060/eval_val_sft/predictions.jsonl').read_text(encoding='utf8').splitlines()]
    byid = {r['id']: r for r in rows}
    first = {r['id'] for r in rows[:8]}
    tok = load_tokenizer(root / 'data/processed_xlsum/tokenizer.json')
    budget = 512 - TITLE_TOKENS - len(tok.encode(PREFIX).ids) - len(tok.encode(SUFFIX).ids) - 1
    found = []
    with tarfile.open(root / 'data/raw/xlsum/chinese_simplified_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
        for member in tar:
            if not (member.isfile() and ('validation' in member.name or member.name.endswith('_val.jsonl'))):
                continue
            for line in tar.extractfile(member):
                raw = json.loads(line)
                title, body = clean(raw['title']), clean(raw['text'])
                if title and body.startswith(title):
                    body = body[len(title):].lstrip(' ：:。- ')
                key = fingerprint(canonical(body))
                if key not in byid:
                    continue
                summary = clean(raw['summary'])
                ref = Counter(characters(title))
                def recall(text):
                    return sum((ref & Counter(characters(text))).values()) / max(1, sum(ref.values()))
                found.append({'id': key, 'source_id': raw['id'], 'url': raw['url'],
                    'reference': title, 'summary': summary, 'summary_in_text': summary in body,
                    'first_eight': key in first,
                    'reference_char_recall_text': recall(body),
                    'reference_char_recall_summary_and_text': recall(summary + body),
                    'reference_char_recall_visible_text': recall(tok.decode(tok.encode(body).ids[:budget])),
                    'reference_char_recall_visible_summary_and_text': recall(tok.decode(tok.encode(summary + ' ' + body).ids[:budget]))})
    result = {'scope': 'original validation source inspection only; no evaluation of a modified task',
        'matched': len(found), 'unique_matched': len({r['id'] for r in found}),
        'summary_absent_from_text_n': sum(not r['summary_in_text'] for r in found),
        'note': 'Original summary is a distinct dataset field. Adding it changes the input/task and requires separate training and evaluation; do not compare as identical inputs.',
        'lexical_means': {k: float(np.mean([r[k] for r in found])) for k in (
            'reference_char_recall_text', 'reference_char_recall_summary_and_text',
            'reference_char_recall_visible_text', 'reference_char_recall_visible_summary_and_text')},
        'samples': sorted(found, key=lambda r: r['id'])}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
