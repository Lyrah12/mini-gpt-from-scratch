"""对照 XL-Sum 原始摘要，检查正文输入包含的标题信息。"""
# 导入本模块使用的标准库工具。
import argparse
import json
from pathlib import Path
import sys
import tarfile


# 将固定验证新闻映射回原始记录，比较摘要带来的信息变化。
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
    # 回查官方验证分区，通过规范化正文摘要匹配已处理记录。
    with tarfile.open(root / 'data/raw/xlsum/chinese_simplified_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
        # 只读取验证分区对应的 JSONL 文件。
        for member in tar:
            if not (member.isfile() and ('validation' in member.name or member.name.endswith('_val.jsonl'))):
                continue
            # 复用原清洗和去重规则，逐条匹配已保存的验证新闻。
            for line in tar.extractfile(member):
                raw = json.loads(line)
                title, body = clean(raw['title']), clean(raw['text'])
                if title and body.startswith(title):
                    body = body[len(title):].lstrip(' ：:。- ')
                key = fingerprint(canonical(body))
                if key not in byid:
                    continue
                # 分别计算完整文本和截断前缀中的标题字符覆盖比例。
                summary = clean(raw['summary'])
                ref = Counter(characters(title))
                # 计算参考标题字符在指定新闻文本中的覆盖比例。
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


# 直接运行模块时，从命令行入口启动。
if __name__ == '__main__':
    main()
