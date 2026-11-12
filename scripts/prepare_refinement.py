"""沿用原有样本、划分和词表，加入 XL-Sum 摘要后构造微调输入。"""
# 导入本模块使用的标准库工具。
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tarfile

# 使用 NumPy 内存映射重新生成微调输入和标签。
import numpy as np

# 复用项目内的数据、模型或实验工具。
from minigpt.common import jsonl, read_json, sha256, write_json
from minigpt.prepare import canonical, clean, fingerprint
from minigpt.tokenizer import PREFIX, SUFFIX, TITLE_TOKENS, load_tokenizer, sft_sequence
from scripts.download_xlsum import ARCHIVE_SHA256


# 清洗摘要后拼接正文，并避免重复加入已有摘要。
def combine_article(summary, body):
    lead = clean(summary)
    return body if not lead or body.startswith(lead) else lead + ' ' + body


# 匹配原始新闻摘要，保留原划分和词表，重新生成微调数组。
def prepare(base, archive, output):
    base, archive, out = Path(base), Path(archive), Path(output)
    manifest = read_json(base / 'manifest.json')
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Original XL-Sum archive checksum mismatch')
    for name in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
        if sha256(base / name) != manifest['files'][name]:
            raise ValueError('Changed base partition: ' + name)
    if sha256(base / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Changed base tokenizer')
    # 新增数据清单记录原始数据版本和摘要来源。
    identity = {'base_manifest_sha256': sha256(base / 'manifest.json'),
                'archive_sha256': ARCHIVE_SHA256,
                'input_definition': 'original summary as lead + original processed text; target=title'}
    # 已有处理结果通过身份与文件摘要校验后直接复用。
    if (out / 'manifest.json').exists():
        cached = read_json(out / 'manifest.json')
        if cached.get('refinement') != identity:
            raise ValueError('Existing refinement data has a different identity')
        for name, digest in cached['files'].items():
            if sha256(out / name) != digest:
                raise ValueError('Changed refinement artifact: ' + name)
        return cached
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty refinement output directory')
    rows = {s: list(jsonl(base / f'{s}.jsonl')) for s in ('train', 'val', 'test')}
    wanted = {(s, r['id']): r for s in rows for r in rows[s]}
    leads = {}
    # 回查官方原始记录，用相同正文清洗规则找到对应摘要。
    with tarfile.open(archive, 'r:bz2') as tar:
        # 遍历原始分区文件，按官方分区和正文摘要定位样本。
        for item in tar:
            if not item.isfile() or not item.name.endswith('.jsonl'):
                continue
            stem = Path(item.name).stem
            split = next((s for s in ('train', 'validation', 'val', 'test') if stem.endswith('_' + s)), None)
            split = 'val' if split == 'validation' else split
            if split is None:
                continue
            # 清洗原始标题与正文，只有两者均匹配时才接受对应摘要。
            for line in tar.extractfile(item):
                raw = json.loads(line)
                title, body = clean(raw['title']), clean(raw['text'])
                if title and body.startswith(title):
                    body = body[len(title):].lstrip(' ：:。- ')
                key = split, fingerprint(canonical(body))
                if key not in wanted:
                    continue
                original = wanted[key]
                if original['title'] == title and original['content'] == body:
                    leads.setdefault(key, {'lead': clean(raw['summary']), 'source_id': raw['id'], 'url': raw['url']})
    # 所有保留文档都应准确映射回原始数据后再生成新输入。
    if set(leads) != set(wanted):
        raise ValueError(f'Could not match {len(set(wanted) - set(leads))} original records exactly')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / 'tokenizer.json', out / 'tokenizer.json')
    tok = load_tokenizer(out / 'tokenizer.json')
    body_budget = manifest['block_size'] - TITLE_TOKENS - len(tok.encode(PREFIX).ids) - len(tok.encode(SUFFIX).ids) - 1
    stats, body_sets = {}, {}
    # 按原有分区和顺序拼接摘要，重建训练与验证的标签数组。
    for split, records in rows.items():
        count = Counter()
        body_sets[split] = set()
        arrays = None
        if split != 'test':
            arrays = [np.lib.format.open_memmap(out / f'sft_{split}_{kind}.npy', mode='w+', dtype=np.int32,
                      shape=(len(records), manifest['block_size'] + 1)) for kind in ('ids', 'labels')]
        # 逐分区保存摘要加正文，同时为训练和验证生成监督数组。
        with (out / f'{split}.jsonl').open('w', encoding='utf8') as stream:
            # 保持原样本 ID 和顺序，仅更新新闻输入并记录输入长度变化。
            for i, original in enumerate(records):
                source = leads[split, original['id']]
                content = combine_article(source['lead'], original['content'])
                row = {**original, 'content': content, 'source_id': source['source_id'], 'url': source['url']}
                body_sets[split].add(fingerprint(canonical(content)))
                count['records'] += 1
                count['lead_added'] += content != original['content']
                count['title_literal_substring_of_lead'] += original['title'] in source['lead']
                count['body_truncated'] += len(tok.encode(content).ids) > body_budget
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                if arrays is not None:
                    arrays[0][i], arrays[1][i] = sft_sequence(tok, row, manifest['block_size'])
        if arrays is not None:
            for array in arrays:
                array.flush()
            del arrays
        stats[split] = dict(count)
        print(f'Prepared lead+text {split}: {len(records)}', flush=True)
    # 检查加入摘要后的完整新闻在各分区间是否重复。
    overlap = {a + '_' + b: len(body_sets[a] & body_sets[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(overlap.values()):
        raise ValueError('Enriched source bodies overlap across partitions')
    # 沿用原样本和词表信息，更新输入定义、统计和输出文件摘要。
    result = {k: v for k, v in manifest.items() if k not in ('files', 'token_counts', 'statistics')}
    result.update(schema=2, refinement=identity, statistics=stats, cross_split_enriched_body_overlap=overlap,
        id_definition='IDs, groups, ordering and partition memberships preserved from original text-only corpus',
        tokenizer_policy='unchanged original training-only tokenizer; no new vocabulary',
        pretraining_policy='reuse original text-only pretraining; no new pretraining corpus',
        files={p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.jsonl', '.npy')})
    write_json(out / 'manifest.json', result)
    return result


# 解析命令行参数后，沿用原有样本、划分和词表，加入 XL-Sum 摘要后构造微调输入。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='data/processed_xlsum')
    ap.add_argument('--archive', default='data/raw/xlsum/chinese_simplified_XLSum_v2.0.tar.bz2')
    ap.add_argument('--output', default='data/processed_xlsum_lead')
    a = ap.parse_args()
    prepare(a.base, a.archive, a.output)


# 直接运行模块时，从命令行入口启动。
if __name__ == '__main__':
    main()
