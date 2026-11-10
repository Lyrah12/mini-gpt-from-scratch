"""Clean, deduplicate, split, learn BPE, and encode news2016zh JSONL."""
import argparse
import gzip
import hashlib
import html
import json
import random
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .common import jsonl, sha256, write_json
from .tokenizer import BOS, EOS, sft_sequence, train_tokenizer


def clean(text):
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c if not unicodedata.category(c).startswith("C") else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def canonical(text):
    return "".join(c.lower() for c in text if c.isalnum())


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_name(group, seed):
    bucket = int(fingerprint(f"{seed}:{group}")[:8], 16) % 1000
    return "test" if bucket < 20 else "val" if bucket < 40 else "train"


def prepare(inputs, output, max_records=300000, seed=42, vocab_size=16000, block_size=512,
            min_chars=100, max_chars=10000, source="https://github.com/brightmart/nlp_chinese_corpus", preserve_splits=False):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty to protect existing data: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in inputs:
        p = Path(item)
        paths.extend(sorted(f for f in p.rglob("*") if f.is_file() and
                            f.name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))) if p.is_dir() else [p])
    paths = sorted(set(p.resolve() for p in paths))
    if not paths:
        raise ValueError("No JSONL input files found")
    if max_records < 100 or block_size < 128:
        raise ValueError("Use at least 100 records and block_size >= 128")
    stats = Counter()
    reservoir = []
    rng = random.Random(seed)
    # Disk-backed keys keep deduplication memory bounded for the full 9 GB corpus.
    db = sqlite3.connect(out / "dedup.sqlite")
    db.execute("CREATE TABLE seen (body TEXT PRIMARY KEY, lead TEXT UNIQUE)")
    source_files = []
    try:
        for path in paths:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8-sig") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    stats["raw_records"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["invalid_json"] += 1
                        continue
                    if not isinstance(row, dict):
                        stats["invalid_schema"] += 1
                        continue
                    provided_split = row.get("_split")
                    if preserve_splits and provided_split not in ("train", "val", "test"):
                        raise ValueError("Every record must provide _split=train/val/test")
                    title, body = clean(row.get("title")), clean(row.get("content"))
                    # Remove a copied headline at the very start of an article.
                    if title and body.startswith(title):
                        body = body[len(title):].lstrip(" ：:。- ")
                    chinese = sum("\u4e00" <= c <= "\u9fff" for c in body)
                    if not (4 <= len(title) <= 80 and min_chars <= len(body) <= max_chars
                            and chinese / max(1, len(body)) >= 0.5):
                        stats["filtered"] += 1
                        continue
                    norm = canonical(body)
                    key, group = fingerprint(norm), fingerprint(norm[:256])
                    try:
                        db.execute("INSERT INTO seen VALUES (?, ?)", (key, group))
                    except sqlite3.IntegrityError:
                        stats["duplicate_body_or_lead"] += 1
                        continue
                    stats["unique_eligible"] += 1
                    item = {"id": key, "group": group, "title": title, "content": body}
                    if preserve_splits:
                        item["_split"] = provided_split
                    n = stats["unique_eligible"]
                    if len(reservoir) < max_records:
                        reservoir.append(item)
                    else:
                        j = rng.randrange(n)
                        if j < max_records:
                            reservoir[j] = item
                    if stats["raw_records"] % 100000 == 0:
                        db.commit()
                        print(f"Scanned {stats['raw_records']:,}; unique {n:,}", flush=True)
            source_files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    finally:
        db.commit()
        db.close()
    if len(reservoir) < 100:
        raise ValueError("Too few usable records after filtering")
    counts = Counter()
    # Sorting makes partition order independent of the reservoir's internal replacement order.
    with ExitStack() as stack:
        writers = {s: stack.enter_context(open(out / f"{s}.jsonl", "w", encoding="utf-8"))
                   for s in ("train", "val", "test")}
        for row in sorted(reservoir, key=lambda r: r["id"]):
            split = row["_split"] if preserve_splits else split_name(row["group"], seed)
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    if min(counts[s] for s in ("train", "val", "test")) < 2:
        raise ValueError("A partition has fewer than 2 documents; use more records")

    def texts():
        for row in jsonl(out / "train.jsonl"):
            yield row["title"] + "\n" + row["content"]

    tok = train_tokenizer(texts(), out / "tokenizer.json", vocab_size)
    token_counts = {}
    for split in ("train", "val"):
        count = 0
        with open(out / f"{split}.bin", "wb") as stream:
            for row in jsonl(out / f"{split}.jsonl"):
                ids = [BOS] + tok.encode(row["title"] + "\n" + row["content"]).ids + [EOS]
                np.asarray(ids, dtype="<u4").tofile(stream)
                count += len(ids)
        token_counts[split] = count
        x = np.lib.format.open_memmap(out / f"sft_{split}_ids.npy", mode="w+", dtype=np.int32,
                                     shape=(counts[split], block_size + 1))
        y = np.lib.format.open_memmap(out / f"sft_{split}_labels.npy", mode="w+", dtype=np.int32,
                                     shape=(counts[split], block_size + 1))
        for i, row in enumerate(jsonl(out / f"{split}.jsonl")):
            x[i], y[i] = sft_sequence(tok, row, block_size)
        x.flush()
        y.flush()
        del x, y
    artifact_files = sorted(p for p in out.iterdir() if p.suffix in (".bin", ".npy", ".jsonl"))
    manifest = {"schema": 1, "source": source,
                "source_files": source_files, "seed": seed, "max_records": max_records,
                "sampling": "reservoir over all unique eligible documents; file paths sorted",
                "dedup": "NFKC + alphanumeric body hash; identical first 256 normalized characters removed",
                "split": ("preserve provided official train/val/test labels; dedup across all partitions" if preserve_splits else
                          "sha256(seed:lead_hash) mod 1000; train 960/1000, val/test 20/1000 each"),
                "limitations": "Heuristic prefix dedup does not guarantee removal of all near-duplicates or events",
                "filter": {"min_chars": min_chars, "max_chars": max_chars, "chinese_ratio_min": 0.5,
                           "title_chars": [4, 80]},
                "counts": dict(counts), "statistics": dict(stats), "token_counts": token_counts,
                "requested_vocab_size": vocab_size, "vocab_size": tok.get_vocab_size(),
                "block_size": block_size, "tokenizer_sha256": sha256(out / "tokenizer.json"),
                "files": {p.name: sha256(p) for p in artifact_files}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"counts": dict(counts), "tokens": token_counts, "vocab": tok.get_vocab_size()}, ensure_ascii=False))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True, help="JSONL files or a raw-data directory")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--max-records", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--source", default="https://github.com/brightmart/nlp_chinese_corpus")
    ap.add_argument("--preserve-splits", action="store_true", help="Keep each record's official _split label")
    args = ap.parse_args()
    prepare(args.input, args.output, args.max_records, args.seed, args.vocab_size, args.block_size, source=args.source, preserve_splits=args.preserve_splits)


if __name__ == "__main__":
    main()
