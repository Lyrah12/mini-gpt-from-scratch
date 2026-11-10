"""Download pinned official XL-Sum simplified-Chinese news and prepare title targets."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import requests

from minigpt.common import sha256, write_json
from minigpt.prepare import prepare

REVISION = "30fece425f9a3866e04321773ca7a80056d55ca6"
ARCHIVE_SHA256 = "eaa26651ca379a6f689cecd459eede68c3a67b0de64ce9e9bea62889c252fb2e"
ARCHIVE_SIZE = 54528291
SOURCE = "https://huggingface.co/datasets/csebuetnlp/xlsum"
FILE = "data/chinese_simplified_XLSum_v2.0.tar.bz2"


def download(root):
    from scripts.range_download import download as ranged_download
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(FILE).name
    errors = []
    for host in ("huggingface.co", "hf-mirror.com"):
        url = f"https://{host}/datasets/csebuetnlp/xlsum/resolve/{REVISION}/{FILE}"
        try:
            return ranged_download(url, archive, ARCHIVE_SIZE, ARCHIVE_SHA256)
        except (RuntimeError, ValueError, requests.RequestException) as error:
            errors.append(str(error))
    raise RuntimeError("All pinned download routes failed: " + "; ".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/xlsum")
    ap.add_argument("--output", default="data/processed_xlsum")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.raw)
    archive = download(root)
    normalized = root / "normalized_v2"
    source_manifest = {"source": SOURCE, "revision": REVISION, "configuration": "chinese_simplified",
                       "archive_sha256": ARCHIVE_SHA256, "archive_bytes": ARCHIVE_SIZE,
                       "license": "CC-BY-NC-SA-4.0", "target": "title (not summary)", "counts": {}}
    if not normalized.exists():
        normalized.mkdir()
        with tarfile.open(archive, "r:bz2") as tar:
            for item in tar.getmembers():
                if not item.isfile() or not item.name.endswith(".jsonl"):
                    continue
                stem = Path(item.name).stem
                split = next((s for s in ("train", "validation", "val", "test") if stem.endswith("_" + s)), None)
                if split is None:
                    continue
                split = "val" if split == "validation" else split
                count = 0
                with tar.extractfile(item) as original, (normalized / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                    for line in original:
                        row = json.loads(line)
                        record = {"content": row["text"], "title": row["title"], "_split": split,
                                  "source_id": row["id"], "url": row["url"]}
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                source_manifest["counts"][split] = count
        if set(source_manifest["counts"]) != {"train", "val", "test"}:
            raise ValueError("Official archive does not contain all expected partitions")
        write_json(root / "download_manifest.json", source_manifest)
    else:
        source_manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(source_manifest, ensure_ascii=False), flush=True)
    if args.download_only:
        return
    manifest = prepare([str(normalized)], args.output, source=SOURCE, preserve_splits=True)
    manifest["upstream_dataset"] = source_manifest
    write_json(Path(args.output) / "manifest.json", manifest)


if __name__ == "__main__":
    main()
