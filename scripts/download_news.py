"""Download the author's news2016zh archive; no automatic multi-GB local download."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import gdown

SOURCE = "https://github.com/brightmart/nlp_chinese_corpus"
DRIVE_ID = "1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/raw")
    args = ap.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "new2016zh.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        print("Author's archive is about 3.6 GB. Download on cloud or use the manual Baidu route.")
        result = gdown.download(id=DRIVE_ID, output=str(archive), quiet=False, resume=True)
        if result is None:
            raise RuntimeError(f"Download failed; use the author's Baidu link at {SOURCE}")
    if not zipfile.is_zipfile(archive):
        raise ValueError("Not a valid ZIP; use the author's download links and verify the file")
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Archive path escapes raw-data directory")
            if target.exists() and not item.is_dir():
                raise FileExistsError(f"Already extracted: {target}; do not overwrite existing data")
        z.extractall(root)
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    (root / "download_manifest.json").write_text(json.dumps({"source": SOURCE,
        "drive_id": DRIVE_ID, "archive_sha256": digest.hexdigest()}, indent=2) + "\n", encoding="utf-8")
    print("Extracted. Pass the news2016zh train/valid JSONL files, not download_manifest.json, to minigpt.prepare.")


if __name__ == "__main__":
    main()
