"""下载并校验 news2016zh 新闻压缩包，作为备用数据来源。"""
# 导入本模块使用的标准库工具。
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

# 使用 gdown 从原作者的 Google Drive 地址下载压缩包。
import gdown

# 保留原作者的发布地址与下载文件标识。
SOURCE = "https://github.com/brightmart/nlp_chinese_corpus"
DRIVE_ID = "1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp"


# 下载备用新闻语料，检查压缩包路径并保存文件摘要。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/raw")
    args = ap.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "new2016zh.zip"
    # 仅在缺少有效压缩包时从原作者的下载入口获取数据。
    if not archive.exists() or not zipfile.is_zipfile(archive):
        print("Author's archive is about 3.6 GB. Download on cloud or use the manual Baidu route.")
        result = gdown.download(id=DRIVE_ID, output=str(archive), quiet=False, resume=True)
        if result is None:
            raise RuntimeError(f"Download failed; use the author's Baidu link at {SOURCE}")
    if not zipfile.is_zipfile(archive):
        raise ValueError("Not a valid ZIP; use the author's download links and verify the file")
    # 解压前核对目标路径和已有文件，避免越界或覆盖。
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Archive path escapes raw-data directory")
            if target.exists() and not item.is_dir():
                raise FileExistsError(f"Already extracted: {target}; do not overwrite existing data")
        z.extractall(root)
    # 记录原始压缩包摘要，便于之后检查来源是否一致。
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    (root / "download_manifest.json").write_text(json.dumps({"source": SOURCE,
        "drive_id": DRIVE_ID, "archive_sha256": digest.hexdigest()}, indent=2) + "\n", encoding="utf-8")
    print("Extracted. Pass the news2016zh train/valid JSONL files, not download_manifest.json, to minigpt.prepare.")


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
