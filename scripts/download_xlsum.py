"""下载固定版本的 XL-Sum 简体中文新闻并准备标题任务数据。"""
# 导入本模块使用的标准库工具。
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# 使用 HTTP 请求库识别并处理下载异常。
import requests

# 复用项目内的数据、模型或实验工具。
from minigpt.common import sha256, write_json
from minigpt.prepare import prepare

# 固定官方数据版本、压缩包大小与摘要，下载后据此校验。
REVISION = "30fece425f9a3866e04321773ca7a80056d55ca6"
ARCHIVE_SHA256 = "eaa26651ca379a6f689cecd459eede68c3a67b0de64ce9e9bea62889c252fb2e"
ARCHIVE_SIZE = 54528291
SOURCE = "https://huggingface.co/datasets/csebuetnlp/xlsum"
FILE = "data/chinese_simplified_XLSum_v2.0.tar.bz2"


# 依次尝试固定版本的下载地址，并校验原始压缩包。
def download(root):
    from scripts.range_download import download as ranged_download
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(FILE).name
    errors = []
    # 主站失败时尝试镜像，两个地址使用相同版本与文件摘要。
    for host in ("huggingface.co", "hf-mirror.com"):
        url = f"https://{host}/datasets/csebuetnlp/xlsum/resolve/{REVISION}/{FILE}"
        try:
            return ranged_download(url, archive, ARCHIVE_SIZE, ARCHIVE_SHA256)
        except (RuntimeError, ValueError, requests.RequestException) as error:
            errors.append(str(error))
    raise RuntimeError("All pinned download routes failed: " + "; ".join(errors))


# 转换官方字段与分区标记，再调用统一的数据准备流程。
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
    # 首次下载后将官方 text、title 与分区标记转换为统一字段。
    if not normalized.exists():
        normalized.mkdir()
        # 直接从压缩包读取官方记录，并保留其原始分区归属。
        with tarfile.open(archive, "r:bz2") as tar:
            # 筛选官方 JSONL 分区文件，再统一字段名和验证集标记。
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
        # 确认训练、验证和测试分区全部存在后保存下载记录。
        if set(source_manifest["counts"]) != {"train", "val", "test"}:
            raise ValueError("Official archive does not contain all expected partitions")
        write_json(root / "download_manifest.json", source_manifest)
    else:
        source_manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(source_manifest, ensure_ascii=False), flush=True)
    if args.download_only:
        return
    # 保持官方划分，调用共用清洗、分词和编码流程。
    manifest = prepare([str(normalized)], args.output, source=SOURCE, preserve_splits=True)
    manifest["upstream_dataset"] = source_manifest
    write_json(Path(args.output) / "manifest.json", manifest)


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
