"""配置读写、随机种子、文件校验和实验环境记录。"""
# 导入本模块使用的标准库工具。
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path


# 读取 UTF-8 JSON，供配置、数据清单和运行状态共用。
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 创建父目录并写入便于阅读的 JSON，同时拒绝非有限数值。
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# 逐行读取非空 JSON 记录，避免一次载入整个语料文件。
def jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


# 分块计算文件摘要，控制大文件校验时的内存占用。
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# 按固定键顺序序列化对象，使相同配置得到相同摘要。
def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# 汇总源码文件摘要，用于记录和校验训练时的代码版本。
def source_fingerprint():
    root = Path(__file__).resolve().parent
    return object_hash({p.name: sha256(p) for p in sorted(root.glob("*.py"))})


# 设置 Python、NumPy、PyTorch 及 CUDA 的随机种子。
def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 记录软件版本、硬件、Git 状态和源码摘要。
def environment():
    import numpy as np
    import torch
    import tokenizers
    # 读取可用的 Git 信息，普通源码目录中允许缺少 Git 仓库。
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    # 合并软硬件版本与线程设置，写入实验的环境记录。
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "tokenizers": tokenizers.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_revision": revision, "git_dirty": dirty, "source_sha256": source_fingerprint(),
            "cpu_threads": torch.get_num_threads(),
            "logical_cpus": os.cpu_count()}
