"""打包源码、配置、测试和依赖文件，供云端运行。"""
# 导入本模块使用的标准库工具。
import argparse
import zipfile
from pathlib import Path

# 明确打包所需的源码目录和根文件。
DIRECTORIES = ["minigpt", "configs", "scripts", "tests"]
FILES = ["README.md", "requirements.txt", "requirements-dev.txt", "pyproject.toml", ".gitignore", "LICENSE"]


# 按源码目录清单打包文件，忽略字节码缓存。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/cloud-upload.zip")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a previous upload bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 收集必需根文件和源码目录，缓存字节码不参与打包。
    candidates = [root / f for f in FILES]
    for d in DIRECTORIES:
        candidates += [p for p in (root / d).rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    # 按路径排序写入压缩包，保留项目中的相对目录结构。
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Cloud source archive: {destination} ({destination.stat().st_size:,} bytes)")
    print("The uncompressed report/ folder stays local; cloud export creates report/generated/ only.")


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
