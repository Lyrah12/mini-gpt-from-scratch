"""Package only source/config/docs for cloud upload; no data, weights, AGENTS or report."""
import argparse
import zipfile
from pathlib import Path

DIRECTORIES = ["minigpt", "configs", "scripts", "tests", "docs"]
FILES = ["README.md", "requirements.txt", "requirements-dev.txt", "pyproject.toml", ".gitignore", "LICENSE"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/cloud-upload.zip")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a previous upload bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = [root / f for f in FILES]
    for d in DIRECTORIES:
        candidates += [p for p in (root / d).rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Cloud source archive: {destination} ({destination.stat().st_size:,} bytes)")
    print("The uncompressed report/ folder stays local; cloud export creates report/generated/ only.")


if __name__ == "__main__":
    main()
