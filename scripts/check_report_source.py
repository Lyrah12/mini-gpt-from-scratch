"""Lightweight source checks only: not a TeX compiler or a layout verification."""
import re
from pathlib import Path


def main():
    root = Path("report")
    for path in root.rglob("*.tex"):
        text = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
        stack = []
        for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
            if match[1] == "begin":
                stack.append(match[2])
            elif not stack or stack.pop() != match[2]:
                raise ValueError(f"Unbalanced environment in {path}: {match[0]}")
        if stack:
            raise ValueError(f"Unclosed environment in {path}: {stack}")
        for filename in re.findall(r"\\input\{([^}]+)\}", text):
            if not (root / filename).is_file():
                raise FileNotFoundError(root / filename)
    text = (root / "main.tex").read_text(encoding="utf-8")
    citations = {item.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for item in group.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    assert citations <= bibitems, citations - bibitems
    print("PASS: LaTeX environment balance, input files and citation keys. PDF not compiled; layout not checked.")


if __name__ == "__main__":
    main()
