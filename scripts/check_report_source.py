"""检查 LaTeX 环境配对、引用和源文件依赖。"""
# 导入本模块使用的标准库工具。
import re
from pathlib import Path


# 检查环境嵌套、输入依赖和参考文献键是否完整。
def main():
    root = Path("report")
    # 先移除 LaTeX 注释，再逐个检查环境栈与输入文件。
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
    # 收集正文引用和文末条目，检查所有引用键都有定义。
    citations = {item.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for item in group.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    assert citations <= bibitems, citations - bibitems
    print("PASS: LaTeX environment balance, input files and citation keys. PDF not compiled; layout not checked.")


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
