"""按字符计算 ROUGE-F1，保留 Unicode 文字和数字。"""
# 导入本模块使用的标准库工具。
from collections import Counter
import unicodedata


# 保留汉字等文字和数字，并将英文统一转为小写。
def characters(text):
    return [c.lower() for c in text if unicodedata.category(c)[0] in ("L", "N")]


# 根据匹配数和两侧总数计算 F1，空分母返回零。
def f1(overlap, pred_count, ref_count):
    return 2 * overlap / (pred_count + ref_count) if pred_count + ref_count else 0.0


# 用字符 n-gram 和最长公共子序列计算三种 ROUGE-F1。
def rouge(prediction, reference):
    p, r = characters(prediction), characters(reference)
    result = {}
    # 使用多重集交集统计重复字符片段，避免重复项被多次匹配。
    for n in (1, 2):
        pc = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        result[f"rouge{n}"] = f1(sum((pc & rc).values()), sum(pc.values()), sum(rc.values()))
    dp = [0] * (len(r) + 1)
    # LCS 只保留上一行。prev 在更新前保存左上角，dp[j] 保存上方单元格。
    for a in p:
        prev = 0
        for j, b in enumerate(r, 1):
            old = dp[j]
            dp[j] = prev + 1 if a == b else max(dp[j], dp[j - 1])
            prev = old
    result["rougeL"] = f1(dp[-1], len(p), len(r))
    return result
