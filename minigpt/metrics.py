"""Macro character-level ROUGE F1; Unicode letters/numbers, no whitespace/punctuation."""
from collections import Counter
import unicodedata


def characters(text):
    return [c.lower() for c in text if unicodedata.category(c)[0] in ("L", "N")]


def f1(overlap, pred_count, ref_count):
    return 2 * overlap / (pred_count + ref_count) if pred_count + ref_count else 0.0


def rouge(prediction, reference):
    p, r = characters(prediction), characters(reference)
    result = {}
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
