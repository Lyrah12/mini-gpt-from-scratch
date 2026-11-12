"""根据试跑的稳定训练步耗时，估算剩余时间和计算费用。"""
# 导入本模块使用的标准库工具。
import argparse
import statistics
from pathlib import Path

# 复用项目内的数据、模型或实验工具。
from minigpt.common import jsonl, read_json


# 用稳定训练步的中位耗时估算剩余训练时间。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/cloud/pretrain")
    ap.add_argument("--hourly-price", type=float, help="Actual instance GPU hourly price shown by AutoDL")
    args = ap.parse_args()
    root = Path(args.run)
    info, status = read_json(root / "run.json"), read_json(root / "status.json")
    # 取预热后的已记录训练步，用中位数降低偶发抖动影响。
    rows = [r for r in jsonl(root / "metrics.jsonl") if r["event"] == "train" and r["step"] >= 10]
    if len(rows) < 2:
        raise ValueError("Need at least two logged training steps after warm-up; run a 30-step pilot")
    seconds = statistics.median(r["step_seconds"] for r in rows)
    hours = seconds * (status["planned_steps"] - status["step"]) / 3600
    print(f"GPU: {info['environment']['gpu']}; precision: {info['actual_precision']}")
    print(f"Median sampled step: {seconds:.3f} s; remaining pretrain compute: {hours:.2f} h")
    print(f"Peak allocated GPU memory: {max(r['peak_cuda_memory_bytes'] for r in rows) / 2**30:.2f} GiB")
    # 用户提供实际单价时，再估算剩余训练计算费用。
    if args.hourly_price is not None:
        if args.hourly_price < 0:
            raise ValueError("Hourly price cannot be negative")
        print(f"Estimated remaining pretrain compute charge: {hours * args.hourly_price:.2f} yuan")
    print("Estimate excludes validation, checkpoint writes, downloads, SFT, evaluation and storage fees.")


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
