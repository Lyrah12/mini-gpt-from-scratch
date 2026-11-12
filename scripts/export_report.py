"""校验已完成实验的来源，导出表格、曲线和续写样例。"""
# 导入本模块使用的标准库工具。
import argparse
import math
import unicodedata
from pathlib import Path

# 复用项目内的数据、模型或实验工具。
from minigpt.common import jsonl, read_json, sha256, write_json


# 转义特殊字符并处理控制字符，使生成文本可用于 LaTeX。
def tex(text):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    text = "".join(("[U+FFFD] " if c == "\ufffd" else f"[U+{ord(c):X}] " if ord(c) > 0xFFFF else c) for c in str(text))
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c in "\n\t")
    return "".join(table.get(c, " " if c in "\n\t" else c) for c in text)


# 核对数据、训练完成状态、初始化来源和评测样本的一致性。
def validate_inputs(root, data):
    root, data = Path(root), Path(data)
    manifest = read_json(data / "manifest.json")
    if not manifest["source"].startswith("https://") or "synthetic" in manifest["source"].lower():
        raise ValueError("Synthetic fixture results cannot be exported as formal experiments")
    manifest_hash = sha256(data / "manifest.json")
    for split in ("train", "val", "test"):
        if sha256(data / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
            raise ValueError(f"Changed partition: {split}")
    infos, statuses, evaluations = {}, {}, {}
    # 核对各训练阶段的完成状态、数据版本和初始化关系。
    for name in ("pretrain", "sft", "scratch"):
        infos[name] = read_json(root / name / "run.json")
        statuses[name] = read_json(root / name / "status.json")
        if not statuses[name]["complete"]:
            raise ValueError(f"{name} is incomplete; finish training before exporting")
        if infos[name]["data_manifest_sha256"] != manifest_hash:
            raise ValueError(f"{name}: data manifest mismatch")
    if infos["sft"]["init_checkpoint_sha256"] != sha256(root / "pretrain/best.pt"):
        raise ValueError("SFT was not initialized from this pretraining result")
    if infos["scratch"]["init_checkpoint"] is not None:
        raise ValueError("Scratch control must start from random weights")
    if infos["scratch"]["config"] != infos["sft"]["config"]:
        raise ValueError("Controlled SFT comparison must use the same config except initialization")
    # 检查最终测试使用的权重、新闻集合和指标定义。
    for name in ("lead", "pretrain", "sft", "scratch"):
        result = read_json(root / f"eval_test_{name}/metrics.json")
        evaluations[name] = result
        if result["split"] != "test" or result["data_manifest_sha256"] != manifest_hash:
            raise ValueError(f"{name}: expected final test evaluation on this data")
        if name != "lead" and result["checkpoint_sha256"] != sha256(root / name / "best.pt"):
            raise ValueError(f"{name}: evaluation checkpoint mismatch")
    if len({r["selection_hash"] for r in evaluations.values()}) != 1:
        raise ValueError("Evaluations used different examples")
    if len({r["metric"] for r in evaluations.values()}) != 1:
        raise ValueError("Incompatible scoring conventions")
    # 续写对照也要与相同数据和预训练检查点对应。
    continuation = read_json(root / "continuations.json")
    if continuation["data_manifest_sha256"] != manifest_hash:
        raise ValueError("Continuation data mismatch")
    for key, path in [("before", "initial.pt"), ("after", "best.pt")]:
        if continuation["checkpoints"][key] != sha256(root / "pretrain" / path):
            raise ValueError("Continuation checkpoint mismatch")
    return manifest, infos, statuses, evaluations, continuation


# 从通过校验的实验记录生成图表、表格和汇总文件。
def export(root, data, output):
    root, data, out = Path(root), Path(data), Path(output)
    manifest, infos, statuses, evaluations, continuation = validate_inputs(root, data)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 读取真实训练日志，分别绘制预训练和两组微调损失。
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    logs = {}
    # 三组实验分别绘制训练与验证损失，使用各自的日志数据。
    for ax, name, title in zip(axes, ("pretrain", "sft", "scratch"),
                                ("Pretraining", "SFT after pretraining", "SFT from random initialization")):
        rows = list(jsonl(root / name / "metrics.jsonl"))
        logs[name] = rows
        for event, label, style in [("train", "Train (sampled steps)", "-"), ("val", "Validation", "o-")]:
            selected = [r for r in rows if r["event"] == event]
            ax.plot([r["step"] for r in selected], [r["loss"] for r in selected], style, label=label, linewidth=1.2, markersize=3)
        ax.set(xlabel="Optimizer steps", ylabel="Cross entropy (nats / target token)", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(out / "loss_curves.png", dpi=180)
    plt.close(fig)
    # 从运行记录提取实际模型与环境信息，生成对应表格。
    mc = infos["pretrain"]["actual_model_config"]
    lines = [r"% Generated from validated real experiment artifacts; do not edit numeric results manually.",
             r"\subsection{实际数据、环境与模型}",
             "本次数据划分与模型信息来自保存的实验清单。训练、验证、测试新闻数分别为"
             f"{manifest['counts']['train']:,}、{manifest['counts']['val']:,}、{manifest['counts']['test']:,}。"
             f"实际词表大小为 {manifest['vocab_size']:,}，模型参数量为 {infos['pretrain']['parameters']:,}。",
             f"模型使用 {mc['n_layer']} 层、{mc['n_head']} 个注意力头、{mc['n_embd']} 维隐藏表示，"
             f"上下文长度为 {mc['block_size']}。",
             "GPU：" + tex(infos["pretrain"]["environment"]["gpu"]) + "；PyTorch：" +
             tex(infos["pretrain"]["environment"]["torch"]) + "；实际计算精度：" + tex(infos["pretrain"]["actual_precision"]) + "。",
             r"\subsection{训练过程}", r"\begin{center}\small",
             r"\begin{tabular}{lrrr}\toprule 阶段 & 完成步数 & 输入 token & 训练计算秒数 \\ \midrule"]
    names = {"pretrain": "中文预训练", "sft": "预训练后微调", "scratch": "随机初始化微调"}
    for stage_name, label in names.items():
        config = infos[stage_name].get("config", {}).get("train")
        if config:
            lines.insert(-3, label + f"的微批量为 {config['batch_size']}，梯度累积 {config['grad_accum']} 次，"
                         + f"有效批量为 {config['batch_size'] * config['grad_accum']}；"
                         + f"最大学习率为 {config['learning_rate']:.6f}，预热 {config['warmup_steps']} 步。")
    for name in names:
        st = statuses[name]
        lines.append(f"{names[name]} & {st['step']:,} & {st['seen_input_tokens']:,} & {st['train_seconds']:.1f}" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{center}",
              "上表秒数只累计训练迭代，包含训练取样与设备传输，不含验证、检查点写入和环境准备；不是云平台计费总时长。输入 token 含微调的右侧填充，监督目标 token 数另见实验 JSON。",
              r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{generated/loss_curves.png}",
              r"\caption{各训练阶段的实际损失曲线。预训练与微调的监督范围不同，不直接比较两者的损失高低。}\end{figure}"]
    vals = [r for r in logs["pretrain"] if r["event"] == "val"]
    best = min(vals, key=lambda r: r["loss"])
    lines.append(f"预训练验证集初始交叉熵为 {vals[0]['loss']:.4f}，最佳验证交叉熵为 {best['loss']:.4f}，"
                 f"对应困惑度为 {best['ppl']:.2f}，最佳检查点位于第 {best['step']} 步。")
    lines += [r"\subsection{预训练前后的中文续写}",
              "以下使用预先固定的提示词，随机种子为 42，温度为 0.8，top-k 为 40，最多生成 64 个 token。无效字节的解码替换符和非BMP字符以 Unicode 编号显示，原始输出保存在 JSON。"]
    for before, after in zip(continuation["samples"]["before"], continuation["samples"]["after"]):
        lines += [r"\paragraph{提示词：" + tex(before["prompt"]) + "}",
                  r"\textbf{随机初始化：}" + tex(before["continuation"]) + r"\par",
                  r"\textbf{预训练后：}" + tex(after["continuation"]) + r"\par"]
    lines += [r"\subsection{标题生成的测试集评测}",
              f"四组方法在同一固定测试子集上评测，共 {evaluations['lead']['n']} 条。数值为字符级 ROUGE F1 乘以 100，按样本宏平均。",
              r"\begin{center}\begin{tabular}{lrrr}\toprule 方法 & ROUGE-1 & ROUGE-2 & ROUGE-L \\ \midrule"]
    eval_names = {"lead": "首句前32字符", "pretrain": "仅预训练", "sft": "预训练后微调", "scratch": "随机初始化微调"}
    for name, label in eval_names.items():
        m = evaluations[name]["metrics"]
        lines.append(f"{label} & {100*m['rouge1']:.2f} & {100*m['rouge2']:.2f} & {100*m['rougeL']:.2f}" + r" \\")
    delta = 100 * (evaluations["sft"]["metrics"]["rougeL"] - evaluations["scratch"]["metrics"]["rougeL"])
    lines += [r"\bottomrule\end{tabular}\end{center}",
              f"在本次固定配置下，预训练后微调相对随机初始化微调的 ROUGE-L 差值为 {delta:+.2f} 个百分点。"
              "该结果只代表当前数据、训练预算和随机种子的比较；单次实验不能说明统计显著性。",
              r"\subsection{固定样例与错误分析材料}",
              "以下为测试子集按文档哈希排序后的前四条，未按模型表现挑选。应结合正文检查事实、实体与数字，不能仅依赖词面重合指标。"]
    examples = list(jsonl(root / "eval_test_sft/predictions.jsonl"))[:4]
    for i, row in enumerate(examples, 1):
        lines += [r"\paragraph{样例 " + str(i) + "}",
                  r"\textbf{正文节选：}" + tex(row["content"][:200]) + r"\par",
                  r"\textbf{参考标题：}" + tex(row["reference"]) + r"\par",
                  r"\textbf{生成标题：}" + tex(row["prediction"]) + r"\par"]
    (out / "results.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "experiment_summary.json", {"data": manifest, "runs": infos, "status": statuses,
                                                "test_evaluations": evaluations, "continuations": continuation})
    print(f"Exported real experiment assets to {out}; review analysis and compile main.tex in Overleaf.")


# 解析命令行参数后，校验已完成实验的来源，导出表格、曲线和续写样例。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/cloud")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--output", default="report/generated")
    args = ap.parse_args()
    export(args.runs, args.data, args.output)
    report_dir = Path(args.output).parent
    if (report_dir / "main.tex").exists():
        from scripts.finalize_report import finalize
        finalize(args.runs, args.data, report_dir)


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
