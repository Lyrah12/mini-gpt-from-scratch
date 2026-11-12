"""根据已完成且通过校验的实验记录填写本地报告。"""
# 导入本模块使用的标准库工具。
from pathlib import Path
import re
from collections import Counter

# 复用项目内的数据、模型或实验工具。
from minigpt.common import jsonl, read_json, write_json
from scripts.export_report import validate_inputs, tex


# 依据保存的真实指标填写报告正文和结果摘要。
def finalize(runs, data, report):
    runs, report = Path(runs), Path(report)
    # 填写正文前先复用完整的实验来源校验。
    manifest, infos, statuses, evaluations, continuation = validate_inputs(runs, data)
    generated = report / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    if not (report / "main.tex").exists():
        return
    # 读取初始与最佳验证结果，并从已保存指标计算各组对照。
    pre_vals = [r for r in jsonl(runs / "pretrain/metrics.jsonl") if r["event"] == "val"]
    first, best = pre_vals[0], min(pre_vals, key=lambda r:r["loss"])
    scores = {name: row["metrics"] for name,row in evaluations.items()}
    main_l = scores["sft"]["rougeL"]*100
    lead_l = scores["lead"]["rougeL"]*100
    scratch_l = scores["scratch"]["rougeL"]*100
    raw_l = scores["pretrain"]["rougeL"]*100
    # 使用实际完成步数和实际得分填入结果摘要。
    abstract = (f"本地实际完成 {statuses['pretrain']['step']} 步预训练及两组各 {statuses['sft']['step']} 步监督微调。"
                f"固定验证窗口交叉熵从 {first['loss']:.4f} 降至最佳 {best['loss']:.4f}。"
                f"在 {evaluations['sft']['n']} 条固定测试新闻上，预训练后微调的字符级 ROUGE-L 为 {main_l:.2f}，"
                f"首句抽取基线为 {lead_l:.2f}，随机初始化微调为 {scratch_l:.2f}。"
                "这些结果限定于本次数据、训练预算与单个随机种子，不代表通用中文任务表现。")
    (generated / "abstract_results.tex").write_text(abstract+"\n",encoding="utf-8")
    # 以百分制分数计算两个已完成方法之间的 ROUGE-L 差值。
    def difference(a,b):
        delta=a-b
        return (f"高 {delta:.2f}" if delta>0 else f"低 {-delta:.2f}" if delta<0 else "相同，为 0.00") + " 个百分点"
    lines = [r"\subsection{结果解释与对照分析}",
        f"预训练的固定验证交叉熵由 {first['loss']:.4f} 降至最佳 {best['loss']:.4f}，"
        f"对应困惑度由 {first['ppl']:.2f} 变为 {best['ppl']:.2f}。"
        "这表明模型对同领域未参与训练文本的下一 token 预测有所改善；它不能直接证明事实知识、对话或推理能力。",
        f"标题任务上，预训练后微调的 ROUGE-L 比仅预训练{difference(main_l,raw_l)}，"
        f"比随机初始化微调{difference(main_l,scratch_l)}，比首句抽取基线{difference(main_l,lead_l)}。",
        ("主方法超过了抽取基线，说明在本次测试子集中，生成模型取得了更高的参考标题词面重合。" if main_l>lead_l else
         "主方法未超过抽取基线，说明当前训练配置下，生成模型的词面重合表现仍有限，不能把微调完成等同于任务已经充分学好。"),
        "随机初始化对照保持相同的微调数据、学习率、步数和采样种子，但未做独立最优调参；预训练组还使用了额外预训练计算，不能据此宣称等算力下的普遍优势。",
        r"\subsection{输出问题的可复查筛查}"]
    rows=list(jsonl(runs/"eval_test_sft/predictions.jsonl"))
    from minigpt.metrics import characters
    empty=sum(not characters(r["prediction"]) for r in rows)
    repeated=0;numeric_flags=0
    for row in rows:
        text=row["prediction"]
        grams=[text[i:i+4] for i in range(max(0,len(text)-3))]
        repeated += bool(grams and max(Counter(grams).values())>=3)
        numeric_flags += any(number not in row["content"] for number in re.findall(r"[0-9]+(?:\.[0-9]+)?",text))
    lines.append(f"主方法的 {len(rows)} 条输出中，不含字母或数字的空输出 {empty} 条；同一连续四字符片段至少出现三次的输出 {repeated} 条；"
                 f"含有未直接出现在完整正文中的阿拉伯数字片段的输出 {numeric_flags} 条。"
                 "后两项只是用于定位重复和数字问题的启发式提示，存在表达转换等误报，不能作为事实错误率。")
    lines.append("以下列出 ROUGE-L 最低的两条输出用于复查，选择规则与前面按文档哈希固定展示的样例不同。低词面重合也可能来自合理改写，不能自动等同于语义错误。")
    for i,row in enumerate(sorted(rows,key=lambda r:(r["scores"]["rougeL"],r["id"]))[:2],1):
        lines += [r"\paragraph{低重合样例 "+str(i)+"}",
                  r"\textbf{参考标题：}"+tex(row["reference"])+r"\par",
                  r"\textbf{生成标题：}"+tex(row["prediction"])+r"\par",
                  f"字符级 ROUGE-L 为 {100*row['scores']['rougeL']:.2f}。应结合原始逐条预测文件中的正文检查遗漏、实体、数字及重复。"]
    lines += [r"\subsection{本地计算与复现评价}"]
    total_seconds=sum(st["train_seconds"] for st in statuses.values())
    lines.append(f"三组训练迭代累计计算时间为 {total_seconds/60:.1f} 分钟，不含验证、保存、依赖安装与生成评测。"
                 "本次以实际 GPU 试跑选择微批量并保持有效批量，说明这一级别的小型模型可以在本机单卡执行。"
                 "显存、吞吐与完成时间仅对本次配置和运行环境成立。")
    lines.append("本实验的结论范围仍受新闻领域、512-token 前缀截断、启发式去重、固定抽样验证和单随机种子限制。后续应使用验证集比较更多训练预算或数据策略，不能根据本次测试分数反复调参后仍把它称为独立最终测试。")
    (generated/"analysis.tex").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    write_json(generated/"report_status.json",{"real_experiments_complete":True,"pdf_compiled_locally":False,
        "name_and_student_id":"left as placeholders at user's request","source":manifest["source"],
        "test_examples":len(rows),"sft_rougeL_percent":main_l,"lead_rougeL_percent":lead_l,
        "scratch_rougeL_percent":scratch_l,"empty_outputs":empty,"repetition_flags":repeated,
        "numeric_screen_flags":numeric_flags})
    (report/"FINAL_STATUS.md").write_text("正式本地实验已完成，报告已填入经过来源校验的实际结果与分析。\n\n姓名与学号按用户要求保留占位。未在本地编译 PDF；请上传完整 report/ 文件夹到 Overleaf，使用 XeLaTeX 编译并检查排版。\n",encoding="utf-8")
    readme=report/"README.md"
    if readme.exists():
        text=readme.read_text(encoding="utf-8")
        text=text.replace("当前是实验过程稿，正在本机 RTX 4060 上运行正式实验，不是可直接提交的完成稿。", "正式本地实验已完成，真实结果和分析已填入。姓名和学号按要求保留占位，提交前请在 Overleaf 编译并检查排版。")
        text=text.replace("`generated/results.tex` 明确说明正式本地实验仍在进行。", "`generated/results.tex` 汇总真实实验，`generated/analysis.tex` 给出实际比较与问题筛查。")
        readme.write_text(text,encoding="utf-8")
    print("Report narrative finalized from real experiment records; no local PDF compilation.",flush=True)
