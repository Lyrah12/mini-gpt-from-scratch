"""检查报告填写是否使用实测结果并保留身份设置。"""
# 导入本模块使用的标准库工具。
import json
# 复用项目内的数据、模型或实验工具。
from minigpt.common import write_json
from scripts import finalize_report


# 核对报告使用传入的指标，并保留原有身份信息。
def test_final_report_uses_measured_scores_and_preserves_identity_placeholders(tmp_path,monkeypatch):
    # 在临时目录创建最小报告模板和身份信息。
    report=tmp_path/"report";report.mkdir()
    (report/"main.tex").write_text("test skeleton",encoding="utf-8")
    metadata="姓名待填写，学号待填写"
    (report/"metadata.tex").write_text(metadata,encoding="utf-8")
    runs=tmp_path/"runs";(runs/"pretrain").mkdir(parents=True)
    (runs/"pretrain/metrics.jsonl").write_text(json.dumps({"event":"val","loss":4.0,"ppl":54.6})+"\n"+json.dumps({"event":"val","loss":2.0,"ppl":7.4})+"\n",encoding="utf-8")
    folder=runs/"eval_test_sft";folder.mkdir()
    row={"id":"dummy","content":"学校举办科技展览。","reference":"学校科技展","prediction":"学校举办展览","scores":{"rougeL":0.4}}
    (folder/"predictions.jsonl").write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
    # 为各组对照设置已知分数，用于检查报告中的结果引用。
    evaluations={name:{"n":1,"metrics":{"rougeL":value}} for name,value in [("sft",0.4),("lead",0.3),("scratch",0.1),("pretrain",0.2)]}
    status={name:{"step":steps,"train_seconds":60.0,"complete":True} for name,steps in [("pretrain",12),("sft",8),("scratch",8)]}
    monkeypatch.setattr(finalize_report,"validate_inputs",lambda *_:({"source":"https://example.test/unit-fixture"},{},status,evaluations,{}))
    # 调用填写逻辑，并核对报告数值、身份设置和结果状态。
    finalize_report.finalize(runs,tmp_path/"unused",report)
    abstract=(report/"generated/abstract_results.tex").read_text(encoding="utf-8")
    assert "12 步预训练" in abstract and "40.00" in abstract and "30.00" in abstract
    assert (report/"metadata.tex").read_text(encoding="utf-8")==metadata
    result=json.loads((report/"generated/report_status.json").read_text(encoding="utf-8"))
    assert result["sft_rougeL_percent"]==40.0 and not result["pdf_compiled_locally"]
