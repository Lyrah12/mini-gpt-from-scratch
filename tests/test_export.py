import json

from minigpt.common import write_json
from scripts import export_report


def test_export_writes_plot_and_escapes_realistic_text(tmp_path, monkeypatch):
    """Exercise presentation of dummy test data; never bypass provenance in the actual CLI."""
    root, output = tmp_path / "dummy_runs", tmp_path / "dummy_assets"
    mc = {"n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 256}
    info = {"actual_model_config": mc, "parameters": 123456, "actual_precision": "fp32",
            "environment": {"gpu": "TEST_ONLY", "torch": "2.6.0"}}
    infos = {s: info for s in ("pretrain", "sft", "scratch")}
    statuses = {s: {"step": 1, "seen_input_tokens": 1024, "train_seconds": 1.2} for s in infos}
    manifest = {"counts": {"train": 100, "val": 5, "test": 5}, "vocab_size": 768}
    result = {"n": 3, "metrics": {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3}}
    evaluations = {s: result for s in ("lead", "pretrain", "sft", "scratch")}
    continuations = {"samples": {s: [{"prompt": "测试_提示", "continuation": "测试&内容50%"}]
                                    for s in ("before", "after")}}
    # Only this unit test substitutes the provenance validator; the CLI has no bypass option.
    monkeypatch.setattr(export_report, "validate_inputs", lambda *_: (manifest, infos, statuses, evaluations, continuations))
    for name in infos:
        (root / name).mkdir(parents=True)
        rows = [{"event": "val", "step": 0, "loss": 3., "ppl": 20.},
                {"event": "train", "step": 1, "loss": 2.}, {"event": "val", "step": 1, "loss": 2., "ppl": 7.}]
        (root / name / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "eval_test_sft").mkdir()
    (root / "eval_test_sft/predictions.jsonl").write_text(json.dumps({
        "content": "测试正文", "reference": "参考标题", "prediction": "生成_标题&50%"}, ensure_ascii=False) + "\n", encoding="utf-8")
    export_report.export(root, tmp_path / "unused_data", output)
    text = (output / "results.tex").read_text(encoding="utf-8")
    assert r"生成\_标题\&50\%" in text
    assert "40.00 & 20.00 & 30.00" in text
    from PIL import Image
    with Image.open(output / "loss_curves.png") as img:
        assert img.width >= 2000 and img.height >= 600
    assert (output / "experiment_summary.json").exists()
