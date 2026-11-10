import pytest

from minigpt.common import sha256, write_json
from minigpt.train import train
from scripts.export_report import tex, validate_inputs


def test_modified_training_tokens_rejected_before_model_construction(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "train.bin").write_bytes(b"changed")
    write_json(tmp_path / "manifest.json", {
        "tokenizer_sha256": sha256(tmp_path / "tokenizer.json"),
        "files": {"train.bin": "0" * 64}})
    cfg = {"stage": "pretrain", "seed": 42, "train": {
        "max_steps": 1, "batch_size": 1, "grad_accum": 1, "eval_every": 1,
        "eval_batches": 1, "log_every": 1, "precision": "fp32"}}
    with pytest.raises(ValueError, match="Data checksum mismatch"):
        train(cfg, tmp_path, tmp_path / "run", device="cpu")


def test_synthetic_runs_cannot_be_exported_as_real_experiments(tmp_path):
    write_json(tmp_path / "manifest.json", {"source": "synthetic local validation fixture"})
    with pytest.raises(ValueError, match="Synthetic fixture"):
        validate_inputs(tmp_path / "runs", tmp_path)


def test_generated_text_is_escaped_for_latex():
    assert tex("A&B_50%#{}$") == r"A\&B\_50\%\#\{\}\$"
