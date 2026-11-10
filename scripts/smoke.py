"""Small CPU end-to-end validation. Synthetic fixture results are NOT homework experiments."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="work/smoke")
    args = ap.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose a fresh smoke output directory")
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "synthetic.jsonl"
    cities = ["北京", "上海", "杭州", "成都", "武汉", "南京", "广州", "深圳"]
    topics = [("科技", "科技展览", "展出人工智能设备"), ("体育", "体育比赛", "参加校园足球活动"),
              ("文化", "文化交流", "展示传统艺术作品"), ("教育", "教育讲座", "学习科学实验方法")]
    with raw.open("w", encoding="utf-8") as f:
        first = None
        for i in range(800):
            city, (_, event, action) = cities[i % 8], topics[(i // 8) % 4]
            body = (f"第{i}号消息，{city}第{i + 1}学校今天举办{event}。学生们在老师带领下{action}。"
                    "负责人介绍，活动通过现场展示和互动体验帮助学生掌握相关知识，吸引了许多家长和居民参加。"
                    "现场还设置了交流区域，参与者分享学习心得，并就今后的活动提出建议。"
                    "学校表示将继续组织类似活动，丰富学生课余生活，推动理论学习与实践体验相结合。")
            row = {"title": f"{city}学校举办{event}", "content": body}
            if first is None:
                first = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps(first, ensure_ascii=False) + "\n")
        f.write(json.dumps({**first, "content": first["content"].replace("。", "！")}, ensure_ascii=False) + "\n")

    def run(*items):
        subprocess.run([sys.executable, "-X", "utf8", *map(str, items)], check=True)

    data = root / "data"
    run("-m", "minigpt.prepare", "--input", raw, "--output", data, "--max-records", "800",
        "--vocab-size", "768", "--block-size", "256", "--source", "synthetic local validation fixture; not real news")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["duplicate_body_or_lead"] == 2
    partitions = [{json.loads(line)["id"] for line in (data / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()}
                  for s in ("train", "val", "test")]
    assert all(not partitions[i] & partitions[j] for i in range(3) for j in range(i))
    cfg = {"stage": "pretrain", "seed": 42,
           "model": {"vocab_size": 768, "block_size": 256, "n_layer": 2, "n_head": 4,
                     "n_embd": 64, "dropout": 0.1, "attention": "sdpa"},
           "train": {"max_steps": 8, "batch_size": 2, "grad_accum": 2, "learning_rate": 0.001,
                     "min_lr_ratio": 0.1, "warmup_steps": 2, "weight_decay": 0.01, "grad_clip": 1.,
                     "eval_every": 4, "eval_batches": 2, "log_every": 1, "precision": "fp32"}}
    pc = root / "pretrain.json"
    pc.write_text(json.dumps(cfg), encoding="utf-8")
    run("-m", "minigpt.train", "--config", pc, "--data", data, "--output", root / "pretrain",
        "--device", "cpu", "--threads", "1", "--deterministic")
    run("-m", "minigpt.train", "--config", pc, "--data", data, "--output", root / "resumed",
        "--device", "cpu", "--threads", "1", "--deterministic", "--stop-after", "3")
    run("-m", "minigpt.train", "--config", pc, "--data", data, "--output", root / "resumed",
        "--device", "cpu", "--threads", "1", "--deterministic", "--resume", root / "resumed/last.pt")
    import torch
    a = torch.load(root / "pretrain/last.pt", map_location="cpu", weights_only=False)
    b = torch.load(root / "resumed/last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "Resume changed weights"
    assert torch.equal(a["batch_rng"], b["batch_rng"])
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            other = b["optimizer"]["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    changed = json.loads(json.dumps(cfg))
    changed["train"]["learning_rate"] *= 2
    bad = root / "changed_config.json"
    bad.write_text(json.dumps(changed), encoding="utf-8")
    rejected = subprocess.run([sys.executable, "-X", "utf8", "-m", "minigpt.train", "--config", str(bad), "--data", str(data),
        "--output", str(root / "resumed"), "--resume", str(root / "resumed/last.pt"), "--device", "cpu",
        "--threads", "1", "--deterministic"], text=True, encoding="utf-8", capture_output=True)
    assert rejected.returncode != 0 and "same config" in rejected.stderr
    cfg["stage"] = "sft"
    cfg["train"]["max_steps"] = 6
    sc = root / "sft.json"
    sc.write_text(json.dumps(cfg), encoding="utf-8")
    run("-m", "minigpt.train", "--config", sc, "--data", data, "--output", root / "sft",
        "--init", root / "pretrain/best.pt", "--device", "cpu", "--threads", "1")
    run("-m", "minigpt.train", "--config", sc, "--data", data, "--output", root / "scratch",
        "--device", "cpu", "--threads", "1")
    hashes = []
    for name, checkpoint in [("lead", None), ("pretrain", root / "pretrain/best.pt"),
                              ("sft", root / "sft/best.pt"), ("scratch", root / "scratch/best.pt")]:
        command = ["-m", "minigpt.evaluate", "--data", data, "--output", root / f"eval_{name}",
                   "--split", "test", "--limit", "3", "--device", "cpu"]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        run(*command)
        hashes.append(json.loads((root / f"eval_{name}/metrics.json").read_text())["selection_hash"])
    assert len(set(hashes)) == 1
    run("-m", "minigpt.generate", "--checkpoint", root / "pretrain/best.pt", "--tokenizer", data / "tokenizer.json",
        "--text", "今天学校举办", "--max-new-tokens", "12", "--device", "cpu")
    run("-m", "scripts.collect_continuations", "--runs", root, "--data", data, "--device", "cpu")
    (root / "validation_summary.json").write_text(json.dumps({"scope": "synthetic CPU validation only",
        "resume_weights_bitwise_equal": True, "resume_rng_equal": True, "evaluation_selection_equal": True,
        "optimizer_state_equal": True, "changed_config_resume_rejected": True,
        "duplicate_filter_checked": True, "partition_ids_disjoint": True,
        "pretrain_steps": 8, "sft_steps": 6, "scratch_sft_steps": 6,
        "formal_experiment_completed": False}, indent=2) + "\n", encoding="utf-8")
    print("PASS: CPU pipeline and exact resume validation. NOT a formal language-quality experiment.")


if __name__ == "__main__":
    main()
