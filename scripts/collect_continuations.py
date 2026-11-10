"""Record fixed prompts before and after pretraining, with matched sampling settings."""
import argparse
from pathlib import Path

import torch

from minigpt.common import read_json, seed_all, sha256, write_json
from minigpt.evaluate import load_model
from minigpt.tokenizer import BOS, EOS, PAD, load_tokenizer

PROMPTS = ["人工智能技术的发展", "今天上午，学校举办", "随着城市公共交通的发展"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/cloud")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    root, data = Path(args.runs), Path(args.data)
    output = root / "continuations.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tok_path = data / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    result = {"seed": 42, "temperature": 0.8, "top_k": 40, "max_new_tokens": 64,
              "data_manifest_sha256": sha256(data / "manifest.json"), "samples": {}, "checkpoints": {}}
    for name, path in [("before", root / "pretrain/initial.pt"), ("after", root / "pretrain/best.pt")]:
        model, _ = load_model(path, tok_path, device)
        result["inference_precision"] = str(next(model.parameters()).dtype)
        samples = []
        for prompt in PROMPTS:
            seed_all(42)
            ids = [BOS] + tok.encode(prompt).ids
            x = torch.tensor([ids], device=device)
            y = model.generate(x, max_new_tokens=64, temperature=0.8, top_k=40, eos_id=EOS, forbidden_ids=(PAD, BOS))
            samples.append({"prompt": prompt, "continuation": tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True)})
        result["samples"][name] = samples
        result["checkpoints"][name] = sha256(path)
        del model
    write_json(output, result)


if __name__ == "__main__":
    main()
