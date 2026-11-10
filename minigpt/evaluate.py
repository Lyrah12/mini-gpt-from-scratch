"""Fixed validation/test subsets; greedy model titles plus extractive lead baseline."""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

from .common import environment, jsonl, object_hash, read_json, seed_all, sha256, write_json
from .metrics import characters, rouge
from .model import GPT, ModelConfig
from .tokenizer import BOS, EOS, PAD, TITLE_TOKENS, load_tokenizer, title_prompt
from .train import load_checkpoint


def load_model(checkpoint, tokenizer_path, device):
    state = load_checkpoint(checkpoint)
    if sha256(tokenizer_path) != state["tokenizer_sha256"]:
        raise ValueError("Checkpoint/tokenizer mismatch")
    model = GPT(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    return model.to(device=device, dtype=dtype).eval(), state


def evaluate(data, output, checkpoint=None, split="val", limit=200, seed=42, device="auto", threads=4):
    if split not in ("val", "test") or limit <= 0:
        raise ValueError("Use val/test and a positive limit")
    torch.set_num_threads(threads)
    seed_all(seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    data, output = Path(data), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {output}")
    manifest = read_json(data / "manifest.json")
    if sha256(data / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
        raise ValueError("Evaluation partition checksum mismatch")
    if sha256(data / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    # Bounded-memory reservoir with a separate RNG, identical for every checkpoint.
    rng, rows, n = random.Random(seed), [], 0
    for n, row in enumerate(jsonl(data / f"{split}.jsonl"), 1):
        if len(rows) < limit:
            rows.append(row)
        else:
            j = rng.randrange(n)
            if j < limit:
                rows[j] = row
    rows.sort(key=lambda r: r["id"])
    if not rows:
        raise ValueError("Empty evaluation partition")
    tok = load_tokenizer(data / "tokenizer.json")
    if checkpoint:
        model, state = load_model(checkpoint, data / "tokenizer.json", device)
        block_size = model.config.block_size
    else:
        model, state, block_size = None, None, manifest["block_size"]
    output.mkdir(parents=True, exist_ok=True)
    sums = {"rouge1": 0., "rouge2": 0., "rougeL": 0.}
    empty = 0
    started = time.perf_counter()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            prompt = title_prompt(tok, row["content"], block_size)
            if model is not None:
                x = torch.tensor([prompt], dtype=torch.long, device=device)
                generated = model.generate(x, max_new_tokens=TITLE_TOKENS, temperature=0,
                                           eos_id=EOS, forbidden_ids=(PAD, BOS))
                prediction = tok.decode(generated[0, len(prompt):].tolist(), skip_special_tokens=True).strip()
            else:
                # Same visible source prefix as the neural model, excluding instruction text.
                from .tokenizer import PREFIX, SUFFIX
                begin = 1 + len(tok.encode(PREFIX).ids)
                end = len(prompt) - len(tok.encode(SUFFIX).ids)
                visible = tok.decode(prompt[begin:end])
                prediction = re.split(r"[。！？!?]", visible, maxsplit=1)[0][:32]
            scores = rouge(prediction, row["title"])
            for key in sums:
                sums[key] += scores[key]
            empty += not bool(characters(prediction))
            f.write(json.dumps({"id": row["id"], "content": row["content"], "reference": row["title"],
                                "prediction": prediction, "scores": scores}, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"Evaluated {i}/{len(rows)}", flush=True)
    result = {"split": split, "n": len(rows), "available_documents": n, "requested_limit": limit,
              "seed": seed, "selection_hash": object_hash([r["id"] for r in rows]),
              "evaluation_data_sha256": manifest["files"][f"{split}.jsonl"],
              "data_manifest_sha256": sha256(data / "manifest.json"),
              "checkpoint": str(checkpoint) if checkpoint else None,
              "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
              "checkpoint_step": state["step"] if state else None,
              "method": "greedy_model" if checkpoint else "lead32",
              "inference_precision": str(next(model.parameters()).dtype) if model is not None else None,
              "metric": "macro character ROUGE F1; Unicode L/N, lowercase, punctuation/whitespace removed",
              "decoding": {"temperature": 0, "max_new_tokens": TITLE_TOKENS, "block_size": block_size},
              "metrics": {k: v / len(rows) for k, v in sums.items()},
              "empty_rate": empty / len(rows), "seconds": time.perf_counter() - started,
              "environment": environment()}
    write_json(output / "metrics.json", result)
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--checkpoint", help="Omit for the lead-32 extractive baseline")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    evaluate(args.data, args.output, args.checkpoint, args.split, args.limit, args.seed, args.device)


if __name__ == "__main__":
    main()
