"""Generate Chinese continuations or news titles from a trusted local checkpoint."""
import argparse

import torch

from .common import seed_all
from .evaluate import load_model
from .tokenizer import BOS, EOS, PAD, TITLE_TOKENS, load_tokenizer, title_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--text", required=True)
    ap.add_argument("--task", choices=["continue", "title"], default="continue")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_model(args.checkpoint, args.tokenizer, device)
    tok = load_tokenizer(args.tokenizer)
    if args.task == "title":
        ids = title_prompt(tok, args.text, model.config.block_size)
        count, temperature = TITLE_TOKENS, 0.0
    else:
        if not 1 <= args.max_new_tokens < model.config.block_size:
            raise ValueError("max_new_tokens must be between 1 and block_size - 1")
        ids = [BOS] + tok.encode(args.text).ids
        ids = ids[:model.config.block_size - args.max_new_tokens]
        count, temperature = args.max_new_tokens, args.temperature
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, count, temperature, eos_id=EOS, forbidden_ids=(PAD, BOS))
    print(tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
