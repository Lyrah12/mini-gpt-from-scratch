"""Optional deterministic title decoding; original GPT implementation stays frozen."""
import argparse
from collections import Counter

import torch

from minigpt.evaluate import load_model
from minigpt.tokenizer import BOS, EOS, PAD, TITLE_TOKENS, load_tokenizer, title_prompt

MODES = {'greedy': {'repetition_penalty': 1., 'no_repeat_ngram': 0},
         'penalty_1p1': {'repetition_penalty': 1.1, 'no_repeat_ngram': 0},
         'no_repeat_3': {'repetition_penalty': 1., 'no_repeat_ngram': 3}}


def constrain(logits, generated, repetition_penalty=1., no_repeat_ngram=0):
    """约束已生成的标题 token，保留输入新闻中的实体复制能力。"""
    if repetition_penalty < 1 or no_repeat_ngram < 0:
        raise ValueError('Invalid decoding constraints')
    result = logits.clone()
    if generated and repetition_penalty != 1:
        seen = list(set(generated))
        values = result[..., seen]
        # 负 logit 乘以惩罚系数、正 logit 除以系数，两者都会降低重复候选得分。
        result[..., seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    n = no_repeat_ngram
    if n and len(generated) >= n - 1:
        prefix = tuple(generated[-(n-1):]) if n > 1 else ()
        # 找到与当前后缀匹配的历史前缀，禁止补出曾出现过的完整 n-gram。
        banned = {generated[i+n-1] for i in range(len(generated)-n+1)
                  if tuple(generated[i:i+n-1]) == prefix}
        if banned:
            result[..., list(banned)] = -float('inf')
    result[..., [PAD, BOS]] = -float('inf')
    return result


@torch.no_grad()
def decode(model, prompt, mode='greedy', max_new_tokens=TITLE_TOKENS):
    settings = MODES[mode]
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.config.block_size:])[:, -1, :].float()
            logits = constrain(logits, generated, **settings)
            nxt = logits.argmax(dim=-1, keepdim=True)
            generated.append(nxt.item())
            ids = torch.cat((ids, nxt), dim=1)
            if generated[-1] == EOS:
                break
    finally:
        model.train(was_training)
    return generated


def repetition_flag(text):
    grams = Counter(text[i:i+4] for i in range(len(text)-3))
    return bool(grams and max(grams.values()) >= 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', default='data/processed_xlsum/tokenizer.json')
    ap.add_argument('--text', required=True)
    ap.add_argument('--lead', default='', help='Original article lead, if available; never the reference headline')
    ap.add_argument('--mode', choices=MODES, default='greedy')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(a.checkpoint, a.tokenizer, a.device)
    tok = load_tokenizer(a.tokenizer)
    from scripts.prepare_refinement import combine_article
    body = combine_article(a.lead, a.text)
    print(tok.decode(decode(model, title_prompt(tok, body, model.config.block_size), a.mode), skip_special_tokens=True))


if __name__ == '__main__':
    main()
