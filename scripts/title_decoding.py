"""为标题生成提供重复惩罚与重复 token 片段限制。"""
# 导入本模块使用的标准库工具。
import argparse
from collections import Counter

# 使用 PyTorch 调整候选 token 得分并拼接生成序列。
import torch

# 复用项目内的数据、模型或实验工具。
from minigpt.evaluate import load_model
from minigpt.tokenizer import BOS, EOS, PAD, TITLE_TOKENS, load_tokenizer, title_prompt

# 声明参与比较的解码方式及各自的重复约束参数。
MODES = {'greedy': {'repetition_penalty': 1., 'no_repeat_ngram': 0},
         'penalty_1p1': {'repetition_penalty': 1.1, 'no_repeat_ngram': 0},
         'no_repeat_3': {'repetition_penalty': 1., 'no_repeat_ngram': 3}}


# 只调整已生成标题的重复候选，并屏蔽 PAD 和 BOS。
def constrain(logits, generated, repetition_penalty=1., no_repeat_ngram=0):
    """约束已生成的标题 token，保留输入新闻中的实体复制能力。"""
    if repetition_penalty < 1 or no_repeat_ngram < 0:
        raise ValueError('Invalid decoding constraints')
    result = logits.clone()
    # 只惩罚已生成标题中的 token，新闻输入中的实体仍可复制。
    if generated and repetition_penalty != 1:
        seen = list(set(generated))
        values = result[..., seen]
        # 负 logit 乘以惩罚系数、正 logit 除以系数，两者都会降低重复候选得分。
        result[..., seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    n = no_repeat_ngram
    # 找到与当前后缀匹配的历史片段，禁止补出重复的完整 n-gram。
    if n and len(generated) >= n - 1:
        prefix = tuple(generated[-(n-1):]) if n > 1 else ()
        banned = {generated[i+n-1] for i in range(len(generated)-n+1)
                  if tuple(generated[i:i+n-1]) == prefix}
        if banned:
            result[..., list(banned)] = -float('inf')
    result[..., [PAD, BOS]] = -float('inf')
    return result


# 逐 token 执行指定解码约束，结束后恢复原训练模式。
@torch.no_grad()
def decode(model, prompt, mode='greedy', max_new_tokens=TITLE_TOKENS):
    settings = MODES[mode]
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = []
    # 保存当前模式，整个逐 token 生成阶段使用评估模式。
    was_training = model.training
    model.eval()
    # 解码结束后通过 finally 恢复原模式。
    try:
        # 每次读取末位置得分，应用重复限制后选择下一个 token。
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.config.block_size:])[:, -1, :].float()
            logits = constrain(logits, generated, **settings)
            nxt = logits.argmax(dim=-1, keepdim=True)
            # 保留实际生成的 EOS，用于结束判断和长度统计。
            generated.append(nxt.item())
            ids = torch.cat((ids, nxt), dim=1)
            if generated[-1] == EOS:
                break
    finally:
        model.train(was_training)
    return generated


# 按原始输出统计四字符片段，筛出至少重复三次的标题。
def repetition_flag(text):
    grams = Counter(text[i:i+4] for i in range(len(text)-3))
    return bool(grams and max(grams.values()) >= 3)


# 解析命令行参数后，为标题生成提供重复惩罚与重复 token 片段限制。
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


# 直接运行模块时，从命令行入口启动。
if __name__ == '__main__':
    main()
