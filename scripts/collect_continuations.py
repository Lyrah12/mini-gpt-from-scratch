"""用相同提示和采样参数记录预训练前后的中文续写。"""
# 导入本模块使用的标准库工具。
import argparse
from pathlib import Path

# 使用 PyTorch 构造相同提示的输入张量。
import torch

# 复用项目内的数据、模型或实验工具。
from minigpt.common import read_json, seed_all, sha256, write_json
from minigpt.evaluate import load_model
from minigpt.tokenizer import BOS, EOS, PAD, load_tokenizer

# 训练前后使用相同提示，便于直接观察续写变化。
PROMPTS = ["人工智能技术的发展", "今天上午，学校举办", "随着城市公共交通的发展"]


# 用同一组提示分别加载训练前后权重，保存可对照的续写。
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
    # 分别加载随机初始化与最佳预训练检查点。
    for name, path in [("before", root / "pretrain/initial.pt"), ("after", root / "pretrain/best.pt")]:
        model, _ = load_model(path, tok_path, device)
        result["inference_precision"] = str(next(model.parameters()).dtype)
        samples = []
        # 每个提示重新设置相同种子，使训练前后采样设置一致。
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


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
