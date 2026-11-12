"""执行预训练和监督微调，按有效 token 累积梯度并保存训练状态。"""
# 导入本模块使用的标准库工具。
import argparse
import contextlib
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

# 导入数组、随机状态和训练损失计算所需的数值库。
import numpy as np
import torch
from torch.nn import functional as F

# 复用项目内的数据、模型或实验工具。
from .common import environment, jsonl, object_hash, read_json, seed_all, sha256, source_fingerprint, write_json
from .data import Batches
from .model import GPT, ModelConfig
from .tokenizer import load_tokenizer


# 先写临时文件，再原子替换检查点。
def atomic_save(state, path):
    """先完整写入临时文件，再替换检查点，避免留下半写入文件。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


# 先线性预热学习率，再按完整训练计划进行余弦衰减。
def learning_rate(step, cfg):
    # 学习率步数使用优化器更新次数，梯度累积中的微批次共用同一步长。
    warmup, total = cfg["warmup_steps"], cfg["max_steps"]
    if step < warmup:
        return cfg["learning_rate"] * (step + 1) / max(1, warmup)
    ratio = (step - warmup) / max(1, total - warmup - 1)
    return cfg["learning_rate"] * (cfg["min_lr_ratio"] +
            (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(1, ratio))))


# 在 BF16 训练中启用 CUDA 自动混合精度。
def amp_context(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()


# 使用独立随机状态估计验证损失，结束后恢复模型模式。
@torch.no_grad()
def validation(model, batches, eval_batches, seed, device, precision):
    # 独立且固定的验证 RNG，使检查点之间使用相同样本，也避免消耗训练采样状态。
    generator = torch.Generator().manual_seed(seed + 10000)
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    # 验证结束时通过 finally 恢复调用前的训练模式。
    try:
        # 逐批累加损失总和与有效标签数，再统一求平均。
        for _ in range(eval_batches):
            x, y = batches.sample("val", generator)
            x, y = x.to(device), y.to(device)
            with amp_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            total_loss += loss.item()
            total_tokens += (y != -100).sum().item()
    finally:
        model.train(was_training)
    return total_loss / total_tokens, total_tokens


# 加载含模型、优化器和随机状态的本地检查点。
def load_checkpoint(path):
    # 优化器和随机状态通过 pickle 恢复，因此加载对象应为本地或可信检查点。
    return torch.load(path, map_location="cpu", weights_only=False)


# 核对配置与数据，执行训练更新、周期验证和状态保存。
def train(config, data_dir, output, device="auto", init=None, resume=None, stop_after=None, threads=4,
          deterministic=False, stop_file=None):
    cfg = read_json(config) if isinstance(config, (str, Path)) else config
    tc = cfg["train"]
    # 在创建模型前校验训练步数、批量和日志间隔。
    for key in ("max_steps", "batch_size", "grad_accum", "eval_every", "eval_batches", "log_every"):
        if tc[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["stage"] not in ("pretrain", "sft") or tc["precision"] not in ("bf16", "fp32"):
        raise ValueError("Invalid stage/precision")
    if init and resume:
        raise ValueError("Use init OR resume")
    if init and cfg["stage"] != "sft":
        raise ValueError("Pretraining starts from random weights")
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    # 固定线程和随机状态，并按硬件能力选择计算精度。
    torch.set_num_threads(threads)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    seed_all(cfg["seed"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    precision = "bf16" if device == "cuda" and tc["precision"] == "bf16" and torch.cuda.is_bf16_supported() else "fp32"
    data_dir, out = Path(data_dir), Path(output)
    # 核对分词器和训练数组，避免混用不同版本的处理数据。
    manifest = read_json(data_dir / "manifest.json")
    tokenizer_path = data_dir / "tokenizer.json"
    tok_hash = sha256(tokenizer_path)
    if tok_hash != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from data manifest")
    required = ([f"{s}.bin" for s in ("train", "val")] if cfg["stage"] == "pretrain" else
                [f"sft_{s}_{kind}.npy" for s in ("train", "val") for kind in ("ids", "labels")])
    for name in required:
        if sha256(data_dir / name) != manifest["files"][name]:
            raise ValueError(f"Data checksum mismatch: {name}")
    # 词表大小以实际分词器为准，模型窗口与处理数据保持一致。
    mc_dict = dict(cfg["model"], vocab_size=load_tokenizer(tokenizer_path).get_vocab_size())
    mc = ModelConfig(**mc_dict)
    if mc.block_size != manifest["block_size"]:
        raise ValueError("Model block_size differs from prepared data")
    # 将配置、数据、精度与源码共同写入训练签名。
    signature = object_hash({"config": cfg, "model": asdict(mc), "data": manifest,
                             "deterministic": deterministic, "precision": precision, "source": source_fingerprint()})
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Run exists; resume it or choose a new output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model = GPT(mc).to(device)
    # 只对矩阵参数施加权重衰减，偏置和归一化参数单独分组。
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=tc["learning_rate"], betas=(0.9, 0.95), fused=device == "cuda")
    generator = torch.Generator().manual_seed(cfg["seed"] + 1)
    step, best_loss, train_seconds, seen_input_tokens, seen_target_tokens = 0, float("inf"), 0.0, 0, 0
    # 微调初始化只恢复预训练模型权重，优化器从当前任务重新建立。
    if init:
        state = load_checkpoint(init)
        if state["tokenizer_sha256"] != tok_hash or state["model_config"] != asdict(mc):
            raise ValueError("Initial checkpoint architecture/tokenizer mismatch")
        if state["stage"] != "pretrain":
            raise ValueError("SFT init must be a pretraining checkpoint")
        model.load_state_dict(state["model"])
    # 续训恢复完整训练状态，并核对签名以接续原有更新。
    if resume:
        state = load_checkpoint(resume)
        if signature != state["signature"]:
            raise ValueError("Resume requires the same config, data, source, precision and deterministic setting")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best_loss = state["step"], state["best_val_loss"]
        train_seconds = state["train_seconds"]
        seen_input_tokens, seen_target_tokens = state["seen_input_tokens"], state["seen_target_tokens"]
        generator.set_state(state["batch_rng"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    batches = Batches(data_dir, cfg["stage"], mc.block_size, tc["batch_size"])
    log_path = out / "metrics.jsonl"
    if resume and log_path.exists():
        # 恢复较早检查点时，裁去其后日志，使记录与当前恢复进度对齐。
        retained = [r for r in jsonl(log_path) if r["step"] <= step]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in retained), encoding="utf-8")
    # 将实际配置、硬件环境和初始化来源写入运行记录。
    info = {"config": cfg, "actual_model_config": asdict(mc), "parameters": model.parameter_count(),
            "environment": environment(), "actual_precision": precision, "signature": signature,
            "data_manifest_sha256": sha256(data_dir / "manifest.json"), "tokenizer_sha256": tok_hash,
            "init_checkpoint": str(init) if init else None,
            "init_checkpoint_sha256": sha256(init) if init else None,
            "deterministic": deterministic, "sampling": "random windows/examples with replacement"}
    if not resume:
        write_json(out / "run.json", info)
    else:
        write_json(out / f"resume_step_{step}.json", {**info, "resume_checkpoint": str(resume)})

    # 同时追加逐步 JSON 日志并输出当前进度。
    def log(row):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # 收集恢复训练所需的权重、优化器、进度和随机状态。
    def snapshot():
        return {"schema": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": asdict(mc), "config": cfg, "stage": cfg["stage"], "step": step,
                "signature": signature, "tokenizer_sha256": tok_hash,
                "best_val_loss": best_loss, "train_seconds": train_seconds,
                "seen_input_tokens": seen_input_tokens, "seen_target_tokens": seen_target_tokens,
                "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}

    # 计算验证损失，更新最近检查点和最佳检查点。
    def evaluate_and_save():
        nonlocal best_loss
        # 使用固定验证采样比较检查点，最佳损失越低越好。
        loss, count = validation(model, batches, tc["eval_batches"], cfg["seed"], device, precision)
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        improved = loss < best_loss
        best_loss = min(best_loss, loss)
        log({"event": "val", "step": step, "loss": loss, "target_tokens": count,
             "ppl": math.exp(loss) if loss < 700 else None})
        state = snapshot()
        atomic_save(state, out / "last.pt")
        if improved:
            atomic_save(state, out / "best.pt")

    stop_requested = False

    # 收到停止信号后先记录请求，留待当前更新完成再保存。
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finish the optimizer step and save last.pt.", flush=True)

    # 注册停止信号，收到请求后在完整优化器步的边界保存退出。
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    if not resume:
        atomic_save({"model": model.state_dict(), "model_config": asdict(mc), "tokenizer_sha256": tok_hash,
                     "stage": cfg["stage"], "step": 0}, out / "initial.pt")
        evaluate_and_save()
    # 训练结束或异常退出时，恢复原有的进程信号处理函数。
    try:
        model.train()
        # 每次循环对应一次优化器更新，内部积累多个微批次的梯度。
        while step < tc["max_steps"] and (stop_after is None or step < stop_after):
            started = time.perf_counter()
            microbatches = [batches.sample("train", generator) for _ in range(tc["grad_accum"])]
            # 各微批次标题长度不同，用整个有效批量的监督 token 数归一化。
            # 逐批 backward 累加的梯度等价于对这些有效 token 一次求平均。
            targets = sum((y != -100).sum().item() for _, y in microbatches)
            inputs = sum(x.numel() for x, _ in microbatches)
            # 清空上一优化器步的梯度，并更新当前步的学习率。
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss = 0.0
            # 各微批次都除以整组有效 token 数，使累积梯度等价于整批平均。
            for x, y in microbatches:
                x, y = x.to(device), y.to(device)
                with amp_context(device, precision):
                    logits = model(x)
                    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                               ignore_index=-100, reduction="sum")
                    loss = loss_sum / targets
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                total_loss += loss_sum.detach().item()
                loss.backward()
            # 所有微批次反向传播后，统一裁剪梯度并执行一次参数更新。
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            # CUDA 同步后计时，累计真实训练步消耗和 token 数。
            seconds = time.perf_counter() - started
            step += 1
            train_seconds += seconds
            seen_input_tokens += inputs
            seen_target_tokens += targets
            # 按日志间隔保存训练指标，在验证间隔更新检查点。
            if step == 1 or step % tc["log_every"] == 0:
                log({"event": "train", "step": step, "loss": total_loss / targets, "lr": lr,
                     "grad_norm": float(norm), "step_seconds": seconds, "input_tokens_per_second": inputs / seconds,
                     "seen_input_tokens": seen_input_tokens, "seen_target_tokens": seen_target_tokens,
                     "train_seconds": train_seconds,
                     "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0})
            if step % tc["eval_every"] == 0 or step == tc["max_steps"]:
                evaluate_and_save()
            if stop_requested or (stop_file is not None and Path(stop_file).exists()):
                break
        # 试跑或中断后保存当前进度，学习率仍按原计划计算。
        atomic_save(snapshot(), out / "last.pt")
        write_json(out / "status.json", {"step": step, "planned_steps": tc["max_steps"],
                   "complete": step == tc["max_steps"], "train_seconds": train_seconds,
                   "seen_input_tokens": seen_input_tokens, "seen_target_tokens": seen_target_tokens})
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return out / "last.pt"


# 解析命令行参数后，执行预训练和监督微调，按有效 token 累积梯度并保存训练状态。
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--init")
    ap.add_argument("--resume")
    ap.add_argument("--stop-after", type=int, help="Total optimizer step to stop at, preserving the full LR schedule")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--stop-file", help="Save and exit after the current step when this file exists")
    args = ap.parse_args()
    train(args.config, args.data, args.output, args.device, args.init, args.resume,
          args.stop_after, args.threads, args.deterministic, args.stop_file)


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
