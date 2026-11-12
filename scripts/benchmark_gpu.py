"""用随机 token 测量约 5000 万参数模型的 CUDA 训练速度和显存。"""
# 导入本模块使用的标准库工具。
import gc
import time
# 使用 CUDA 张量和交叉熵测量完整训练步的开销。
import torch
from torch.nn import functional as F
# 复用项目内的数据、模型或实验工具。
from minigpt.common import environment, write_json
from minigpt.model import GPT, ModelConfig


# 比较候选微批量的稳定训练耗时和峰值内存。
def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    results = []
    # 分别测试候选微批量，每轮重新建立模型与优化器。
    for batch in (2, 4):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = GPT(ModelConfig()).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=True)
        x = torch.randint(3, 16000, (batch, 512), device="cuda")
        y = torch.randint(3, 16000, (batch, 512), device="cuda")
        times = []
        # 记录显存不足的候选，并保留其他批量的测速机会。
        try:
            # 先预热两步，再以 CUDA 同步后的训练耗时统计稳定速度。
            for i in range(7):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter()-started)
            import statistics
            seconds = statistics.median(times)
            result = {"batch_size": batch, "parameters": model.parameter_count(), "microstep_seconds": seconds,
                      "input_tokens_per_second": batch*512/seconds,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "global_free_bytes_after": torch.cuda.mem_get_info()[0], "quality_experiment": False}
            results.append(result)
            print(result, flush=True)
        except torch.OutOfMemoryError:
            results.append({"batch_size": batch, "oom": True})
            print(f"OOM at batch {batch}", flush=True)
        # 结束当前候选后释放模型、梯度和中间结果。
        del model, optimizer, x, y
        if 'logits' in locals(): del logits, loss
        gc.collect()
    write_json("work/cuda_benchmark.json", {"environment": environment(), "precision": "bf16", "results": results})


# 直接运行模块时，从命令行入口启动。
if __name__ == "__main__":
    main()
