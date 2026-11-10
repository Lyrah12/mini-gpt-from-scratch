"""Measure real 50M-model CUDA training speed/memory using random IDs; not a quality experiment."""
import gc
import time
import torch
from torch.nn import functional as F
from minigpt.common import environment, write_json
from minigpt.model import GPT, ModelConfig


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    results = []
    for batch in (2, 4):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = GPT(ModelConfig()).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=True)
        x = torch.randint(3, 16000, (batch, 512), device="cuda")
        y = torch.randint(3, 16000, (batch, 512), device="cuda")
        times = []
        try:
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
        del model, optimizer, x, y
        if 'logits' in locals(): del logits, loss
        gc.collect()
    write_json("work/cuda_benchmark.json", {"environment": environment(), "precision": "bf16", "results": results})


if __name__ == "__main__":
    main()
