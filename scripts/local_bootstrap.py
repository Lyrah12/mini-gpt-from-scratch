"""Finish the already-running CUDA setup, benchmark, pilot, and run local experiments."""
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from minigpt.common import read_json,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_path=root/"work/local_bootstrap.json"
    python=root/"work/venv_cuda/Scripts/python.exe"
    def update(phase,**more):
        write_json(state_path,{"pid":os.getpid(),"phase":phase,"updated_utc":datetime.now(timezone.utc).isoformat(),**more})
        print(phase,flush=True)
    def run(name,args):
        update(name)
        with (root/f"work/{name}.log").open("a",encoding="utf-8") as log:
            subprocess.run([str(python),"-X","utf8","-u",*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        update("waiting_cuda_install")
        deadline=time.monotonic()+4*3600
        while True:
            text=(root/"work/install_cuda.log").read_text(encoding="utf-8",errors="replace") if (root/"work/install_cuda.log").exists() else ""
            if "Successfully installed" in text:break
            if "ERROR:" in text:raise RuntimeError("CUDA installation failed; see work/install_cuda.log")
            if time.monotonic()>deadline:raise TimeoutError("CUDA installation did not finish within four hours")
            time.sleep(5)
        run("gpu_benchmark",["-m","scripts.benchmark_gpu"])
        benchmark=read_json(root/"work/cuda_benchmark.json")
        candidates=[r for r in benchmark["results"] if not r.get("oom") and r["global_free_bytes_after"]>=1024**3]
        if not candidates:raise RuntimeError("No benchmark batch leaves a 1 GiB memory margin")
        chosen=max(candidates,key=lambda r:r["input_tokens_per_second"])
        batch=chosen["batch_size"]
        for name in ("pretrain","sft","scratch_sft"):
            path=root/f"configs/local_{name}.json"
            cfg=read_json(path)
            cfg["train"]["batch_size"]=batch
            cfg["train"]["grad_accum"]=(64 if name=="pretrain" else 32)//batch
            write_json(path,cfg)
        update("gpu_batch_selected",batch_size=batch,benchmark=chosen)
        run("local_pilot",["-m","scripts.run_local","--pilot"])
        status=read_json(root/"runs/local_4060/pretrain/status.json")
        if status["step"]<30:raise RuntimeError("Pilot stopped before step 30")
        run("local_pilot_summary",["-m","scripts.pilot_summary","--run","runs/local_4060/pretrain"])
        update("formal_training_starting")
        run("local_full_pipeline",["-m","scripts.run_local"])
        pipeline=read_json(root/"runs/local_4060/pipeline.json")
        update("completed" if pipeline["complete"] else pipeline["phase"],pipeline=pipeline)
    except Exception as error:
        update("failed",error=str(error))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":main()
