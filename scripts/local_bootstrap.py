"""等待本机 CUDA 环境就绪，依次执行测速、试跑和正式实验。"""
# 导入本模块使用的标准库工具。
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
# 复用项目内的数据、模型或实验工具。
from minigpt.common import read_json,write_json


# 等待环境安装并按测速结果配置本机训练批量。
def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_path=root/"work/local_bootstrap.json"
    python=root/"work/venv_cuda/Scripts/python.exe"
    # 写入启动流程状态，供外部查看当前阶段。
    def update(phase,**more):
        write_json(state_path,{"pid":os.getpid(),"phase":phase,"updated_utc":datetime.now(timezone.utc).isoformat(),**more})
        print(phase,flush=True)
    # 使用 CUDA 环境启动子任务，并将输出追加到独立日志。
    def run(name,args):
        update(name)
        with (root/f"work/{name}.log").open("a",encoding="utf-8") as log:
            subprocess.run([str(python),"-X","utf8","-u",*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    # 完整流程结束或报错时，恢复 Windows 默认的休眠设置。
    try:
        update("waiting_cuda_install")
        deadline=time.monotonic()+4*3600
        # 轮询安装日志，在成功、失败或超时时离开等待阶段。
        while True:
            text=(root/"work/install_cuda.log").read_text(encoding="utf-8",errors="replace") if (root/"work/install_cuda.log").exists() else ""
            if "Successfully installed" in text:break
            if "ERROR:" in text:raise RuntimeError("CUDA installation failed; see work/install_cuda.log")
            if time.monotonic()>deadline:raise TimeoutError("CUDA installation did not finish within four hours")
            time.sleep(5)
        run("gpu_benchmark",["-m","scripts.benchmark_gpu"])
        benchmark=read_json(root/"work/cuda_benchmark.json")
        # 从测速结果选择仍有可用显存的最快微批量。
        candidates=[r for r in benchmark["results"] if not r.get("oom") and r["global_free_bytes_after"]>=1024**3]
        if not candidates:raise RuntimeError("No benchmark batch leaves a 1 GiB memory margin")
        chosen=max(candidates,key=lambda r:r["input_tokens_per_second"])
        batch=chosen["batch_size"]
        # 调整微批量后同步修改累积次数，以维持原有效批量。
        for name in ("pretrain","sft","scratch_sft"):
            path=root/f"configs/local_{name}.json"
            cfg=read_json(path)
            cfg["train"]["batch_size"]=batch
            cfg["train"]["grad_accum"]=(64 if name=="pretrain" else 32)//batch
            write_json(path,cfg)
        update("gpu_batch_selected",batch_size=batch,benchmark=chosen)
        # 先运行短时试跑，确认完成后再进入正式训练。
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


# 直接运行模块时，从命令行入口启动。
if __name__=="__main__":main()
