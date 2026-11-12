"""串行执行本机预训练、微调、对照和评测，保存流水线状态。"""
# 导入本模块使用的标准库工具。
import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
# 复用项目内的数据、模型或实验工具。
from minigpt.common import read_json, sha256, write_json


# 用独立异常表示训练已按暂停请求保存。
class LocalPaused(Exception):
    pass


# 依次运行本机训练、对照和评测，并维护运行锁。
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",default="data/processed_xlsum")
    ap.add_argument("--runs",default="runs/local_4060")
    ap.add_argument("--pilot",action="store_true")
    a=ap.parse_args()
    root=Path(a.runs);root.mkdir(parents=True,exist_ok=True)
    logs=root/"logs";logs.mkdir(exist_ok=True)
    lock=root/"pipeline.lock"
    # 检查上次进程是否仍在运行，清理已退出进程遗留的锁。
    if lock.exists():
        old=read_json(lock);pid=old["pid"]
        # Windows 使用进程句柄检查锁文件中的进程是否仍在运行。
        if os.name=="nt":
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
            running=False
            if handle:
                code=ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                running=code.value==259
        else:
            # 在类 Unix 系统中，通过零信号检查对应进程是否存在。
            try:os.kill(pid,0);running=True
            except ProcessLookupError:running=False
        if running:raise RuntimeError(f"Pipeline already running, PID {pid}")
        lock.unlink()
    # 独占建立运行锁，防止同一输出目录被重复启动。
    with lock.open("x",encoding="utf-8") as f:json.dump({"pid":os.getpid()},f)
    state={"pid":os.getpid(),"phase":"starting","complete":False}
    # 记录流水线阶段、时间和完成状态。
    def update(phase,**extra):
        state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(root/"pipeline.json",state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    # 在当前 Python 环境启动命令，将输出写入阶段日志。
    def execute(name,args):
        update(name)
        with (logs/f"{name}.log").open("a",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-X","utf8","-u",*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    # 为指定训练阶段选择配置和初始化或续训检查点。
    def training(name,initial=None,pilot=False):
        config={"pretrain":"local_pretrain","sft":"local_sft","scratch":"local_scratch_sft"}[name]
        output=root/name
        args=["-m","minigpt.train","--config",f"configs/{config}.json","--data",a.data,"--output",output,"--device","cuda","--stop-file",root/"pause.request"]
        # 优先续训已有检查点，否则按当前阶段选择预训练初始化。
        if (output/"last.pt").exists():args += ["--resume",output/"last.pt"]
        elif initial:args += ["--init",initial]
        if pilot:args += ["--stop-after","30"]
        execute(name,args)
        if not pilot and not read_json(output/"status.json")["complete"]:
            raise LocalPaused("Training saved at a requested pause")
    # 训练期间保持系统唤醒，避免 Windows 自动休眠。
    if os.name=="nt":ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    # 按顺序执行实验，暂停和异常分别写入状态，结束时释放运行锁。
    try:
        if (root/"pause.request").exists():
            update("paused",complete=False)
            return
        # 先完成预训练，试跑模式在指定步数保存后返回。
        training("pretrain",pilot=a.pilot)
        if a.pilot:
            update("pilot_complete",complete=False)
            return
        # 分别执行预训练初始化微调和随机初始化对照。
        training("sft",root/"pretrain/best.pt")
        training("scratch")
        # 各组模型使用相同的验证和测试新闻集合。
        for split,limit in (("val",200),("test",500)):
            # 分别执行首句抽取和三组模型评测，输出保存到独立目录。
            for name in ("lead","pretrain","sft","scratch"):
                output=root/f"eval_{split}_{name}"
                checkpoint=None if name=="lead" else root/name/"best.pt"
                # 复用评测记录前核对数据与检查点，避免混入旧结果。
                if (output/"metrics.json").exists():
                    result=read_json(output/"metrics.json")
                    if result["data_manifest_sha256"]!=sha256(Path(a.data)/"manifest.json"):
                        raise ValueError("Existing evaluation uses different data")
                    if checkpoint and result["checkpoint_sha256"]!=sha256(checkpoint):
                        raise ValueError("Existing evaluation uses different checkpoint")
                    continue
                args=["-m","minigpt.evaluate","--data",a.data,"--output",output,"--split",split,"--limit",limit,"--device","cuda"]
                if checkpoint:args += ["--checkpoint",checkpoint]
                execute(f"eval_{split}_{name}",args)
        if not (root/"continuations.json").exists():
            execute("continuations",["-m","scripts.collect_continuations","--runs",root,"--data",a.data,"--device","cuda"])
        # 训练与评测完成后导出原始结果素材。
        execute("export_report",["-m","scripts.export_report","--runs",root,"--data",a.data,"--output","report/generated"])
        update("experiments_complete_report_assets_exported",complete=True,
               note="Review report narrative and personal reflection in Overleaf; no local PDF compilation.")
    except LocalPaused as e:
        update("paused",complete=False,note=str(e))
    except Exception as e:
        update("failed",complete=False,error=str(e))
        raise
    finally:
        # 结束后恢复 Windows 默认的休眠设置。
        if os.name=="nt":ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


# 直接运行模块时，从命令行入口启动。
if __name__=="__main__":main()
