"""Run/resume the real RTX 4060 experiment pipeline; save logs and report assets."""
import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from minigpt.common import read_json, sha256, write_json


class LocalPaused(Exception):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",default="data/processed_xlsum")
    ap.add_argument("--runs",default="runs/local_4060")
    ap.add_argument("--pilot",action="store_true")
    a=ap.parse_args()
    root=Path(a.runs);root.mkdir(parents=True,exist_ok=True)
    logs=root/"logs";logs.mkdir(exist_ok=True)
    lock=root/"pipeline.lock"
    if lock.exists():
        old=read_json(lock);pid=old["pid"]
        if os.name=="nt":
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
            running=False
            if handle:
                code=ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                running=code.value==259
        else:
            try:os.kill(pid,0);running=True
            except ProcessLookupError:running=False
        if running:raise RuntimeError(f"Pipeline already running, PID {pid}")
        lock.unlink()
    with lock.open("x",encoding="utf-8") as f:json.dump({"pid":os.getpid()},f)
    state={"pid":os.getpid(),"phase":"starting","complete":False}
    def update(phase,**extra):
        state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(root/"pipeline.json",state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    def execute(name,args):
        update(name)
        with (logs/f"{name}.log").open("a",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-X","utf8","-u",*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    def training(name,initial=None,pilot=False):
        config={"pretrain":"local_pretrain","sft":"local_sft","scratch":"local_scratch_sft"}[name]
        output=root/name
        args=["-m","minigpt.train","--config",f"configs/{config}.json","--data",a.data,"--output",output,"--device","cuda","--stop-file",root/"pause.request"]
        if (output/"last.pt").exists():args += ["--resume",output/"last.pt"]
        elif initial:args += ["--init",initial]
        if pilot:args += ["--stop-after","30"]
        execute(name,args)
        if not pilot and not read_json(output/"status.json")["complete"]:
            raise LocalPaused("Training saved at a requested pause")
    if os.name=="nt":ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if (root/"pause.request").exists():
            update("paused",complete=False)
            return
        training("pretrain",pilot=a.pilot)
        if a.pilot:
            update("pilot_complete",complete=False)
            return
        training("sft",root/"pretrain/best.pt")
        training("scratch")
        for split,limit in (("val",200),("test",500)):
            for name in ("lead","pretrain","sft","scratch"):
                output=root/f"eval_{split}_{name}"
                checkpoint=None if name=="lead" else root/name/"best.pt"
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
        execute("export_report",["-m","scripts.export_report","--runs",root,"--data",a.data,"--output","report/generated"])
        update("experiments_complete_report_assets_exported",complete=True,
               note="Review report narrative and personal reflection in Overleaf; no local PDF compilation.")
    except LocalPaused as e:
        update("paused",complete=False,note=str(e))
    except Exception as e:
        update("failed",complete=False,error=str(e))
        raise
    finally:
        if os.name=="nt":ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=="__main__":main()
