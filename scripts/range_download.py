"""并发下载 HTTPS 文件分段，支持续传并校验完整文件。"""
# 导入本模块使用的标准库工具。
import argparse
import concurrent.futures
import hashlib
import math
import shutil
import time
import urllib.parse
from pathlib import Path
# 使用流式 HTTP 请求下载指定字节范围。
import requests


# 下载或接续各文件分段，合并后核对完整文件摘要。
def download(url, output, size, expected_sha256, chunk_size=8*1024*1024, workers=4):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 流式计算摘要，供已存在文件和合并文件共同校验。
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
        return h.hexdigest()
    # 已存在完整文件时先校验摘要，通过后直接返回。
    if output.exists():
        if digest(output)!=expected_sha256:raise ValueError("Existing file checksum mismatch")
        return output
    parts=output.with_name(output.name+".parts")
    parts.mkdir(exist_ok=True)
    count=math.ceil(size/chunk_size)
    # 为单个分段续传并有限重试，严格检查响应范围与长度。
    def fetch(index):
        # 将目标文件划分为互不重叠的字节区间。
        start=index*chunk_size;end=min(size,start+chunk_size)-1
        path=parts/f"{index:05d}.part"
        if path.exists() and path.stat().st_size==end-start+1:return index
        last=None
        # 每个分段最多尝试三次，部分文件从已有字节数继续下载。
        for attempt in range(3):
            # 网络或长度校验失败时保留分段进度，并等待后重试。
            try:
                have=path.stat().st_size if path.exists() else 0
                if have>=end-start+1:have=0
                requested_start=start+have
                join="&" if "?" in url else "?"
                target=url+join+f"range_start={requested_start}&attempt={attempt}"
                # 请求指定字节范围并关闭内容压缩，确保分段位置可校验。
                with requests.get(target,headers={"Range":f"bytes={requested_start}-{end}","Accept-Encoding":"identity"},stream=True,timeout=(20,40)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {requested_start}-{end}/{size}":
                        raise ValueError(f"Server did not honor requested range: {r.status_code} {r.headers.get('Content-Range')}")
                    with path.open("ab" if have else "wb") as f:
                        for chunk in r.iter_content(65536):
                            if chunk:f.write(chunk)
                if path.stat().st_size!=end-start+1:raise ValueError("Incomplete range")
                return index
            except (requests.RequestException,ValueError) as e:
                last=e
                time.sleep(1)
        raise RuntimeError(f"Range {index} failed: {last}")
    completed=0
    # 并发下载各段，主线程仅汇总已完成数量。
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(count)]):
            result.result();completed+=1
            if completed%4==0 or completed==count:
                print(f"Ranges {completed}/{count} ({100*completed/count:.1f}%)",flush=True)
    temp=output.with_name(output.name+".assembling")
    # 按分段编号顺序合并到临时文件，完整摘要通过后改为正式名称。
    with temp.open("wb") as dest:
        for i in range(count):
            with (parts/f"{i:05d}.part").open("rb") as source:shutil.copyfileobj(source,dest)
    actual=digest(temp)
    if actual!=expected_sha256:raise ValueError(f"SHA256 mismatch: {actual}")
    temp.rename(output)
    print(f"Verified SHA256: {actual}; saved {output}",flush=True)
    return output


# 解析命令行参数后，并发下载 HTTPS 文件分段，支持续传并校验完整文件。
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--size",type=int,required=True);ap.add_argument("--sha256",required=True)
    ap.add_argument("--chunk-mb",type=int,default=8);ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args()
    download(a.url,a.output,a.size,a.sha256,a.chunk_mb*1024*1024,a.workers)


# 直接运行模块时，从命令行入口启动。
if __name__=="__main__":main()
