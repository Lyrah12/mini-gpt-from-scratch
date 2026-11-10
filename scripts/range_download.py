"""Verified, resumable HTTPS range downloader for large public artifacts."""
import argparse
import concurrent.futures
import hashlib
import math
import shutil
import time
import urllib.parse
from pathlib import Path
import requests


def download(url, output, size, expected_sha256, chunk_size=8*1024*1024, workers=4):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
        return h.hexdigest()
    if output.exists():
        if digest(output)!=expected_sha256:raise ValueError("Existing file checksum mismatch")
        return output
    parts=output.with_name(output.name+".parts")
    parts.mkdir(exist_ok=True)
    count=math.ceil(size/chunk_size)
    def fetch(index):
        start=index*chunk_size;end=min(size,start+chunk_size)-1
        path=parts/f"{index:05d}.part"
        if path.exists() and path.stat().st_size==end-start+1:return index
        last=None
        for attempt in range(3):
            try:
                have=path.stat().st_size if path.exists() else 0
                if have>=end-start+1:have=0
                requested_start=start+have
                join="&" if "?" in url else "?"
                target=url+join+f"range_start={requested_start}&attempt={attempt}"
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(count)]):
            result.result();completed+=1
            if completed%4==0 or completed==count:
                print(f"Ranges {completed}/{count} ({100*completed/count:.1f}%)",flush=True)
    temp=output.with_name(output.name+".assembling")
    with temp.open("wb") as dest:
        for i in range(count):
            with (parts/f"{i:05d}.part").open("rb") as source:shutil.copyfileobj(source,dest)
    actual=digest(temp)
    if actual!=expected_sha256:raise ValueError(f"SHA256 mismatch: {actual}")
    temp.rename(output)
    print(f"Verified SHA256: {actual}; saved {output}",flush=True)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--size",type=int,required=True);ap.add_argument("--sha256",required=True)
    ap.add_argument("--chunk-mb",type=int,default=8);ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args()
    download(a.url,a.output,a.size,a.sha256,a.chunk_mb*1024*1024,a.workers)


if __name__=="__main__":main()
