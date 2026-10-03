#!/usr/bin/env python3
"""Bounded verification/expansion only; never execute recovered code."""
import argparse,hashlib,json,lzma
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent
with (root/'manifest.json').open('rb') as f:mb=f.read(16385)
need(len(mb)<=16384,'manifest too large');m=json.loads(mb);out={};seen=set()
limits={'capsule.json.xz':2<<20,'fresh_endpoint.py.xz':64<<10,'README.md':32<<10,'decode.py':16<<10}
for f in m['files']:
    n=f['path'];need(n in limits and n not in seen,'unsafe/duplicate path');seen.add(n)
    with (root/n).open('rb') as stream:b=stream.read((1<<20)+1)
    need(len(b)<=1<<20 and len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],'stored bytes/hash')
    if n.endswith('.xz'):
        z=lzma.LZMADecompressor(memlimit=128<<20);raw=z.decompress(b,max_length=limits[n]+1)
        need(z.eof and not z.unused_data and len(raw)<=limits[n],'XZ output/EOF/trailing or multistream')
        need(len(raw)==f['decoded_bytes'] and hashlib.sha256(raw).hexdigest()==f['decoded_sha256'],'decoded bytes/hash')
        out[n[:-3]]=raw
    else:need(len(b)<=limits[n],'text too large')
need(seen==set(limits),'incomplete inventory');json.loads(out['capsule.json'])
if a.output_dir:
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for n,b in out.items():(a.output_dir/n).write_bytes(b)
print('Verified selective capsule and replay sources; no scientific code executed')
