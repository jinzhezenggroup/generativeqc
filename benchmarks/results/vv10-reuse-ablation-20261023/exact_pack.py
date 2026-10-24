"""Lossless logical-family packs with exact original bytes, modes and hashes."""

import sys
import os
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE') not in (None,'','0'):
    raise RuntimeError('optimized Python is unsupported for exact receipt integrity')
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import stat


SCHEMA = "generativeqc.exact-receipt-pack.v1"
BINARY_SCHEMA = "generativeqc.exact-receipt-pack.v2"


def require(condition,reason):
    if not condition:raise ValueError(reason)


def relative(name):
    require(isinstance(name,str) and name and "\\" not in name and ":" not in name,'invalid relative member')
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ('','.','..') for p in name.split('/')),'escaping or ambiguous member')
    return name


def checked_path(root,name):
    path=root/relative(name)
    require(path.resolve().is_relative_to(root),'member escaped root')
    for part in (path,*path.parents):
        require(not part.is_symlink(),'symlink in member path')
        if part==root:break
    return path


def encode(root, names):
    root = Path(root).resolve()
    rows, seen = [], set()
    binary=False
    for name in names:
        name = relative(name)
        require(name not in seen,'duplicate member')
        seen.add(name)
        path = checked_path(root,name)
        require(path.is_file(),'invalid source member')
        raw = path.read_bytes()
        row={'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mode':stat.S_IMODE(path.stat().st_mode)}
        try:row['utf8']=raw.decode('utf-8')
        except UnicodeDecodeError:
            binary=True;row['binary_base64']=base64.b64encode(raw).decode('ascii')
        rows.append(row)
    require(bool(rows),'empty receipt family')
    return lzma.compress((json.dumps({'schema':BINARY_SCHEMA if binary else SCHEMA,'members':rows},separators=(',',':'),allow_nan=False)+'\n').encode(),preset=9)


def member_bytes(row):
    require(('utf8' in row)!=('binary_base64' in row),'one explicit member encoding required')
    require(isinstance(row.get('utf8',row.get('binary_base64')),str),'invalid member payload')
    return row['utf8'].encode('utf-8') if 'utf8' in row else base64.b64decode(row['binary_base64'],validate=True)


def decode(data):
    value = json.loads(lzma.decompress(data))
    require(value['schema'] in (SCHEMA,BINARY_SCHEMA) and isinstance(value['members'],list) and value['members'],'invalid exact pack schema/members')
    seen = set()
    for row in value["members"]:
        name = relative(row["path"])
        require(name not in seen,'duplicate restored member')
        seen.add(name)
        require(type(row['mode']) is int and 0<=row['mode']<=0o777,'invalid stored mode')
        require(type(row['bytes']) is int and row['bytes']>=0,'invalid stored byte count')
        require(value['schema']==BINARY_SCHEMA or 'binary_base64' not in row,'binary member in legacy schema')
        raw=member_bytes(row)
        require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'original byte/hash mismatch')
    return value["members"]


def restore(data, destination):
    require(not Path(destination).is_symlink(),'symlink destination')
    destination = Path(destination).resolve()
    rows = decode(data)
    targets=[checked_path(destination,row['path']) for row in rows]
    require(len(set(targets))==len(targets),'destination path collision')
    for target in targets:
        require(not target.exists(),'restore refuses existing member')
        require(not any(parent in targets for parent in target.parents),'file/directory member collision')
        require(all(not parent.exists() or parent.is_dir() for parent in target.parents),'restore ancestor is not a directory')
    for row,target in zip(rows,targets,strict=True):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(member_bytes(row))
        if os.name == "posix":
            target.chmod(row["mode"])
            require(stat.S_IMODE(target.stat().st_mode)==row['mode'],'POSIX mode not restored')
        require(hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],'restored byte/hash mismatch')
    return {"members": len(rows), "bytes_verified": True,
            "original_modes_retained": True, "posix_modes_applied": os.name == "posix"}
