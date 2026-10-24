"""Verify or restore retained raw files into a new directory; never execute them."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from replay import load_raw, sha

HERE = Path(__file__).resolve().parent


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("unsafe retained path")
    return path


def members(directory):
    water = load_raw(directory / "raw-water.json.xz")
    audit = json.loads(water["files"]["input/root-audit.json"])
    modes = {row["path"]: row for row in audit["members"]}
    restored = {}
    for filename in ("raw-build.json.xz", "raw-water.json.xz", "raw-h2.json.xz",
                     "raw-support.json.xz", "raw-audits.json.xz"):
        bundle = load_raw(directory / filename)
        for name, text in bundle["files"].items():
            data = text.encode()
            if filename == "raw-build.json.xz":
                path = (name if name.startswith("input/") else
                        "output/" + name if name.startswith("controller") else
                        "output/attempt1/" + name)
                row = modes[path]
                mode = row["mode"]
            elif filename == "raw-audits.json.xz":
                path = "audit/" + name
                row = bundle["file_metadata"][name]
                mode = None  # Windows records retain their mode context in metadata.
            elif name == "identity.json":
                path = "output/attempt1/identity.json"
                row = modes[path]
                mode = row["mode"]
            else:
                row = bundle["file_metadata"][name]
                path, mode = row["path"], row["mode"]
            safe_path(path)
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise ValueError("retained metadata mismatch: " + path)
            value = (data, mode)
            if path in restored and restored[path] != value:
                raise ValueError("incompatible repeated raw path: " + path)
            restored[path] = value
    return restored


def restore(directory, output=None):
    files = members(directory)
    if output is not None:
        output.mkdir(exist_ok=False)
        for name, (data, mode) in files.items():
            path = output / str(safe_path(name))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            if mode is not None and os.name == "posix":
                path.chmod(mode)
                if path.stat().st_mode & 0o777 != mode:
                    raise ValueError("restored mode mismatch: " + name)
            if path.read_bytes() != data:
                raise ValueError("restored byte mismatch: " + name)
    return {"raw_paths": len(files), "bytes_and_hashes": "PASS",
            "Linux_raw_modes": "retained; applied on POSIX extraction",
            "scope": "lossless recovery only; no scientific execution"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, help="new destination; existing paths are rejected")
    args = parser.parse_args()
    print(json.dumps(restore(args.directory, args.output), indent=2))
