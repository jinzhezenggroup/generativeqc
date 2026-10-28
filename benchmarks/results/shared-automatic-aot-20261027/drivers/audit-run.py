"""Apply the incumbent resource ceilings to actual final execution receipts."""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--build-root", type=Path, required=True)
p.add_argument("--library", type=Path, required=True)
p.add_argument("--receipts", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
paths = sorted(a.receipts.glob("*.json"))
assert paths
subprocess.run(
    [
        sys.executable,
        "tools/audit_native_automatic_aot.py",
        "--library",
        str(a.library),
        "--build-tree",
        str(a.build_root / "build/pro6000-cuda"),
        "--cuobjdump",
        "/group/software/cuda-12.9.1/bin/cuobjdump",
        "--max-stack",
        "4096",
        "--max-local",
        "4096",
        "--max-shared",
        "49152",
        "--output",
        str(a.output),
        *[part for path in paths for part in ("--receipt", str(path))],
    ],
    check=True,
    timeout=900,
)
