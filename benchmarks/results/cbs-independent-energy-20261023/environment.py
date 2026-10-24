"""Snapshot numerical libraries and CPU allocation without reading credentials."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy  # noqa: F401
import pyscf
from threadpoolctl import threadpool_info


def main() -> None:
    pools = threadpool_info()
    libraries = set(Path(pyscf.__file__).parent.glob("lib/**/*.so*"))
    libraries.update(Path(p["filepath"]) for p in pools)
    libraries.add(Path(sys.executable).resolve())
    libraries = {p for p in libraries if p.is_file()}
    record = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "affinity_cpus": sorted(os.sched_getaffinity(0)),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("pyscf", "numpy", "scipy", "h5py", "threadpoolctl")
        },
        "libraries": {
            str(p): {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in sorted(libraries)
        },
        "threadpools": pools,
        "cgroup": {
            name: Path("/sys/fs/cgroup", name).read_text().strip()
            for name in ("cpu.max", "memory.max", "cpuset.cpus.effective")
            if Path("/sys/fs/cgroup", name).is_file()
        },
    }
    Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
