"""Record this CPU allocation and numerical libraries without credential reads."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy  # noqa: F401 -- load BLAS before inspecting the actual thread pools
import pyscf
from threadpoolctl import threadpool_info


def main() -> None:
    pools = threadpool_info()
    libraries = {
        p.resolve()
        for p in Path(pyscf.__file__).parent.glob("lib/**/*.so*")
        if p.is_file()
    }
    libraries.update(Path(p["filepath"]).resolve() for p in pools)
    libraries.add(Path(sys.executable).resolve())
    record = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "affinity_cpus": sorted(os.sched_getaffinity(0)),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("pyscf", "numpy", "scipy", "h5py", "threadpoolctl")
        },
        "cgroup": {
            name: Path("/sys/fs/cgroup", name).read_text().strip()
            for name in ("cpu.max", "memory.max", "cpuset.cpus.effective")
            if Path("/sys/fs/cgroup", name).is_file()
        },
        "threadpools": pools,
        "libraries": {
            str(p): {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(libraries)
        },
    }
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
