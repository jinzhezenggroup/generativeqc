"""Run paired fresh processes inside one finite Slurm GPU allocation."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
out = root / "evidence/performance"
out.mkdir(exist_ok=True)
assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
hardware = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    check=True,
    timeout=30,
)
(out / "environment.json").write_text(
    json.dumps(
        {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "node": os.uname().nodename,
            "python": sys.version,
            "gpu_inventory": hardware.stdout.splitlines(),
        },
        indent=2,
    )
    + "\n"
)
for i, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
    lib = root / (
        "evidence/libgenerativeqc-force-matrix.so"
        if arm == "baseline"
        else "build/native-cuda/libgenerativeqc.so"
    )
    subprocess.run(
        [sys.executable, "evidence/bench-native.py", str(out / f"{i}-{arm}.json")],
        env={**os.environ, "GENERATIVEQC_LIBRARY": str(lib)},
        check=True,
        timeout=1200,
    )
    print(i, arm, "completed", flush=True)
