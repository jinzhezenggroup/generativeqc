"""Paired ABBA/BAAB comparison against exact #2827 built on the same machine."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
scope = os.environ.get("PACKAGE_SCOPE", "narrow")
out = root / f"evidence/performance-{scope}"
out.mkdir(exist_ok=True)
(out / "environment.json").write_text(
    json.dumps(
        dict(
            node=platform.node(),
            python=platform.python_version(),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            gpu_inventory=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).splitlines(),
            cpu_threads={
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        ),
        indent=2,
    )
    + "\n"
)
base = Path(
    f"/home/jzzeng/tmp/qc-1123-baseline-20261027/evidence/{scope}-incumbent-e599.so"
)
candidate = root / f"evidence/{scope}-candidate-e599.so"
for index, arm in enumerate(
    (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    )
):
    subprocess.run(
        [sys.executable, "bench-native.py", str(out / f"{index}-{arm}.json")],
        env={
            **os.environ,
            "GENERATIVEQC_LIBRARY": str(base if arm == "baseline" else candidate),
        },
        check=True,
        timeout=3600,
    )
    print(index, arm, "completed", flush=True)
