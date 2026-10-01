"""Repeat the no-reuse H2 control with alternating independent processes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
for engine in ("baseline", "candidate"):
    parser.add_argument(f"--{engine}-source", type=Path, required=True)
    parser.add_argument(f"--{engine}-library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pairs", type=int, default=20)
args = parser.parse_args()
script = Path(__file__).resolve().parent.parent / "cpu-eri-20261001" / "benchmark.py"
args.output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.pop("LD_PRELOAD", None)
env.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONHASHSEED="0",
)
status = []
for repeat in range(args.pairs):
    engines = ["baseline", "candidate"]
    if repeat % 2:
        engines.reverse()
    for engine in engines:
        env.update(
            GQC_SOURCE_ROOT=str(getattr(args, f"{engine}_source").resolve()),
            GENERATIVEQC_LIBRARY=str(getattr(args, f"{engine}_library").resolve()),
        )
        stem = f"h2-sto3g-{engine}-{repeat + 1}"
        start = time.perf_counter()
        with (args.output / f"{stem}.log").open("w") as log:
            run = subprocess.run(
                [sys.executable, str(script), "gqc", "h2-sto3g", str(args.output / f"{stem}.json")],
                check=False,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        row = {
            "repeat": repeat + 1,
            "engine": engine,
            "returncode": run.returncode,
            "process_wall_seconds": time.perf_counter() - start,
            "json_file": f"{stem}.json",
        }
        status.append(row)
        (args.output / "run-status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if run.returncode:
            raise RuntimeError(f"Endpoint process failed: {stem}")
