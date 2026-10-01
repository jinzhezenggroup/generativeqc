"""Repeat the frozen, interleaved CPU exact-RHF endpoint protocol."""

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
args = parser.parse_args()
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
cases = list(json.loads((root / "cases.json").read_text()))
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
for repeat in range(3):
    for case in cases:
        engines = ["baseline", "candidate", "pyscf-auto"]
        if repeat % 2:
            engines.reverse()
        for engine in engines:
            native = "candidate" if engine == "candidate" else "baseline"
            env.update(
                GQC_SOURCE_ROOT=str(getattr(args, f"{native}_source").resolve()),
                GENERATIVEQC_LIBRARY=str(getattr(args, f"{native}_library").resolve()),
            )
            stem = f"{case}-{engine}-{repeat + 1}"
            start = time.perf_counter()
            with (args.output / f"{stem}.log").open("w") as log:
                run = subprocess.run(
                    [
                        sys.executable,
                        str(root / "benchmark.py"),
                        "pyscf-auto" if engine == "pyscf-auto" else "gqc",
                        case,
                        str(args.output / f"{stem}.json"),
                    ],
                    check=False,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
            row = {
                "case": case,
                "engine": engine,
                "repeat": repeat + 1,
                "returncode": run.returncode,
                "process_wall_seconds": time.perf_counter() - start,
                "json_file": f"{stem}.json",
            }
            status.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "run-status.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )
            if run.returncode:
                raise RuntimeError(f"Endpoint process failed: {stem}")
