"""Recover the four hash-bound benchmark inputs from retained Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="ignored directory for compressed inputs")
    args = parser.parse_args()
    identity = json.loads((HERE / "provenance.json").read_text())["inputs"]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, digest in identity["sha256"].items():
        source = f"benchmarks/results/pbe0-derivative-work-20261005/drivers/{name}"
        raw = subprocess.check_output(["git", "show", f"{identity['source_commit']}:{source}"], cwd=HERE)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"retained input hash mismatch: {name}")
        (args.output / name).write_bytes(raw)
    print("Recovered and verified all four retained geometry/basis inputs")


if __name__ == "__main__":
    main()
