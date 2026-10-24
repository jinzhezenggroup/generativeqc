"""Bounded control process: observe the assigned CUDA-0, never call a solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from native_worker import device, helper
from transport import read, release, write_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    if args.release is None:
        for name, expected in plan["task_files"].items():
            from transport import checked

            checked(args.plan.parent / name, expected)
        write_new(args.output, helper(plan).device_observation())
    else:
        plan, proof = release(args.plan, args.release)
        device(plan, proof, args.output)
