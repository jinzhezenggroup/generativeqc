"""Evaluate one of the two original projected displacement grids."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import sleep

import numpy as np
import pyscf
from check_df_minima import STEPS, differences, point_energy
from check_df_minima import main as assemble
from check_minima import load_canonical_basis, vibrational_basis
from run import dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("case")
    parser.add_argument("phase", choices=("coarse", "fine"))
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    pyscf.lib.num_threads(1)
    if pyscf.__version__ != "2.14.0":
        parser.error(f"expected PySCF 2.14.0, got {pyscf.__version__}")
    orbital = load_canonical_basis(args.payload / "cc-pcvtz.json")
    auxiliary_path = args.payload / "cc-pwcvtz-rifit.json"
    auxiliary = load_canonical_basis(auxiliary_path)
    (case,) = [
        item
        for item in json.loads((args.payload / "cohort.json").read_text())["molecules"]
        if item["id"] == args.case
    ]
    native = args.native / args.case / "df-cc-pcvtz"
    if json.loads((native / "receipt.json").read_text())["status"] != "converged":
        raise ValueError("native geometry has not passed the gate")
    source = (native / "stdout.json").read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    xyz = np.array(json.loads(source)["coordinates_bohr"])
    rank, active = vibrational_basis(xyz)
    if rank != 6:
        raise ValueError(f"expected nonlinear geometry, rigid rank is {rank}")

    dest = args.out / args.case
    center_path = dest / "center.json"
    marker = args.out.parent / "fine-complete.json"
    if args.phase == "coarse":
        dest.mkdir(parents=True)
        center_energy = point_energy(case, xyz, orbital, auxiliary, center_path)
    else:
        while not center_path.is_file():
            sleep(30)
        center = json.loads(center_path.read_text())
        if center["coordinates_bohr"] != xyz.tolist():
            raise ValueError("center coordinates differ from the native geometry")
        center_energy = center["energy_hartree"]
    step = STEPS[0 if args.phase == "coarse" else 1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        differences(
            case, xyz, orbital, auxiliary, active, step, center_energy, dest, pool
        )
    print(args.phase, args.case, "displacements complete", flush=True)
    center_hash = hashlib.sha256(center_path.read_bytes()).hexdigest()
    identity = {"source": source_hash, "center": center_hash, "step": STEPS[1]}
    if args.phase == "fine":
        partial = marker.with_suffix(".partial")
        dump(partial, identity)
        partial.replace(marker)
        return
    while not marker.is_file():
        sleep(30)
    if json.loads(marker.read_text()) != identity:
        raise ValueError("fine displacement grid is for a different center")
    sys.argv = [
        "check_df_minima.py",
        str(args.payload),
        str(args.native),
        str(args.out),
        "--resume",
        "--molecule",
        args.case,
        "--auxiliary",
        str(auxiliary_path),
    ]
    assemble()


if __name__ == "__main__":
    main()
