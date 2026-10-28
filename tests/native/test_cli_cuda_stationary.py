"""Run the curated native CUDA DFT energy+force CLI endpoints on a real NVIDIA GPU.

Run with the *installed* standalone CLI under a Slurm GPU allocation. This
executes the C++ binary as a subprocess, not the Python Calculator/science path.
Unlike a host-only link check, the request must fail without a working GPU.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

METHODS = (
    "lda-rks",
    "lda-uks",
    "pbe-rks",
    "pbe-uks",
    "pbe0-rks",
    "pbe0-uks",
    "r2scan-rks",
    "r2scan-uks",
    "b3lyp-rks",
    "b3lyp-uks",
    "wb97m-v-rks",
    "wb97m-v-uks",
    "m06-2x-rks",
    "m06-2x-uks",
    "mn15-rks",
    "mn15-uks",
)
# Non-symmetry-aligned H2 coordinates, in Bohr; only s shell functions.
BASE_COORDS = ((0.12, -0.09, -0.71), (-0.08, 0.13, 0.79))
BOHR_STEP = 5.0e-4


def write_xyz(path: Path, displacement: float = 0.0) -> None:
    coords = [list(point) for point in BASE_COORDS]
    coords[1][2] += displacement
    path.write_text(
        "2\nnative CLI force endpoint in bohr\n"
        + "".join(f"H {x:.12f} {y:.12f} {z:.12f}\n" for x, y, z in coords),
        encoding="utf-8",
    )


def calculate(cli: Path, xyz: Path, method: str, *, forces: bool, fitting: str) -> dict:
    open_shell = method.endswith("-uks")
    command = [
        str(cli),
        "run",
        str(xyz),
        "--units",
        "bohr",
        "--method",
        method,
        "--basis",
        "sto-3g",
        "--backend",
        "cuda",
        "--representation",
        "cartesian",
        "--density-fitting",
        fitting,
        "--precision",
        "fp64",
        "--charge",
        "1" if open_shell else "0",
        "--multiplicity",
        "2" if open_shell else "1",
        "--max-iterations",
        "200",
        "--energy-tolerance",
        "1e-12",
        "--density-tolerance",
        "1e-10",
    ]
    if forces:
        command.append("--forces")
    command.append("--json")
    run = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PATH": "/no-python-or-compiler",
            "GENERATIVEQC_PROFILE": "off",
        },
    )
    if run.returncode:
        raise AssertionError(
            f"{method} {'E+F' if forces else 'E'} failed "
            f"(exit {run.returncode}):\n{run.stderr}\n{run.stdout}"
        )
    result = json.loads(run.stdout)
    if result.get("method") != method or result.get("backend") != "cuda":
        raise AssertionError(f"{method}: wrong method/backend in result: {result}")
    if not math.isfinite(result["energy_hartree"]):
        raise AssertionError(f"{method}: nonfinite energy")
    return result


def check_method(cli: Path, tmp: Path, method: str, fitting: str) -> None:
    xyz = tmp / "h2.xyz"
    write_xyz(xyz)
    baseline = calculate(cli, xyz, method, forces=False, fitting=fitting)
    result = calculate(cli, xyz, method, forces=True, fitting=fitting)
    if abs(baseline["energy_hartree"] - result["energy_hartree"]) > 5e-8:
        raise AssertionError(f"{method}: E+F changed the converged SCF energy")
    vectors = result.get("forces_hartree_per_bohr")
    if not isinstance(vectors, list) or len(vectors) != 2:
        raise AssertionError(f"{method}: missing complete two-atom force array")
    if any(
        not isinstance(v, list)
        or len(v) != 3
        or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in v)
        for v in vectors
    ):
        raise AssertionError(f"{method}: invalid analytic forces: {vectors}")
    for axis in range(3):
        if abs(vectors[0][axis] + vectors[1][axis]) > 2e-5:
            raise AssertionError(
                f"{method}: translation invariance failed on axis {axis}"
            )
    energies = []
    for tag, shift in (("plus", BOHR_STEP), ("minus", -BOHR_STEP)):
        displaced = tmp / f"{tag}.xyz"
        write_xyz(displaced, shift)
        energies.append(
            calculate(cli, displaced, method, forces=False, fitting=fitting)[
                "energy_hartree"
            ]
        )
    reference_force = -(energies[0] - energies[1]) / (2 * BOHR_STEP)
    if abs(vectors[1][2] - reference_force) > 2e-3:
        raise AssertionError(
            f"{method}: analytic z force {vectors[1][2]:.9g} differs from "
            f"fully reconverged finite difference {reference_force:.9g}"
        )
    print(
        f"PASS {method}/{fitting}: energy, full analytic forces, translation and reconverged FD",
        flush=True,
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_cli_cuda_stationary.py /path/to/bin/generativeqc")
    assert os.environ.get("SLURM_JOB_ID"), "run this gate through srun"
    cli = Path(sys.argv[1]).resolve()
    if not cli.is_file():
        raise SystemExit(f"no CLI executable: {cli}")
    with tempfile.TemporaryDirectory(prefix="generativeqc-cli-cuda-ef-") as directory:
        tmp = Path(directory)
        for method in METHODS:
            for fitting in ("none", "cuda"):
                check_method(cli, tmp, method, fitting)
    print(f"PASS all {2 * len(METHODS)} native CUDA CLI E+F endpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
