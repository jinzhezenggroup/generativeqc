"""Complete named CUDA CLI gates; invoke explicitly inside a finite Slurm job.

CPU comparison checks backend transport; reconverged finite differences test
complete correction forces. WB97M-V uses independent PySCF/Libxc fixtures with
identical unpruned product grids. No Python runtime participates in the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def calculate(
    cli: Path, xyz: Path, method: str, backend: str, forces: bool = False
) -> dict:
    """Execute the standalone native endpoint with explicit numerical controls."""
    command = [
        str(cli),
        "run",
        str(xyz),
        "--method",
        method,
        "--basis",
        "sto-3g",
        "--backend",
        backend,
        "--units",
        "bohr",
        "--grid-radial-points",
        "8",
        "--grid-polar-points",
        "4",
        "--grid-azimuth-points",
        "8",
        "--energy-tolerance",
        "1e-12",
        "--density-tolerance",
        "1e-10",
        "--max-iterations",
        "200",
        "--json",
    ]
    if method.endswith("-uks"):
        command += ["--charge", "1", "--multiplicity", "2"]
    if forces:
        command += ["--forces"]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PATH": "/no-python-runtime", "GENERATIVEQC_PROFILE": "off"},
    )
    return json.loads(result.stdout)


def main() -> None:
    """Validate energies, forces and refusal without borrowing an unallocated GPU."""
    assert os.environ.get("SLURM_JOB_ID"), "run this gate through srun"
    cli = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="named-cuda-") as directory:
        xyz = Path(directory) / "h2.xyz"
        xyz.write_text("2\nH2 in bohr\nH 0 0 -0.7\nH 0 0 0.7\n")
        for spin, energy in (("rks", -1.150312466420175), ("uks", -0.528250800668613)):
            result = calculate(cli, xyz, f"wb97m-v-{spin}", "cuda")
            assert abs(result["energy_hartree"] - energy) < 2e-8, result
            print(
                json.dumps(
                    {
                        "method": result["method"],
                        "energy_error": abs(result["energy_hartree"] - energy),
                    }
                ),
                flush=True,
            )
        for method in (
            "pbe-d3(bj)-rks",
            "pbe-d3(0)-uks",
            "pbe-d3(bj)-atm-rks",
            "pbe0-d4(bj-eeq-atm)-rks",
            "b3lyp-d4(bj-eeq-atm)-rks",
        ):
            xyz.write_text("2\nH2 in bohr\nH 0 0 -0.7\nH 0 0 0.7\n")
            gpu = calculate(cli, xyz, method, "cuda", True)
            cpu = calculate(cli, xyz, method, "cpu", True)
            energy_error = abs(gpu["energy_hartree"] - cpu["energy_hartree"])
            forces = np.array(gpu["forces_hartree_per_bohr"])
            force_error = float(np.max(np.abs(forces - cpu["forces_hartree_per_bohr"])))
            assert energy_error < 2e-8 and force_error < 2e-7, (
                method,
                energy_error,
                force_error,
            )
            errors = []
            for step in (3e-4, 1e-4):
                energies = []
                for sign in (-1, 1):
                    xyz.write_text(
                        f"2\nH2 in bohr\nH 0 0 -0.7\nH 0 0 {0.7 + sign * step:.14f}\n"
                    )
                    energies.append(
                        calculate(cli, xyz, method, "cuda")["energy_hartree"]
                    )
                error = abs(forces[1, 2] + (energies[1] - energies[0]) / (2 * step))
                errors.append(error)
                assert error < 2e-5, (method, step, error)
            print(
                json.dumps(
                    {
                        "method": method,
                        "energy_error": energy_error,
                        "force_error": force_error,
                        "finite_difference_errors": errors,
                    }
                ),
                flush=True,
            )
        xyz.write_text("3\nWater in bohr\nO 0 0 0\nH 1.8 0 0\nH -0.45 1.75 0\n")
        for method in ("pbe-d3(bj)-atm-rks", "pbe0-d4(bj-eeq-atm)-rks"):
            gpu = calculate(cli, xyz, method, "cuda", True)
            cpu = calculate(cli, xyz, method, "cpu", True)
            energy_error = abs(gpu["energy_hartree"] - cpu["energy_hartree"])
            force_error = float(
                np.max(
                    np.abs(
                        np.asarray(gpu["forces_hartree_per_bohr"])
                        - cpu["forces_hartree_per_bohr"]
                    )
                )
            )
            assert energy_error < 2e-8 and force_error < 2e-7, (
                method,
                energy_error,
                force_error,
            )
            print(
                json.dumps(
                    {
                        "method": method,
                        "system": "water",
                        "energy_error": energy_error,
                        "force_error": force_error,
                    }
                ),
                flush=True,
            )
    print("PASS complete named CUDA CLI endpoints", flush=True)


if __name__ == "__main__":
    main()
