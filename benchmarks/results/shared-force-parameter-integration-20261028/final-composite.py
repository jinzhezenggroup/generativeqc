"""Exercise integrated external derivative callbacks through the native CLI."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests/native")
from test_cli_cuda_named import calculate as _calculate


def calculate(
    cli: Path, xyz: Path, method: str, backend: str, forces: bool = False
) -> dict:
    """Keep native command diagnostics visible when a qualification fails."""
    try:
        return _calculate(cli, xyz, method, backend, forces)
    except subprocess.CalledProcessError as e:
        print(e.stdout, e.stderr, flush=True)
        raise


assert os.environ.get("SLURM_JOB_ID")
cli = Path("build/integration/generativeqc").resolve()
rows = []
with tempfile.TemporaryDirectory() as t:
    xyz = Path(t) / "h2.xyz"
    for method in ("pbe-d3(bj)-rks", "pbe-d3(0)-uks", "wb97m-v-rks", "wb97m-v-uks"):
        xyz.write_text("2\nH2 bohr\nH 0 0 -0.7\nH 0 0 0.7\n")
        gpu = calculate(cli, xyz, method, "cuda", True)
        cpu = calculate(cli, xyz, method, "cpu", False)
        energy_error = abs(gpu["energy_hartree"] - cpu["energy_hartree"])
        force = np.array(gpu["forces_hartree_per_bohr"])
        assert energy_error < 2e-8, (method, energy_error)
        if method.startswith("wb97m-v"):
            reference = (
                -1.150312466420175 if method.endswith("rks") else -0.528250800668613
            )
            assert abs(gpu["energy_hartree"] - reference) < 2e-8
        errors = []
        for step in (3e-4, 1e-4):
            es = []
            cpu_es = []
            for sign in (-1, 1):
                xyz.write_text(
                    f"2\nH2 bohr\nH 0 0 -0.7\nH 0 0 {0.7 + sign * step:.14f}\n"
                )
                es.append(calculate(cli, xyz, method, "cuda")["energy_hartree"])
                cpu_es.append(calculate(cli, xyz, method, "cpu")["energy_hartree"])
            error = abs(force[1, 2] + (es[1] - es[0]) / (2 * step))
            assert error < 2e-5, (method, error)
            cpu_error = abs(force[1, 2] + (cpu_es[1] - cpu_es[0]) / (2 * step))
            assert cpu_error < 2e-5, (method, cpu_error)
            errors.append({"gpu": error, "cpu": cpu_error})
        rows.append(
            {
                "method": method,
                "energy_error": energy_error,
                "finite_difference_errors": errors,
                "result": gpu,
            }
        )
        print(
            json.dumps({k: v for k, v in rows[-1].items() if k != "result"}), flush=True
        )
out = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
    "rows": rows,
}
Path(".artifacts/final-validation/composite-cli.json").write_text(json.dumps(out, indent=2) + "\n")
