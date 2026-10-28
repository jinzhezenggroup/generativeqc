"""Python is the independent harness; each installed process has no tool PATH."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/python")
from test_libxc_automatic_point_native import NAMES

root = Path.cwd()
cli = root / "install/bin/generativeqc"
out = root / "evidence/installed"
out.mkdir(exist_ok=True)
xyz = out / "h4.xyz"
xyz.write_text("4\nBohr\nH 0 0 0\nH 0.1 0.2 1.4\nH 1.6 0.4 0.3\nH 1.8 0.3 1.8\n")
env = {
    **os.environ,
    "PATH": str(out / "no-toolchain"),
    "CUDA_DISABLE_PTX_JIT": "1",
    "GENERATIVEQC_PROFILE": "off",
}
# The installed executable links through its $ORIGIN/../lib RUNPATH. Do not
# retain the Python harness's build-tree override in the installed process.
env.pop("GENERATIVEQC_LIBRARY", None)
identities = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in (
        ("cli", cli),
        ("library", root / "install/lib/libgenerativeqc.so"),
    )
}
records = []
for spins in (1, 2):
    for name in NAMES:
        args = [
            str(cli),
            "run",
            str(xyz),
            "--units",
            "bohr",
            "--method",
            f"libxc-{'rks' if spins == 1 else 'uks'}:{name}",
            "--backend",
            "cuda",
            "--precision",
            "fp64",
            "--charge",
            str(spins - 1),
            "--multiplicity",
            str(spins),
            "--forces",
            "--initial-guess",
            "none",
            "--max-iterations",
            "300",
            "--energy-tolerance",
            "1e-13",
            "--density-tolerance",
            "1e-12",
            "--screening-tolerance",
            "1e-12",
            "--grid-radial-points",
            "24",
            "--grid-polar-points",
            "8",
            "--grid-azimuth-points",
            "16",
            "--json",
        ]
        started = time.perf_counter()
        p = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=120, check=False
        )
        seconds = time.perf_counter() - started
        assert p.returncode == 0, (name, spins, p.stderr)
        result = json.loads(p.stdout)
        expected = json.loads(
            (
                root / f"evidence/forces/{name}-{spins}-True-False-False-h4.json"
            ).read_text()
        )["rows"][0]
        import numpy as np

        assert result["backend"] == "cuda"
        np.testing.assert_allclose(
            result["energy_hartree"],
            sum(expected["scf"]["components"].values()),
            atol=3e-9,
            rtol=0,
        )
        np.testing.assert_allclose(
            result["forces_hartree_per_bohr"], expected["forces"], atol=3e-8, rtol=0
        )
        records.append(
            {"name": name, "spins": spins, "seconds": seconds, "result": result}
        )
        (out / "results.json").write_text(
            json.dumps(
                {
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "artifacts": identities,
                    "rows": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(name, spins, "passed", round(seconds, 3), flush=True)
# A structurally registered provider omitted from this bounded package remains
# unavailable. The installed executable must not attempt on-demand generation.
args[args.index("--method") + 1] = "libxc-rks:LDA_C_RC04"
args[args.index("--charge") + 1] = "0"
args[args.index("--multiplicity") + 1] = "1"
p = subprocess.run(
    args, env=env, capture_output=True, text=True, timeout=120, check=False
)
assert p.returncode != 0, (p.stdout, p.stderr)
(out / "omitted-provider.txt").write_text(p.stderr)
print("omitted provider rejected:", p.stderr.strip(), flush=True)
