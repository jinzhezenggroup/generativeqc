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

scope = os.environ.get("PACKAGE_SCOPE", "narrow")
root = Path.cwd()
cli = root / f"install-{scope}/bin/generativeqc"
out = root / f"evidence/installed-{scope}"
out.mkdir(exist_ok=True)
library = root / f"install-{scope}/lib/libgenerativeqc.so"
identities = {
    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (cli, library)
}
names = (*NAMES, *(f"LDA_C_VWN_{i}" for i in range(1, 4))) if scope == "full" else NAMES
xyz = out / "h4.xyz"
xyz.write_text("4\nBohr\nH 0 0 0\nH 0.1 0.2 1.4\nH 1.6 0.4 0.3\nH 1.8 0.3 1.8\n")
env = {
    **os.environ,
    "PATH": str(out / "no-toolchain"),
    "CUDA_DISABLE_PTX_JIT": "1",
    "GENERATIVEQC_PROFILE": "off",
    "LD_LIBRARY_PATH": str(library.parent) + ":/group/software/cuda-12.9.1/lib64",
}
# Authenticate the actual loader choice, including caller LD_LIBRARY_PATH.
loader = subprocess.run(
    ["/lib64/ld-linux-x86-64.so.2", "--list", str(cli)],
    env=env,
    capture_output=True,
    text=True,
    check=True,
)
loaded = [
    line.split("=>", 1)[1].strip().split()[0]
    for line in loader.stdout.splitlines()
    if "libgenerativeqc.so" in line and "=>" in line
]
assert len(loaded) == 1 and Path(loaded[0]).resolve() == library.resolve(), (
    loader.stdout
)
records = []
for spins in (1, 2):
    for name in names:
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
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
        seconds = time.perf_counter() - started
        assert p.returncode == 0, (name, spins, p.stderr)
        result = json.loads(p.stdout)
        references = "alias-forces" if name not in NAMES else "forces"
        expected = json.loads(
            (
                root / f"evidence/{references}/{name}-{spins}-True-False-False-h4.json"
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
        records.append(dict(name=name, spins=spins, seconds=seconds, result=result))
        (out / "results.json").write_text(
            json.dumps(
                dict(
                    slurm_job_id=os.environ["SLURM_JOB_ID"],
                    identities=identities,
                    loaded_library=str(Path(loaded[0]).resolve()),
                    no_toolchain_path=env["PATH"],
                    ptx_jit_disabled=True,
                    rows=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(name, spins, "passed", round(seconds, 3), flush=True)
for path, digest in identities.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
if scope == "narrow":
    # A structurally registered provider omitted from this bounded package remains
    # unavailable. The installed executable must not attempt on-demand generation.
    args[args.index("--method") + 1] = "libxc-rks:LDA_C_RC04"
    args[args.index("--charge") + 1] = "0"
    args[args.index("--multiplicity") + 1] = "1"
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode != 0, (p.stdout, p.stderr)
    (out / "omitted-provider.txt").write_text(p.stderr)
    print("omitted provider rejected:", p.stderr.strip(), flush=True)
