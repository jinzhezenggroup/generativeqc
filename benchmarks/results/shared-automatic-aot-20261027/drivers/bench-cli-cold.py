"""Paired installed-process cold SCF+E+F, including loader/context and teardown."""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

root = Path.cwd()
scope = os.environ["PACKAGE_SCOPE"]
out = root / f"evidence/cli-cold-{scope}"
out.mkdir(exist_ok=True)
if "--analyze" not in sys.argv:
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
geometries = {
    "h4": "4\nBohr\nH 0 0 0\nH 0.1 0.2 1.4\nH 1.6 0.4 0.3\nH 1.8 0.3 1.8\n",
    "water": "3\nBohr\nO 0 0 0\nH 0.2 1.5 1.1\nH -0.3 -1.4 1.0\n",
}
for name, xyz in geometries.items():
    (out / f"{name}.xyz").write_text(xyz)
cases = [
    (system, name, spin)
    for system, names in (
        ("h4", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA")),
        ("water", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_C_TPSS")),
    )
    for name in names
    for spin in (1, 2)
]
order = (
    "baseline",
    "candidate",
    "candidate",
    "baseline",
    "candidate",
    "baseline",
    "baseline",
    "candidate",
)
prefixes = {
    "baseline": Path("/home/jzzeng/tmp/qc-1123-baseline-20261027") / f"install-{scope}",
    "candidate": root / f"install-{scope}",
}
analyze_only = "--analyze" in sys.argv
records = (
    json.loads((out / "samples.json").read_text())["records"] if analyze_only else []
)
for index, arm in enumerate(() if analyze_only else order):
    prefix = prefixes[arm]
    cli = prefix / "bin/generativeqc"
    library = (prefix / "lib/libgenerativeqc.so").resolve()
    identities = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (cli, library)
    }
    env = {
        **os.environ,
        "PATH": str(out / "no-toolchain"),
        "CUDA_DISABLE_PTX_JIT": "1",
        "LD_LIBRARY_PATH": str(library.parent) + ":/group/software/cuda-12.9.1/lib64",
        "GENERATIVEQC_PROFILE": "off",
    }
    loader = subprocess.check_output(
        ["/lib64/ld-linux-x86-64.so.2", "--list", str(cli)], env=env, text=True
    )
    paths = [
        line.split("=>", 1)[1].strip().split()[0]
        for line in loader.splitlines()
        if "libgenerativeqc.so" in line and "=>" in line
    ]
    assert len(paths) == 1 and Path(paths[0]).resolve() == library
    rows = []
    for system, name, spin in cases:
        charge, multiplicity = (spin - 1, spin) if system == "h4" else (0, 1)
        method = f"libxc-{'rks' if spin == 1 else 'uks'}:{name}"
        args = [
            str(cli),
            "run",
            str(out / f"{system}.xyz"),
            "--units",
            "bohr",
            "--method",
            method,
            "--basis",
            "sto-3g",
            "--backend",
            "cuda",
            "--precision",
            "fp64",
            "--charge",
            str(charge),
            "--multiplicity",
            str(multiplicity),
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
        start = time.perf_counter()
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=120
        )
        seconds = time.perf_counter() - start
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["backend"] == "cuda"
        rows.append(dict(system=system, method=method, seconds=seconds, result=payload))
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in identities.items()
    )
    records.append(dict(arm=arm, identities=identities, rows=rows))
    (out / "samples.json").write_text(
        json.dumps(
            dict(
                schema="generativeqc.installed-process-cold-comparison.v1",
                scope=scope,
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                order=order,
                no_toolchain=True,
                ptx_jit_disabled=True,
                complete=False,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(index, arm, "installed cold endpoints completed", flush=True)
gates = []
for case_index, (system, name, spin) in enumerate(cases):
    values = {arm: [] for arm in prefixes}
    anchor = records[0]["rows"][case_index]["result"]
    for record in records:
        row = record["rows"][case_index]
        actual = row["result"]
        if actual["iterations"] != anchor["iterations"]:
            # Report the extra native SCF iteration, never lower a tolerance or
            # attribute process-cold speed to an isolated force kernel.
            assert (system, name, spin) == ("h4", "MGGA_X_LTA", 2)
            assert (
                record["arm"] == "candidate"
                and actual["iterations"] == anchor["iterations"] + 1
            )
        np.testing.assert_allclose(
            actual["energy_hartree"], anchor["energy_hartree"], atol=3e-9, rtol=0
        )
        np.testing.assert_allclose(
            actual["forces_hartree_per_bohr"],
            anchor["forces_hartree_per_bohr"],
            atol=3e-8,
            rtol=0,
        )
        values[record["arm"]].append(row["seconds"])
    ratio = statistics.median(values["candidate"]) / statistics.median(
        values["baseline"]
    )
    iterations = {
        arm: [
            r["rows"][case_index]["result"]["iterations"]
            for r in records
            if r["arm"] == arm
        ]
        for arm in prefixes
    }
    gates.append(
        dict(
            system=system,
            name=name,
            spin=spin,
            samples=values,
            iterations=iterations,
            exact_iteration_parity=len(
                {n for values in iterations.values() for n in values}
            )
            == 1,
            ratio=ratio,
            passed=ratio <= 1.05,
        )
    )
data = json.loads((out / "samples.json").read_text())
data.update(
    complete=True, limit=1.05, gates=gates, passed=all(g["passed"] for g in gates)
)
(out / "samples.json").write_text(json.dumps(data, indent=2) + "\n")
print([(g["system"], g["name"], g["spin"], g["ratio"]) for g in gates], flush=True)
assert data["passed"], "installed process cold regression exceeds 5%"
