"""Independent conventional PySCF HF/MP2/CCSD(T) canonical-basis anchors."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_gpu import ROOT, append, basis_name, bohr_geometry, sha

# Complete reactions selected before running for small-basis anchors across
# neutral doublet transfer, association, isomerization and anionic substitution.
PAIRS = {"HTBH38": (3, 9, 17, 23), "NHTBH38": (3, 13, 29, 31, 37)}
THREADS = 4
LIMIT_SECONDS = 480
EXPECTED_CPU = "Xeon(R) CPU Max 9462"


def reference_endpoint(system: dict, cardinal: int) -> dict:
    from pyscf import cc, gto, lib, mp, scf

    lib.num_threads(THREADS)
    wall = time.perf_counter()
    meta = {"orbital_basis": basis_name(system, cardinal)}
    try:
        molecule = gto.M(
            atom=bohr_geometry(system),
            basis=meta["orbital_basis"],
            unit="Bohr",
            charge=system["charge"],
            spin=system["multiplicity"] - 1,
            verbose=0,
            max_memory=11000,
        )
        reference = scf.RHF(molecule) if not molecule.spin else scf.UHF(molecule)
        reference.init_guess = "minao"
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.max_cycle = 200
        reference.kernel()
        meta.update(
            nao=molecule.nao_nr(),
            nelectron=molecule.nelectron,
            scf_converged=bool(reference.converged),
            scf_cycles=int(reference.cycles),
            hf_hartree=float(reference.e_tot),
            spin_square=float(reference.spin_square()[0]),
            orbital_energies=reference.mo_energy.tolist(),
            orbital_occupations=reference.mo_occ.tolist(),
        )
        if not reference.converged:
            raise RuntimeError("conventional SCF did not converge")
        if molecule.nelectron == 1:
            # An exact one-electron state has no electron-pair correlation.
            meta.update(mp2_correlation=0.0, ccsd_correlation=0.0, triples=0.0)
        else:
            perturbation = mp.MP2(reference, frozen=0)
            e_mp2, _ = perturbation.kernel(with_t2=False)
            meta["mp2_correlation"] = float(e_mp2)
            solver = cc.CCSD(reference, frozen=0)
            solver.max_cycle = 150
            solver.conv_tol = 1e-10
            solver.conv_tol_normt = 1e-8
            solver.kernel()
            meta.update(
                ccsd_converged=bool(solver.converged),
                ccsd_cycles=int(solver.cycles),
                ccsd_correlation=float(solver.e_corr),
            )
            if not solver.converged:
                raise RuntimeError("conventional CCSD did not converge")
            meta["triples"] = float(solver.ccsd_t())
        meta["ccsdt_hartree"] = math.fsum(
            meta[key] for key in ("hf_hartree", "ccsd_correlation", "triples")
        )
        if not all(
            math.isfinite(meta[key]) for key in ("ccsdt_hartree", "mp2_correlation")
        ):
            raise ValueError("nonfinite conventional correlation energy")
        meta["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        # Preserve every scientific endpoint failure and continue the cohort.
        meta.update(
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    meta["wall_seconds"] = time.perf_counter() - wall
    return meta


def selected_keys(data: dict) -> list[str]:
    keys = set()
    for row in data["barriers"]:
        index = row["index"]
        if (index if index % 2 else index - 1) in PAIRS[row["set"]]:
            keys.update((row["transition"], *row["side"]))
    return sorted(keys, key=lambda key: (len(data["systems"][key]["angstrom"]), key))


def run_point(task: tuple[str, int, Path]) -> dict:
    key, cardinal, tmp = task
    path = tmp / f"{key.replace('/', '_')}-{cardinal}.json"
    t0 = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, __file__, "point", key, str(cardinal), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=LIMIT_SECONDS,
            check=False,
            text=True,
        )
        if process.returncode:
            result = {"status": "process_failed", "returncode": process.returncode}
        else:
            result = json.loads(path.read_text())
        if process.stdout:
            result["process_output"] = process.stdout
    except subprocess.TimeoutExpired as exc:
        result = {
            "status": "timeout",
            "limit_seconds": LIMIT_SECONDS,
            "process_output": (exc.stdout or b"").decode(errors="replace"),
        }
    return {
        "system": key,
        "cardinal": cardinal,
        **result,
        "process_wall_seconds": time.perf_counter() - t0,
    }


def campaign(output: Path, data: dict) -> None:
    output.mkdir()
    cpu = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        }
    )
    affinity = sorted(os.sched_getaffinity(0))
    metadata = {
        "cpu": cpu,
        "affinity": affinity,
        "host": platform.node(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("pyscf", "numpy", "scipy", "h5py")
        },
        "job": {
            k: os.environ.get(k)
            for k in (
                "GQC_QZ_JOB_NAME",
                "GQC_QZ_PROJECT",
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
            )
        },
        "inputs_sha256": sha(ROOT / "inputs.json"),
        "driver_sha256": sha(__file__),
        "shared_driver_sha256": sha(ROOT / "run_gpu.py"),
        "selected_pairs": PAIRS,
        "threads_per_process": THREADS,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not cpu or any(EXPECTED_CPU not in name for name in cpu):
        raise RuntimeError(
            f"allocation CPU model differs from validation target: {cpu}"
        )
    if int(os.environ["SLURM_CPUS_PER_TASK"]) < 2 * THREADS:
        raise RuntimeError("allocation has insufficient CPU threads")
    with (
        ThreadPoolExecutor(max_workers=2) as workers,
        tempfile.TemporaryDirectory(dir=output) as temporary,
        (output / "records.jsonl").open("x") as log,
    ):
        for cardinal in (2, 3, 4):
            tasks = [(key, cardinal, Path(temporary)) for key in selected_keys(data)]
            for key, _, _ in tasks:
                append(log, {"system": key, "cardinal": cardinal, "status": "planned"})
            for row in workers.map(run_point, tasks):
                append(log, row)
                print(row["system"], cardinal, row["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("campaign")
    run.add_argument("output", type=Path)
    point = commands.add_parser("point")
    point.add_argument("system")
    point.add_argument("cardinal", type=int, choices=(2, 3, 4))
    point.add_argument("output", type=Path)
    options = parser.parse_args()
    data = json.loads((ROOT / "inputs.json").read_text())
    if options.command == "point":
        record = reference_endpoint(data["systems"][options.system], options.cardinal)
        with options.output.open("x") as stream:
            append(stream, record)
    else:
        campaign(options.output, data)
