"""Acquire conventional all-electron PySCF anchors for preregistered barriers."""

import argparse
import faulthandler
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from protocol import ROOT, TIME_LIMIT_SECONDS, append, basis, endpoint, load, sha

THREADS = 8
WORKERS = 2
EXPECTED_CPU = "Intel(R) Xeon(R) CPU Max 9462"
MEMORY_MB = 40_000


def singlepoint(system: dict, cardinal: int) -> dict:
    from pyscf import cc, gto, lib, mp, scf

    lib.num_threads(THREADS)
    start = time.perf_counter()
    row = {"basis": basis(cardinal)}
    try:
        mol = gto.M(
            atom=system["bohr"],
            basis=row["basis"],
            unit="Bohr",
            charge=system["charge"],
            spin=system["multiplicity"] - 1,
            verbose=0,
            max_memory=MEMORY_MB,
        )
        hf = scf.RHF(mol)
        hf.init_guess = "minao"
        hf.conv_tol = 1e-12
        hf.conv_tol_grad = 1e-9
        hf.max_cycle = 200
        hf.kernel()
        row.update(
            nao=mol.nao_nr(),
            nelectron=mol.nelectron,
            scf_cycles=int(hf.cycles),
            hf_hartree=float(hf.e_tot),
            scf_converged=bool(hf.converged),
        )
        if not hf.converged:
            raise RuntimeError("conventional RHF did not converge")
        solver = mp.MP2(hf, frozen=0)
        emp2, _ = solver.kernel(with_t2=False)
        row["mp2_correlation"] = float(emp2)
        solver = cc.CCSD(hf, frozen=0)
        solver.max_cycle = 150
        solver.conv_tol = 1e-10
        solver.conv_tol_normt = 1e-8
        solver.kernel()
        row.update(
            ccsd_cycles=int(solver.cycles),
            ccsd_converged=bool(solver.converged),
            ccsd_correlation=float(solver.e_corr),
        )
        if not solver.converged:
            raise RuntimeError("conventional CCSD did not converge")
        row["triples"] = float(solver.ccsd_t())
        row["ccsdt_hartree"] = math.fsum(
            row[key] for key in ("hf_hartree", "ccsd_correlation", "triples")
        )
        if not all(
            math.isfinite(row[key]) for key in ("mp2_correlation", "ccsdt_hartree")
        ):
            raise ValueError("nonfinite conventional correlation energy")
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row.update(
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    row["wall_seconds"] = time.perf_counter() - start
    return row


def anchor_keys(data: dict, dataset: str) -> list[str]:
    return sorted(
        {
            key
            for row in data["barriers"]
            if row["set"] == dataset
            and row["index"] in data["canonical_anchors"][dataset]
            for key in row["stoichiometry"]
        },
        key=lambda key: (len(data["systems"][key]["bohr"]), key),
    )


def campaign(data: dict, dataset: str, output: Path) -> None:
    keys = anchor_keys(data, dataset)
    tasks = [(key, n) for n in (2, 3, 4) for key in keys]
    cpu = {
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    }
    output.mkdir()
    metadata = {
        "dataset": dataset,
        "host": platform.node(),
        "cpu_models": sorted(cpu),
        "affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("pyscf", "numpy", "scipy", "h5py")
        },
        "inputs_sha256": sha(ROOT / "inputs.json"),
        "driver_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(ROOT / "protocol.py"),
        "systems": keys,
        "threads_per_process": THREADS,
        "worker_count": WORKERS,
        "pyscf_memory_mb_per_process": MEMORY_MB,
        "endpoint_limit_seconds": TIME_LIMIT_SECONDS,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "GQC_QZ_JOB_NAME",
                "GQC_QZ_PROJECT",
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
            )
        },
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if cpu != {EXPECTED_CPU}:
        raise RuntimeError(
            f"CPU model differs from the reference target: {sorted(cpu)}"
        )
    if int(os.environ["SLURM_CPUS_PER_TASK"]) < THREADS * WORKERS:
        raise RuntimeError("allocation has insufficient task CPUs")

    def run(task: tuple[str, int]) -> None:
        key, cardinal = task
        with (
            tempfile.TemporaryDirectory(dir=output) as tmp,
            (output / f"{key.split('/')[1]}-{cardinal}.jsonl").open("x") as journal,
            (output / f"{key.split('/')[1]}-{cardinal}.txt").open("x") as log,
        ):
            request = {"system": key, "cardinal": cardinal}
            append(journal, {**request, "status": "started", "utc_unix": time.time()})
            row = endpoint(
                Path(__file__), [key, str(cardinal)], Path(tmp) / "point.json", log
            )
            append(journal, {**request, **row, "finished_utc_unix": time.time()})
            print(key, cardinal, row["status"], row["process_wall_seconds"], flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run, tasks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("campaign")
    run.add_argument("dataset", choices=("CRBH20",))
    run.add_argument("output", type=Path)
    point = commands.add_parser("point")
    point.add_argument("system")
    point.add_argument("cardinal", type=int, choices=(2, 3, 4))
    point.add_argument("output", type=Path)
    options = parser.parse_args()
    data = load(ROOT / "inputs.json")
    if options.command == "campaign":
        campaign(data, options.dataset, options.output)
    else:
        faulthandler.dump_traceback_later(600, repeat=True)
        try:
            row = singlepoint(data["systems"][options.system], options.cardinal)
        finally:
            faulthandler.cancel_dump_traceback_later()
        with options.output.open("x") as stream:
            append(stream, row)


if __name__ == "__main__":
    main()
