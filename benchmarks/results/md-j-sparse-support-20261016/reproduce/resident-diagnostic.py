"""Source-matched resident raw-J experiment, never a complete SCF endpoint."""

from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
from time import perf_counter

import cupy as cp
import numpy as np
from generativeqc.fock import FockBuildSpec, FockTerm
from generativeqc_compiler.dft import NativeAO
from md_j_resident_probe import ResidentMdJProbe

from benchmarks._md_j_reference import reference_molecule
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_hf_scaling import scaling_cases


def main():
    assert os.environ.get("SLURM_JOB_ID")
    root = Path("/data/jzzeng/gqc-direct-md-20261016")
    directory = root / "results" / os.environ["DIAGNOSTIC_ARM"] / os.environ.get("DIAGNOSTIC_LABEL", "diagnostic")
    directory.mkdir(parents=True, exist_ok=True)
    sizes = [int(value) for value in os.environ.get("DIAGNOSTIC_SIZES", "96,48").split(",")]
    repeats = int(os.environ.get("DIAGNOSTIC_REPEATS", "3"))
    budget = int(os.environ.get("DIAGNOSTIC_BUDGET_BYTES", str(1 << 30)))
    screening = float(os.environ.get("DIAGNOSTIC_SCREENING", "1e-12"))
    payload = {"kind": "resident-raw-J-only-not-SCF", "rows": [], "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    npy = directory / "arrays"
    npy.mkdir(exist_ok=True)
    spec = replace(FockBuildSpec.hf("restricted", derivative_order=0), exchange=FockTerm(False))
    if os.environ.get("DIAGNOSTIC_PROFILE") == "1":
        cp.cuda.runtime.profilerStart()
    for size in sizes:
        case = scaling_cases()[f"water-{size}"]
        basis, _ = load_comparison_basis(Path("python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json"), case, role="orbital", compute_forces=True)
        with NativeAO(case.atoms, basis, representation="spherical") as ao:
            generator = np.random.default_rng(240616)
            dense = generator.normal(size=(ao.nao, ao.nao)) * 0.01
            dense = (dense + dense.T) / 2
            molecule, scales = reference_molecule(ao)
            from pyscf.scf import hf
            atomic = hf.init_guess_by_minao(molecule) / np.outer(scales, scales)
            for density_name, density in (("dense", dense), ("atomic", atomic)):
                reference = None
                for mode in ("normal", "md"):
                    if mode == "normal":
                        os.environ["GENERATIVEQC_DISABLE_MD_J"] = "1"
                    else:
                        os.environ.pop("GENERATIVEQC_DISABLE_MD_J", None)
                    os.environ.pop("GENERATIVEQC_MD_J_WORK_COUNTS", None)
                    started = perf_counter()
                    with ResidentMdJProbe(ao, spec, screening_tolerance=screening, device_budget_bytes=budget) as plan:
                        preparation = perf_counter() - started
                        row = {"atoms": size, "nao": ao.nao, "density": density_name, "mode": mode, "prepare_seconds": preparation, "device_budget_bytes": budget, "screening": screening, **plan.diagnostics}
                        if mode == "md" and not row["direct_schedule"].startswith("md-j-"):
                            row["status"] = "not-admitted"
                            payload["rows"].append(row)
                            print(json.dumps(row), flush=True)
                            continue
                        wall, device, errors, gates = [], [], [], []
                        cp.cuda.nvtx.RangePush(f"{size}/{density_name}/{mode}")
                        for repetition in range(repeats):
                            started = perf_counter()
                            values = plan.evaluate(density).coulomb
                            wall.append(perf_counter() - started)
                            device.append(plan.last_device_seconds)
                            if mode == "normal":
                                if reference is None:
                                    reference = values.copy()
                            errors.append(float(np.max(np.abs(values-reference))))
                            gates.append(bool(np.all(np.abs(values-reference) <= 3e-11 + 3e-11*np.abs(reference))))
                        cp.cuda.nvtx.RangePop()
                        np.save(npy / f"{size}-{density_name}-{mode}.npy", values, allow_pickle=False)
                        row.update(status="complete" if all(gates) else "accuracy-failed", wall_seconds=wall, device_seconds=device, device_median=statistics.median(device), maximum_error=max(errors), maximum_errors=errors, accuracy_gates=gates)
                        payload["rows"].append(row)
                        print(json.dumps(row), flush=True)
                        (directory / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
                if os.environ.get("DIAGNOSTIC_PROFILE") != "1":
                    os.environ["GENERATIVEQC_MD_J_WORK_COUNTS"] = "1"
                    os.environ.pop("GENERATIVEQC_DISABLE_MD_J", None)
                    with ResidentMdJProbe(ao, spec, screening_tolerance=screening, device_budget_bytes=budget) as counted:
                        if counted.diagnostics["direct_schedule"].startswith("md-j-"):
                            values = counted.evaluate(density).coulomb
                            error = float(np.max(np.abs(values-reference)))
                            print(json.dumps({"diagnostic_counted_call": True, "atoms": size, "density": density_name, "maximum_error": error}), flush=True)
                    os.environ.pop("GENERATIVEQC_MD_J_WORK_COUNTS", None)
    if os.environ.get("DIAGNOSTIC_PROFILE") == "1":
        cp.cuda.runtime.profilerStop()
    (directory / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if any(row["status"] == "accuracy-failed" for row in payload["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
