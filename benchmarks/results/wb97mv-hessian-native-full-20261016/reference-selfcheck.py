"""Diagnostic only: compare installed analytic reference with its own primal."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from benchmarks._retention import raw_output_path


def make_engine(atoms: list, nlc: bool):
    from benchmarks._gpu4pyscf_grid import preserve_reference_grid_order
    from benchmarks.readme_wb97mv import reference_engine
    from generativeqc import GridSpec

    engine = reference_engine(atoms, "sto-3g", GridSpec(8, 4, 8))
    preserve_reference_grid_order(engine)
    if not nlc:
        engine.nlc = 0
    engine.conv_tol = 1e-13
    engine.conv_tol_grad = 1e-11
    engine.conv_tol_cpscf = 1e-11
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 240
    assert bool(engine.do_nlc()) is nlc
    energy = float(engine.kernel())
    if not engine.converged:
        raise RuntimeError("Reference primal SCF did not converge")
    return engine, energy


def run(output: Path, report: dict, operation) -> None:
    """Reserve a new journal before numerical work and retain its descriptor."""
    output = raw_output_path(output)
    if output.is_symlink() or output.parent.is_symlink():
        raise FileExistsError("diagnostic output and its parent cannot be symlinks")
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.open("x+", encoding="utf-8")

    def save():
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        journal.seek(0)
        journal.write(serialized)
        journal.truncate()
        journal.flush()

    failure = None
    try:
        report["status"] = "running"
        save()
        operation(report, save)
        report["status"] = "completed"
        save()
    except BaseException as error:
        failure = error
        try:
            report.update(status="failed", failure=f"{type(error).__name__}: {error}")
            save()
        except BaseException:
            # A broken failure journal must not replace the numerical/I/O error.
            pass
        raise
    finally:
        try:
            journal.close()
        except BaseException:
            # Cleanup failures still surface when there is no earlier failure.
            if failure is None:
                raise


def collect(report: dict, save) -> None:
    import cupy as cp

    from benchmarks.compare_wb97mv_hessian_gpu4pyscf import CASES

    for name in ("h2", "water"):
        for nlc in (True, False):
            atoms = CASES[name]["atoms"]
            engine, origin_energy = make_engine(atoms, nlc)
            hessian = engine.Hessian()
            hessian.grid_response = True
            matrix = cp.asnumpy(hessian.kernel()).transpose(0, 2, 1, 3).reshape(
                len(atoms) * 3, -1
            )
            row = {
                "nlc": nlc,
                "method": "WB97M_V" if nlc else "WB97M_V with NLC explicitly disabled (control only)",
                "origin_energy": origin_energy,
                "analytic_hzz": float(matrix[-1, -1]),
                "samples": [],
                "extrapolants": [],
            }
            report["cases"][f"{name}/nlc={nlc}"] = row
            save()
            del hessian, engine
            for step in (0.02, 0.01, 0.004, 0.002):
                energies = []
                for sign in (-1, 1):
                    displaced = [(element, list(xyz)) for element, xyz in atoms]
                    displaced[-1][1][-1] += sign * step
                    engine, energy = make_engine(displaced, nlc)
                    energies.append(energy)
                    del engine
                curvature = (sum(energies) - 2 * origin_energy) / step**2
                row["samples"].append(
                    {"step": step, "energies": energies, "curvature": curvature}
                )
                save()
            for offset in (0, 2):
                coarse, fine = row["samples"][offset : offset + 2]
                value = (4 * fine["curvature"] - coarse["curvature"]) / 3
                row["extrapolants"].append(
                    {
                        "steps": [coarse["step"], fine["step"]],
                        "curvature": value,
                        "analytic_difference": row["analytic_hzz"] - value,
                    }
                )
            save()
            print(name, nlc, json.dumps(row["extrapolants"]), flush=True)


def main() -> None:
    import cupy as cp
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks as gradient_rks
    from gpu4pyscf.hessian import rks as hessian_rks

    if not os.environ.get("SLURM_JOB_ID") or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("Requires a finite single-visible-GPU Slurm step")
    for owner in (numint, gradient_rks, hessian_rks):
        owner.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-8
    output = Path(".artifacts/reference-selfcheck.json")
    report = {
        "claim": "diagnostic own-primal consistency; no comparator/tolerance modification",
        "source": "84abba9dc59359767c06ad188cbff815b0ac7268",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "grid": [8, 4, 8],
        "coordinate": "last atom z / Bohr",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_hessian_sha256": hashlib.sha256(
            Path(hessian_rks.__file__).read_bytes()
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("gpu4pyscf-cuda12x", "pyscf", "cupy-cuda12x")
        },
        "cases": {},
    }

    run(output, report, collect)
    print(output, flush=True)


if __name__ == "__main__":
    main()
