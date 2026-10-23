"""Compare conservative block screening on complete strict-FP64 Direct E/F.

Run each arm in a fresh process under finite Slurm. Timings include native
initialization in cold and retain the owner's own frozen cold density in warm.
Independent tighter GPU4PySCF references run only after all native timings.
This is a matched-default-grid regression check, not grid convergence evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import (
    METHODS,
    digest,
    native_calculator,
    native_settings,
    orbital_basis,
    reference_r2scan3c_correction,
)


def reference(atoms: list, method: str) -> dict:
    """Solve independently at tighter SCF/screening tolerances on the same grid."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from gpu4pyscf.scf import hf
    from pyscf import gto, lib

    from benchmarks.readme_wb97mv import configure_reference_full_fock

    lib.num_threads(1)
    mol = gto.M(
        atom=atoms, unit="Bohr", basis=orbital_basis(method), cart=False, verbose=0
    )
    engine = hf.RHF(mol) if method == "hf" else rks.RKS(mol)
    if method != "hf":
        engine.xc = METHODS[method][1]
    engine.conv_tol = 1e-12
    engine.conv_tol_grad = 1e-9
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 150
    configure_reference_full_fock(engine)
    energy = float(engine.kernel())
    if not engine.converged:
        raise RuntimeError("independent reference did not converge")
    gradient = engine.nuc_grad_method()
    if method != "hf":
        gradient.grid_response = True
    forces = -cp.asnumpy(gradient.kernel())
    if method == "r2scan-3c":
        correction_energy, correction_gradient = reference_r2scan3c_correction(atoms)()
        energy += float(correction_energy)
        forces -= correction_gradient
    return {
        "energy": energy,
        "forces": forces.tolist(),
        "iterations": int(engine.cycles),
    }


def main() -> None:
    """Record every sample/failure, native image, controls and observable gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("pbe0-96", "r2scan-3c-96", "c7", "pbe0-12"), required=True
    )
    parser.add_argument("--fp32", type=int, choices=(0, 1), required=True)
    parser.add_argument("--qm7", type=Path)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--moved", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        parser.error("GPU execution requires a finite Slurm allocation")
    if args.output.exists() or args.warm_repeats < 0:
        parser.error("output must be new and warm repeats nonnegative")
    os.environ["GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING"] = str(args.fp32)
    import cupy as cp
    import numpy as np

    from benchmarks.readme_cbs import geometry
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_method_endpoints import binary_identity

    if args.case == "c7":
        if args.qm7 is None:
            parser.error("C7 requires the pinned --qm7 dataset")
        atoms = geometry(args.qm7, 7)
        method = "hf"
    else:
        method, count = args.case.rsplit("-", 1)
        atoms = scaling_cases()[f"water-{count}"].atoms
    moved = np.asarray([xyz for _, xyz in atoms], dtype=float)
    moved[1, 2] += 0.02
    geometries = {"original": atoms}
    if args.moved:
        geometries["moved"] = [
            (symbol, xyz.tolist())
            for (symbol, _), xyz in zip(atoms, moved, strict=True)
        ]
    record: dict[str, Any] = {
        "schema": "generativeqc.direct-screening-endpoint.v1",
        "case": args.case,
        "fp32_block_screening": bool(args.fp32),
        "geometries_bohr": geometries,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "pci_bus_id": cp.cuda.Device(0).pci_bus_id,
        "driver_sha256": digest(Path(__file__)),
        "protocol": "Direct Standard FP64 E/F; cold includes constructor+prepare; frozen same-owner warm; optional changed geometry; independent oracle after timing",
        "gates": {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6},
        "samples": [],
        "status": "running",
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    calculator = owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        calculator = native_calculator(method, "standard", "direct", Path("."))
        owner = calculator.prepare_batch([atoms], warm_start=True)

        def execute(
            phase: str, key: str, coordinates: Any = None, cold: float | None = None
        ) -> None:
            cp.cuda.runtime.deviceSynchronize()
            tick = time.perf_counter() if cold is None else cold
            cp.cuda.nvtx.RangePush(phase)
            try:
                item = owner.execute(
                    coordinates, strict=True, properties=("energy", "forces")
                ).items[0]
                cp.cuda.runtime.deviceSynchronize()
            finally:
                cp.cuda.nvtx.RangePop()
            seconds = time.perf_counter() - tick
            if not item.converged or item.executed_backend != "cuda":
                raise RuntimeError("native CUDA endpoint did not converge")
            if method == "r2scan-3c" and not (item.dispersion and item.dispersion.ok):
                raise RuntimeError("incomplete r2SCAN-3c correction endpoint")
            row = {
                "phase": phase,
                "geometry": key,
                "seconds": seconds,
                "energy": float(item.energy),
                "forces": np.asarray(item.forces).tolist(),
                "iterations": item.iterations,
                "fock_builds": item.fock_builds,
                "precision": item.precision,
                "settings": native_settings(calculator, item),
                "ks_diagnostic": item.ks_diagnostic.to_payload()
                if item.ks_diagnostic
                else None,
                "warm_start_used": item.warm_start_used,
                "warm_start_fallback": item.warm_start_fallback,
            }
            record["samples"].append(row)
            save()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("phase", "seconds", "iterations", "fock_builds")
                    }
                ),
                flush=True,
            )

        execute("cold", "original", cold=start)
        record["native_build"] = binary_identity(calculator)
        owner.set_warm_start_updates(False)
        for _ in range(args.warm_repeats):
            execute("warm", "original")
        if args.moved:
            owner.set_warm_start_updates(True)
            execute("moved", "moved", [moved])
        if args.oracle:
            record["references"] = {
                key: reference(value, method) for key, value in geometries.items()
            }
            for row in record["samples"]:
                oracle = record["references"][row["geometry"]]
                error = np.asarray(row["forces"]) - np.asarray(oracle["forces"])
                row["errors"] = {
                    "energy_per_atom": abs(row["energy"] - oracle["energy"])
                    / len(atoms),
                    "force_max": float(np.abs(error).max()),
                    "force_rms": float(np.sqrt(np.mean(error**2))),
                }
                row["oracle_pass"] = all(
                    row["errors"][key] <= gate for key, gate in record["gates"].items()
                )
            if not all(row["oracle_pass"] for row in record["samples"]):
                raise RuntimeError("independent observable gate failed")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        if owner is not None:
            owner.close()
        if calculator is not None:
            calculator.clear_cache()


if __name__ == "__main__":
    main()
