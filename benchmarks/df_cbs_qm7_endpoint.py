"""Source-matched complete CUDA DF-MP2/DF-CBS E+F qualification on local QM7.

This driver does not download data or introduce a production oracle. It records
the supplied dataset checksum/index and independently reconverged PySCF energy
and directional-force gates outside the native endpoint timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from generativeqc import Calculator, CBSBasis, CCSDTCBSCalculator
from scipy.io import loadmat

from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/api/c_api_context.cpp",
    "src/api/c_api_batch.cpp",
    "src/api/handles.hpp",
    "include/generativeqc/generativeqc.h",
    "src/api/c_api_calculation.cpp",
    "src/methods/mp2_method.cpp",
    "src/methods/rccsd_method.cpp",
    "src/methods/rccsd_method.hpp",
    "src/methods/df_rccsdt_method.cpp",
    "src/methods/df_ccsdt_force.cu",
    "src/methods/df_ccsdt_force.hpp",
    "src/cc/df_source.hpp",
    "src/cc/df_source_cuda.cu",
    "src/cc/df_triples_cuda.cu",
    "src/scf/cuda/df_final_validation.cpp",
    "src/tensor/cuda_square_linalg.hpp",
    "src/posthf/mp2_derivative_common.cpp",
    "src/posthf/mp2_derivative_common.hpp",
    "python/generativeqc/cbs.py",
    "python/generativeqc/calculator.py",
    "python/generativeqc/_native.py",
    "python/generativeqc/basis_capabilities.py",
    "python/generativeqc_compiler/mp2/equations.py",
    "tools/generate_df_cc_source.py",
    "tools/generativeqc_validation/correlation_df_oracle.py",
    "benchmarks/df_cbs_qm7_endpoint.py",
)


def digest(path: Path) -> str:
    """Checksum actual local bytes, not an exported checkout's stale Git HEAD."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run one finite scheduler-owned molecular endpoint and retain raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm7", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--mode", choices=("mp2", "cbs"), default="cbs")
    parser.add_argument("--low", default="cc-pvdz")
    parser.add_argument("--low-fit", default="cc-pvdz-ri")
    parser.add_argument("--high", default="cc-pvtz")
    parser.add_argument("--high-fit", default="cc-pvtz-ri")
    parser.add_argument("--budget-mib", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-GPU qualification must run through srun")
    data = loadmat(args.qm7)
    if not 0 <= args.index < len(data["Z"]) or args.budget_mib <= 0:
        parser.error("dataset index and positive memory budget are required")
    selected = data["Z"][args.index] > 0
    numbers = data["Z"][args.index][selected].astype(int)
    coordinates = data["R"][args.index][selected].astype(float)
    atoms = [
        (int(number), position)
        for number, position in zip(numbers, coordinates, strict=True)
    ]
    low = basis_snapshot(args.low, numbers)
    low_fit = basis_snapshot(args.low_fit, numbers)
    budget = args.budget_mib << 20
    if args.mode == "mp2":
        owner = Calculator(
            method="mp2",
            basis=low,
            auxiliary_basis=low_fit,
            density_fitting="correlation_cuda",
            device="cuda",
            precision="fp64",
            correlation_memory_budget_bytes=budget,
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-11,
        )
    else:
        owner = CCSDTCBSCalculator(
            (CBSBasis(low, 2), CBSBasis(basis_snapshot(args.high, numbers), 3)),
            auxiliary_bases=(low_fit, basis_snapshot(args.high_fit, numbers)),
            device="cuda",
            calculator_options={"correlation_memory_budget_bytes": budget},
        )
    from pyscf import df, gto

    molecule = gto.M(atom=atoms, unit="Bohr", basis=args.low, verbose=0)
    payload = {
        "schema": "generativeqc.df-cbs-qm7-endpoint/1",
        "dataset_sha256": digest(args.qm7),
        "qm7_index_zero_based": args.index,
        "atoms_bohr": [(number, position.tolist()) for number, position in atoms],
        "basis": args.low,
        "auxiliary": args.low_fit,
        "mode": args.mode,
        "electron_count": molecule.nelectron,
        "low_nao": molecule.nao_nr(),
        "low_naux": df.make_auxmol(molecule, args.low_fit).nao_nr(),
        "budget_bytes": budget,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "library_sha256": digest(Path(os.environ["GENERATIVEQC_LIBRARY"])),
        "source_sha256": {path: digest(ROOT / path) for path in SOURCE_FILES},
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    if args.mode == "cbs":
        high_molecule = gto.M(atom=atoms, unit="Bohr", basis=args.high, verbose=0)
        payload.update(
            high_basis=args.high,
            high_auxiliary=args.high_fit,
            high_nao=high_molecule.nao_nr(),
            high_naux=df.make_auxmol(high_molecule, args.high_fit).nao_nr(),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        started = time.perf_counter()
        result = owner.singlepoint(atoms, properties=("energy", "forces"))
        payload["complete_endpoint_seconds"] = time.perf_counter() - started
        payload["native_energy"] = result.energy
        payload["native_forces"] = result.forces.tolist()
        payload["executed_backend"] = result.executed_backend
        payload["converged"] = result.converged
        if args.mode == "cbs":
            payload["groups"] = result.execution_groups
            payload["basis_metadata"] = result.basis_metadata
            payload["components"] = [
                {
                    "method": component.method,
                    "roles": component.roles,
                    "energy": component.result.energy,
                    "forces": component.result.forces.tolist(),
                    "correlation": None
                    if component.result.correlation is None
                    else asdict(component.result.correlation),
                    "cc_performance": None
                    if component.result.cc_performance is None
                    else asdict(component.result.cc_performance),
                }
                for component in result.components
            ]
        else:
            payload["correlation"] = asdict(result.correlation)

        def oracle(geometry: list) -> float:
            lower = energies(
                geometry, args.low, args.low_fit, triples=args.mode == "cbs"
            )
            if args.mode == "mp2":
                return lower["mp2"]
            higher = energies(geometry, args.high, args.high_fit)
            return float(
                sum(
                    weight * component["rhf"]
                    for weight, component in zip(
                        owner.hf_weights, (lower, higher), strict=True
                    )
                )
                + sum(
                    weight * (component["mp2"] - component["rhf"])
                    for weight, component in zip(
                        owner.mp2_weights, (lower, higher), strict=True
                    )
                )
                + lower["df-ccsd(t)"]
                - lower["mp2"]
            )

        payload["oracle_energy"] = oracle(atoms)
        payload["energy_error_hartree"] = abs(result.energy - payload["oracle_energy"])
        direction = np.random.default_rng(709).normal(size=coordinates.shape)
        direction /= np.linalg.norm(direction)
        payload["direction"] = direction.tolist()
        analytic = -float(np.vdot(result.forces, direction))
        derivatives = []
        for step in (1e-4, 3e-5):
            displaced_energies = [
                oracle(
                    [
                        (int(number), position)
                        for number, position in zip(
                            numbers, coordinates + sign * step * direction, strict=True
                        )
                    ]
                )
                for sign in (-1, 1)
            ]
            derivative = (displaced_energies[1] - displaced_energies[0]) / (2 * step)
            derivatives.append(
                {
                    "step_bohr": step,
                    "oracle_derivative": derivative,
                    "analytic_derivative": analytic,
                    "error": abs(derivative - analytic),
                }
            )
        payload["directional_derivatives"] = derivatives
        payload["translation_error"] = float(np.max(np.abs(result.forces.sum(axis=0))))
        if (
            payload["energy_error_hartree"] > 5e-8
            or max(item["error"] for item in derivatives) > 5e-6
        ):
            raise RuntimeError("independent molecular energy/force gate failed")
        if (
            not result.converged
            or result.executed_backend != "cuda"
            or payload["translation_error"] > 1e-6
        ):
            raise RuntimeError("native backend/convergence/translation gate failed")
        payload["status"] = "qualified"
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = str(error)
        raise
    finally:
        if args.mode == "mp2":
            owner.clear_cache()
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "qm7_index_zero_based",
                    "mode",
                    "complete_endpoint_seconds",
                    "energy_error_hartree",
                    "directional_derivatives",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
