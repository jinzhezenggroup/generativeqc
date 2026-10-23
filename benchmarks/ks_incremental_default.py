"""Source-matched default/off cold KS qualification under finite Slurm jobs.

Run each native arm in a fresh process. Cold includes calculator construction,
preparation, energy and forces; warm replays the frozen native cold density.
The separate reference command never supplies a production initial density.
This qualifies solver behavior on the stock grid, not quadrature convergence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.readme_defaults import METHODS, native_calculator, native_settings
from benchmarks.readme_hf_scaling import scaling_cases

SELECTORS = (
    "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
    "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK",
)
GATES = {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6}


def main() -> None:
    """Write raw samples and provenance, including failed numerical gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("standard", "fast", "reference"), required=True
    )
    parser.add_argument("--mode", choices=("direct", "df"), default="direct")
    parser.add_argument("--method", choices=("pbe0", "b3lyp", "m06-2x"), default="pbe0")
    parser.add_argument("--atoms", type=int, choices=(12, 48, 96), default=12)
    parser.add_argument("--selector", choices=("unset", "off"), default="unset")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("run inside a finite Slurm GPU allocation")
    for name in SELECTORS:
        os.environ.pop(name, None)
    if args.selector == "off":
        os.environ[SELECTORS[0]] = "off"
    import cupy as cp

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    xyz = np.array([point for _, point in atoms])
    moved = xyz.copy()
    moved[1, 0] += 0.02
    record = {
        "schema_version": 1,
        "source_revision": args.source_revision,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "method": args.method,
        "mode": args.mode,
        "arm": args.arm,
        "selector": args.selector,
        "atoms": atoms,
        "gates": GATES,
        "samples": [],
    }
    if args.arm == "reference":
        import gpu4pyscf
        import pyscf
        from gpu4pyscf.dft import rks
        from pyscf import gto, lib

        from benchmarks.readme_wb97mv import configure_reference_full_fock

        lib.num_threads(1)
        record["versions"] = {
            "gpu4pyscf": gpu4pyscf.__version__,
            "pyscf": pyscf.__version__,
        }
        record["controls"] = {
            "energy": 1e-12,
            "gradient": 1e-9,
            "screening": 1e-14,
            "full_density_fock": True,
            "grid_response": True,
        }
        for phase, coordinates in (("original", xyz), ("moved", moved)):
            mol = gto.M(
                atom=[
                    (atom[0], point)
                    for atom, point in zip(atoms, coordinates, strict=True)
                ],
                unit="Bohr",
                basis="def2-svp",
                cart=False,
                verbose=0,
            )
            ref = rks.RKS(mol)
            if args.mode == "df":
                basis = json.loads(
                    (args.inputs / f"{args.method}-reference.json").read_text()
                )
                ref = ref.density_fit(auxbasis=basis["basis"])
            ref.xc = METHODS[args.method][1]
            ref.conv_tol, ref.conv_tol_grad, ref.direct_scf_tol = 1e-12, 1e-9, 1e-14
            ref.max_cycle = 150
            configure_reference_full_fock(ref)
            energy = float(ref.kernel())
            if not ref.converged:
                raise RuntimeError("independent full-density reference failed")
            grad = ref.nuc_grad_method()
            grad.grid_response = True
            record["samples"].append(
                {
                    "phase": phase,
                    "energy": energy,
                    "forces": cp.asnumpy(-grad.kernel()).tolist(),
                }
            )
    else:
        from benchmarks.readme_method_endpoints import binary_identity

        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        calculator = native_calculator(args.method, args.arm, args.mode, args.inputs)
        owner = calculator.prepare_batch([atoms], warm_start=True)
        try:
            phases = ["cold", "warm", "warm"]
            if args.lifecycle:
                phases += ["moved", "moved-warm", "reset"]
            for phase in phases:
                if phase == "moved":
                    owner.set_warm_start_updates(True)
                if phase == "reset":
                    owner.clear_warm_starts()
                if phase != "cold":
                    cp.cuda.runtime.deviceSynchronize()
                    started = perf_counter()
                batch = owner.execute(
                    [moved] if phase in ("moved", "moved-warm", "reset") else None,
                    strict=True,
                    properties=("energy", "forces"),
                )
                item = batch.items[0]
                cp.cuda.runtime.deviceSynchronize()
                seconds = perf_counter() - started
                if not item.converged or item.executed_backend != "cuda":
                    raise RuntimeError("native complete CUDA endpoint failed")
                candidate = calculator if args.arm == "standard" else owner._calculator
                if "native_build" not in record:
                    record["native_build"] = binary_identity(candidate)
                requested = (
                    args.selector == "unset"
                    and args.arm == "standard"
                    and args.mode == "direct"
                )
                active = requested and phase in ("cold", "reset")
                record["samples"].append(
                    {
                        "phase": phase,
                        "seconds": seconds,
                        "energy": float(item.energy),
                        "forces": np.asarray(item.forces).tolist(),
                        "iterations": item.iterations,
                        "fock_builds": item.fock_builds,
                        "incremental": item.incremental_direct_jk,
                        "admission_passed": (
                            item.incremental_direct_jk["requested"] == requested
                            and item.incremental_direct_jk["active"] == active
                        ),
                        "initial_guess": item.initial_guess,
                        "energy_change": item.energy_change,
                        "density_rms": item.density_rms,
                        "physical_residual_rms": item.physical_residual_rms,
                        "warm_start_used": item.warm_start_used,
                        "warm_start_fallback": item.warm_start_fallback,
                        "precision": item.precision,
                        "precision_validation": item.precision_validation.to_dict()
                        if item.precision_validation
                        else None,
                        "batch_precision_validation": batch.precision_validation,
                        "settings": native_settings(candidate, item),
                    }
                )
                if phase == "cold":
                    owner.set_warm_start_updates(False)
        finally:
            owner.close()
            calculator.clear_cache()
        if args.reference:
            oracle = json.loads(args.reference.read_text())
            for key in ("method", "mode"):
                if oracle[key] != record[key]:
                    raise ValueError(f"reference {key} mismatch")
            if oracle["atoms"] != json.loads(json.dumps(atoms)):
                raise ValueError("reference geometry mismatch")
            record["reference"] = str(args.reference)
            refs = {sample["phase"]: sample for sample in oracle["samples"]}
            for sample in record["samples"]:
                ref = refs[
                    "original" if sample["phase"] in ("cold", "warm") else "moved"
                ]
                delta = np.array(sample["forces"]) - ref["forces"]
                sample["errors"] = {
                    "energy_per_atom": abs(sample["energy"] - ref["energy"])
                    / args.atoms,
                    "force_max": float(np.max(np.abs(delta))),
                    "force_rms": float(np.sqrt(np.mean(delta**2))),
                }
                sample["passed"] = all(
                    np.isfinite(value) and value < GATES[key]
                    for key, value in sample["errors"].items()
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if any(
        not sample.get("passed", True) or not sample.get("admission_passed", True)
        for sample in record["samples"]
    ):
        raise RuntimeError(f"numerical/admission gates failed; retained {args.output}")


if __name__ == "__main__":
    main()
