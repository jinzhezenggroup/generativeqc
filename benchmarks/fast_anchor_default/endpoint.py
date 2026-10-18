"""Qualify one production Fast anchor cold E+force endpoint under Slurm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import traceback
import uuid
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
from generativeqc import GridSpec, KsOptions, PrecisionCalculator

from benchmarks.compare_gpu4pyscf_batch import (
    load_comparison_basis,
    native_build_metadata,
)
from benchmarks.pbe0_initial_guesses import export_density
from benchmarks.readme_hf_scaling import scaling_cases

if TYPE_CHECKING:
    from generativeqc.basis import BasisSet

    from benchmarks._cases import BenchmarkCase


def payload(value: Any) -> Any:
    """Export diagnostics without dropping nested work or validation fields."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [payload(item) for item in value]
    if hasattr(value, "to_dict"):
        return payload(value.to_dict())
    if hasattr(value, "to_payload"):
        return payload(value.to_payload())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unrecorded diagnostic type: {type(value).__name__}")


def publish_json(path: Path, record: dict[str, Any]) -> None:
    """Expose complete JSON atomically without replacing files or symlinks."""
    data = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=".receipt-", dir=path.parent) as staging:
        staged = Path(staging) / path.name
        with staged.open("x") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)


def json_evidence(value: Any, invalid: list[str], location: str = "$") -> Any:
    """Keep nonfinite diagnostics readable while refusing their qualification."""
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append(location)
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_evidence(item, invalid, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_evidence(item, invalid, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def publish_density(owner: Any, path: Path) -> dict[str, Any]:
    """Stage the existing export and publish it without clobbering evidence."""
    with tempfile.TemporaryDirectory(prefix=".density-", dir=path.parent) as staging:
        staged = Path(staging) / path.name
        result = export_density(owner, staged)
        density = np.load(staged, allow_pickle=False)
        if not density.size or not np.isfinite(density).all():
            raise ValueError("Final density is empty or nonfinite")
        os.link(staged, path)
    return {**result, "path": path.name}


def reference_gate(
    record: dict[str, Any], basis: BasisSet, case: BenchmarkCase, reference_path: Path
) -> dict[str, Any]:
    """Read the independent oracle only after the target endpoint and teardown."""
    reference = json.loads(reference_path.read_text())
    protocol = reference["protocol"]
    expected_geometry = [[atom[0], list(atom[1])] for atom in case.atoms]
    if protocol["basis_identity"] != basis.identity:
        raise RuntimeError("Independent oracle basis identity differs")
    if protocol["geometries_bohr"][0] != expected_geometry:
        raise RuntimeError("Independent oracle geometry differs")
    oracle = reference["records"][0]
    force_delta = np.asarray(record["forces"]) - np.asarray(oracle["forces"])
    energy_error = abs(record["energy"] - oracle["energy"])
    force_max = float(np.max(np.abs(force_delta)))
    force_rms = float(np.sqrt(np.mean(force_delta**2)))
    limits = (1e-8, 1e-7, 1e-7) if record["policy"] == "fp64" else (1e-5, 1e-4, 3e-5)
    return {
        "reference": str(reference_path),
        "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "engine": reference["engine"],
        "energy_error_hartree": energy_error,
        "force_max_error_hartree_per_bohr": force_max,
        "force_rms_error_hartree_per_bohr": force_rms,
        "limits_energy_force_max_force_rms": list(limits),
        "gate": bool(
            energy_error <= limits[0]
            and force_max <= limits[1]
            and force_rms <= limits[2]
        ),
    }


def main() -> None:
    """Keep all failures; each process owns exactly one fresh native calculator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(12, 96), required=True)
    parser.add_argument("--policy", choices=("fp64", "fast"), required=True)
    parser.add_argument("--delta", choices=("0", "1", "auto"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = {
        "schema": 1,
        "atoms": args.atoms,
        "policy": args.policy,
        "delta": args.delta,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "method": "wb97m-v-rks",
        "grid": [48, 16, 32],
        "density_fitting": "none",
        "initial_guess_request": "native auto (no external seed)",
        "warm_owner": "fresh; export enabled but no previous execute or imported seed",
        "timing_scope": "constructor+prepare+first synchronized E/F+host/density export+owner teardown+clear_cache; imports/basis parsing/provenance/oracle excluded",
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("GENERATIVEQC_")
        },
        "production_defaults_changed": True,
    }
    calculator = None
    started = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            raise RuntimeError("Real GPU endpoints require Slurm-assigned visibility")
        for path in (args.output, args.output.with_suffix(".density.npy")):
            if os.path.lexists(path):
                raise FileExistsError(path)
        selected = os.environ.get("GENERATIVEQC_KS_FAST_ANCHOR_DELTA")
        if selected != (None if args.delta == "auto" else args.delta):
            raise RuntimeError("Argument and native Fast anchor selection differ")
        root = Path(os.environ["EXPERIMENT_ROOT"])
        case = scaling_cases()[f"water-{args.atoms}"]
        basis_path = root / "def2-svp.json"
        basis, _ = load_comparison_basis(
            basis_path, case, role="orbital", compute_forces=True
        )
        record.update(
            basis_identity=basis.identity,
            basis_sha256=hashlib.sha256(basis_path.read_bytes()).hexdigest(),
            atoms_bohr=[[atom[0], list(atom[1])] for atom in case.atoms],
        )
        started = perf_counter()
        calculator = PrecisionCalculator(
            method="wb97m-v-rks",
            basis=basis,
            device="cuda",
            policy=args.policy,
            basis_representation="spherical",
            density_fitting="none",
            ks_options=KsOptions(grid=GridSpec(48, 16, 32)),
            max_iterations=100,
            diis_history=8,
        )
        cp.cuda.runtime.deviceSynchronize()
        record["construction_seconds"] = perf_counter() - started
        prepare_started = perf_counter()
        with calculator.prepare_batch([case.atoms], warm_start=True) as owner:
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - prepare_started
            execution_started = perf_counter()
            batch = owner.execute(strict=False, properties=("energy", "forces"))
            cp.cuda.runtime.deviceSynchronize()
            record["execute_seconds"] = perf_counter() - execution_started
            export_started = perf_counter()
            item = batch.items[0]
            for name in (
                "status",
                "status_message",
                "succeeded",
                "converged",
                "energy",
                "iterations",
                "fock_builds",
                "density_rms",
                "physical_residual_rms",
                "energy_change",
                "initial_guess",
                "warm_start_used",
                "warm_start_fallback",
                "precision",
                "precision_validation",
                "ks_diagnostic",
                "incremental_direct_jk",
                "basis_metadata",
            ):
                record[name] = payload(getattr(item, name, None))
            record["forces"] = payload(item.forces)
            record["finite_forces"] = bool(
                item.forces is not None and np.isfinite(item.forces).all()
            )
            record["batch_precision_validation"] = payload(batch.precision_validation)
            record["ks_transport_diagnostics"] = payload(owner.ks_transport_diagnostics)
            record["resource_diagnostics"] = payload(owner.resource_diagnostics)
            if item.succeeded and item.converged:
                if owner._owner is None:
                    raise RuntimeError(
                        "Audit detached the native owner; no candidate density exported"
                    )
                record["final_density"] = publish_density(
                    owner._owner, args.output.with_suffix(".density.npy")
                )
            cp.cuda.runtime.deviceSynchronize()
            record["export_seconds"] = perf_counter() - export_started
            teardown_started = perf_counter()
        calculator.clear_cache()
        cp.cuda.runtime.deviceSynchronize()
        record["teardown_seconds"] = perf_counter() - teardown_started
        record["cold_endpoint_seconds"] = perf_counter() - started
        record["native_build"] = native_build_metadata(calculator._candidate)
        record["candidate_controls"] = {
            name: payload(getattr(calculator._candidate, f"_{name}", None))
            for name in (
                "energy_tolerance",
                "density_tolerance",
                "screening_tolerance",
                "precision",
            )
        }
        if record.get("succeeded") and record.get("converged"):
            record["independent_gate"] = reference_gate(
                record, basis, case, root / f"reference{args.atoms}.json"
            )
    except Exception:  # noqa: BLE001 - Retain every failed endpoint and its traceback.
        record["error"] = traceback.format_exc()
        if started is not None:
            record.setdefault("failed_elapsed_seconds", perf_counter() - started)
    finally:
        if calculator is not None:
            try:
                calculator.clear_cache()
            except Exception:  # noqa: BLE001 - Cleanup cannot erase the first failure.
                record["cleanup_error"] = traceback.format_exc()
    invalid: list[str] = []
    record = json_evidence(record, invalid)
    if invalid:
        record["nonfinite_fields"] = invalid
    try:
        publish_json(args.output, record)
    except FileExistsError:
        record.setdefault("error", "Requested receipt path is already occupied")
        record["publication_error"] = traceback.format_exc()
        failure = args.output.with_name(
            f"{args.output.stem}.failed-{uuid.uuid4().hex}.json"
        )
        publish_json(failure, record)
        print(f"Failure receipt: {failure}", flush=True)
    print(
        json.dumps(
            {
                name: record.get(name)
                for name in (
                    "atoms",
                    "policy",
                    "delta",
                    "converged",
                    "energy",
                    "iterations",
                    "fock_builds",
                    "cold_endpoint_seconds",
                    "independent_gate",
                    "error",
                )
            }
        ),
        flush=True,
    )
    if (
        record.get("error")
        or record.get("cleanup_error")
        or record.get("nonfinite_fields")
        or not record.get("succeeded")
        or not record.get("converged")
        or not record.get("finite_forces")
        or record.get("warm_start_used")
        or not record.get("independent_gate", {}).get("gate")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
