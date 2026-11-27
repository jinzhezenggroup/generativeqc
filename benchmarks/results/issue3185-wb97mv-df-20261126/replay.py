"""Acquire native WB97M-V DF E+F or independent Direct references on one H100."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import closing
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import cupy as cp

from benchmarks._retention import raw_output_path
from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.direct_phase_moved import identity
from benchmarks.readme_defaults import (
    allocation_metadata,
    digest,
    native_calculator,
    native_force_tile,
    native_settings,
    save,
)
from benchmarks.readme_defaults_accuracy import solve
from benchmarks.readme_method_endpoints import binary_identity

if TYPE_CHECKING:
    from typing import Any

METHOD = "wb97m-v"


def read_frames(path: Path) -> dict:
    """Read full or reduced frames without altering geometry precision."""
    data = json.loads(path.read_text())
    for frame in data["steps"]:
        if identity(frame["geometry_bohr"]) != frame["geometry_sha256"]:
            raise ValueError(f"geometry identity mismatch in step {frame['step']}")
    return data


def inspect_provider(owner: Any) -> dict:
    from generativeqc._ks_snapshot import NativeKsSnapshot

    with closing(NativeKsSnapshot(owner._owner, 0)) as snapshot:
        coulomb, exchange, cutoff = snapshot.fock_provider_proof()
        return {
            "primary_coulomb": coulomb,
            "primary_exchange": exchange,
            "relative_cutoff": cutoff,
            "long_range": "fitted" if snapshot.range_density_fitted else "exact",
        }


def run_native(args: argparse.Namespace, frames: list, record: dict) -> None:
    """Time each public endpoint, carrying only the last successful density."""
    build = json.loads(args.build_receipt.read_text())
    if build["status"] != "complete":
        raise ValueError("expected a complete release build receipt")
    record["source_revision"] = build["source_revision"]
    record["build_receipt_sha256"] = digest(args.build_receipt)
    record["auxiliary_sha256"] = digest(args.inputs / f"{METHOD}-auxiliary.json")
    properties = ("energy",) if args.energy_only else ("energy", "forces")
    record["properties"] = properties
    calculator = None
    owner = None
    last_good_step = None
    try:
        for index, frame in enumerate(frames):
            step = frame["step"]
            atoms = frame["geometry_bohr"]
            if index and args.mode == "moved-cold":
                owner.clear_warm_starts()
                last_good_step = None
            record["active_step"] = step
            save(args.output, record)
            cp.cuda.runtime.deviceSynchronize()
            start = perf_counter()
            if owner is None:
                calculator = native_calculator(METHOD, "standard", "df", args.inputs)
                owner = calculator.prepare_batch([atoms], warm_start=True)
            coordinates = [[xyz for _, xyz in atoms]] if index else None
            batch = owner.execute(coordinates, properties=properties, strict=False)
            cp.cuda.runtime.deviceSynchronize()
            elapsed = perf_counter() - start
            item = batch.items[0]
            native = owner._owner
            detail = (
                native._library.generativeqc_context_get_last_detail(native._context)
                if native is not None
                else None
            )
            if index == 0:
                actual = binary_identity(owner._calculator)
                if (
                    actual["source_identity"] != build["native_source_identity"]
                    or actual["library_sha256"] != build["library_sha256"]
                ):
                    raise ValueError(
                        "loaded native image differs from the build receipt"
                    )
                record["native_build"] = actual
            diagnostic = item.ks_diagnostic
            completed = (
                item.succeeded
                and item.converged
                and item.executed_backend == "cuda"
                and (args.energy_only or item.forces is not None)
                and item.precision_validation is not None
                and item.precision_validation.profile == "standard"
            )
            sample = {
                "step": step,
                "geometry_sha256": frame["geometry_sha256"],
                "seconds": elapsed,
                "complete": completed,
                "iterations": item.iterations if item.succeeded else None,
                "fock_builds": item.fock_builds,
                "terminal_attempt_history_rows": len(diagnostic.history)
                if diagnostic is not None
                else None,
                "terminal_attempt_fock_builds": diagnostic.fock_builds
                if diagnostic is not None
                else None,
                "status": item.status,
                "status_message": item.status_message,
                "native_detail": detail.decode(errors="replace") if detail else None,
                "converged": item.converged,
                "executed_backend": item.executed_backend if item.succeeded else None,
                "seed_step": last_good_step if item.warm_start_used else None,
                "warm_start_used": item.warm_start_used,
                "warm_start_fallback": item.warm_start_fallback,
                "energy": float(item.energy) if item.succeeded else None,
                "forces": item.forces.tolist() if item.forces is not None else None,
                "settings": {
                    **native_settings(owner._calculator, item),
                    "resolved_force_tile_points": native_force_tile(native),
                },
                "precision_validation": item.precision_validation.to_dict()
                if item.precision_validation
                else None,
            }
            if completed and args.inspect_route:
                sample["provider"] = inspect_provider(owner)
            if completed:
                last_good_step = step
            record["samples"].append(sample)
            save(args.output, record)
            print(
                f"{step} {elapsed:.3f}s cycles={sample['iterations']} "
                f"seed={sample['seed_step']} status={item.status}",
                flush=True,
            )
            if not completed and args.mode != "replay":
                break
    finally:
        if owner is not None:
            owner.close()
        if calculator is not None:
            calculator.clear_cache()


def run_oracle(args: argparse.Namespace, frames: list, record: dict) -> None:
    """Retain each independent dense Direct endpoint, including failures."""
    for frame in frames:
        for level in (4, 5):
            record["active_step"] = frame["step"]
            record["active_level"] = level
            save(args.output, record)
            sample = {
                "step": frame["step"],
                "level": level,
                "geometry_sha256": frame["geometry_sha256"],
            }
            try:
                sample.update(
                    solve(METHOD, frame["geometry_bohr"], backend="cuda", level=level)
                )
                sample["complete"] = True
            except (RuntimeError, OSError, ValueError, ArithmeticError) as error:
                sample.update(complete=False, error=f"{type(error).__name__}: {error}")
            record["samples"].append(sample)
            save(args.output, record)
            print(
                f"oracle step={frame['step']} level={level} complete={sample['complete']}",
                flush=True,
            )


def main() -> int:
    """Check the allocation and produce one fresh-process acquisition journal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument(
        "--mode",
        choices=("replay", "cold", "pair", "moved-cold", "oracle"),
        required=True,
    )
    parser.add_argument("--step", action="append", type=int, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--inspect-route", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode != "oracle" and (args.inputs is None or args.build_receipt is None):
        parser.error("native modes require --inputs and --build-receipt")
    if args.mode != "oracle" and len(args.step) != 1:
        parser.error("native modes require exactly one terminal --step")

    trajectory_sha = digest(args.trajectory)
    if trajectory_sha != args.expected_sha256:
        raise ValueError("frozen trajectory SHA-256 mismatch")
    data = read_frames(args.trajectory)
    frozen = {frame["step"]: frame for frame in data["steps"]}
    step = args.step[0]
    if args.mode == "replay":
        indices = range(step + 1)
    elif args.mode in ("pair", "moved-cold"):
        indices = (step - 1, step)
    else:
        indices = args.step
    frames = [frozen[index] for index in indices]
    device = cuda_accelerator_metadata(cp, sampling_point="before acquisition")
    if cp.cuda.runtime.getDeviceCount() != 1 or "H100" not in device["name"]:
        raise ValueError("expected a single scheduler-visible H100")
    record = {
        "schema": "generativeqc.wb97mv-df-rank-boundary.v1",
        "system": data["system"],
        "method": METHOD,
        "mode": args.mode,
        "trajectory_sha256": trajectory_sha,
        "driver_sha256": digest(Path(__file__)),
        "allocation": allocation_metadata(os.environ.get("GQC_QZ_JOB_NAME")),
        "accelerator": device,
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                "cupy": ("cupy-cuda12x",),
                "numpy": ("numpy",),
            }
        ),
        "status": "running",
        "samples": [],
    }
    if args.mode != "oracle":
        # Import before starting the cold timer, as in the source campaign.
        import generativeqc

        record["engine_module"] = generativeqc.__file__
    save(args.output, record)
    try:
        if args.mode == "oracle":
            from gpu4pyscf.dft import rks

            record["engine_module"] = rks.__file__
            if record["environment"]["packages"]["gpu4pyscf"] != "1.9.0":
                raise ValueError("the independent oracle requires GPU4PySCF 1.9.0")
            run_oracle(args, frames, record)
        else:
            run_native(args, frames, record)
        record["status"] = (
            "complete" if all(s["complete"] for s in record["samples"]) else "failed"
        )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, record)
    return int(record["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
