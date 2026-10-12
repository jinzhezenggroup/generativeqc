"""Screen bounded DFT mixed warmup using synchronized public endpoint calls.

This exploratory runner consumes frozen DFT-MP-v1 inputs/grid, but does not
declare the contract's product or default-promotion rows passed. References are
separate CPU calculations, never inputs to the measured native SCF endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "tools/dft_mp_v1/inputs"


def digest(path: Path) -> str:
    """Hash the actual installed artifact or frozen input, outside timing."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def mixed_environment(residual: float, iterations: int) -> dict[str, str]:
    """Resolve only approximate handoff controls, never final tolerances."""
    if not math.isfinite(residual) or residual < 0 or iterations < 0:
        raise ValueError("warmup controls must be finite and nonnegative")
    return {
        "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL": str(residual),
        "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS": str(iterations),
    }


def inputs(case: str, changed: bool = False) -> tuple[dict[str, Any], Path]:
    """Read the frozen geometry without regenerating or changing its bytes."""
    path = INPUTS / f"{case}{'-changed' if changed else ''}.json"
    return json.loads(path.read_text()), path


def grid_spec() -> Any:
    """Use the contract's common explicit moving quadrature for every method."""
    from generativeqc import GridSpec

    manifest = json.loads((ROOT / "tools/dft_mp_v1/manifest.json").read_text())
    values = dict(manifest["model"]["grid_spec"])
    values["element_radii"] = tuple(tuple(row) for row in values["element_radii"])
    return GridSpec(**values)


def calculator(args: argparse.Namespace, geometry: dict[str, Any], device: str) -> Any:
    """Prepare the unchanged physical method and target convergence controls."""
    from generativeqc import Calculator, KsOptions

    return Calculator(
        method=args.method,
        basis="def2-SVP",
        basis_representation="spherical",
        device=device,
        precision=args.precision if device == "cuda" else "fp64",
        ks_options=KsOptions(grid=grid_spec()),
        energy_tolerance=getattr(args, "energy_tolerance", 1e-10),
        density_tolerance=getattr(args, "density_tolerance", 1e-8),
        screening_tolerance=1e-12,
        max_iterations=100,
    )


def native_sample(args: argparse.Namespace) -> dict[str, Any]:
    """Measure fresh preparation, optional reuse and changed-geometry endpoints."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "real GPU samples require an existing finite Slurm allocation"
        )
    import cupy
    import numpy as np

    os.environ.update(mixed_environment(args.switch_residual, args.warmup_max))
    geometry, input_path = inputs(args.case)
    records = []
    stages = ("cold", "warm", "changed_geometry") if args.lifecycle else ("cold",)
    properties = ("energy", "forces") if args.forces else ("energy",)
    cupy.cuda.Stream.null.synchronize()
    print(json.dumps({"phase": "prepare_begin", "case": args.case}), flush=True)
    started = perf_counter()
    calc = calculator(args, geometry, "cuda")
    owner = calc.prepare_batch(
        [geometry["atoms"]],
        charges=[geometry.get("charge", 0)],
        multiplicities=[geometry.get("multiplicity", 1)],
        warm_start=True,
    )
    cupy.cuda.Stream.null.synchronize()
    prepare_seconds = perf_counter() - started
    print(
        json.dumps({"phase": "prepare_complete", "seconds": prepare_seconds}),
        flush=True,
    )
    try:
        for stage in stages:
            coordinates = None
            if stage == "changed_geometry":
                moved, _ = inputs(args.case, True)
                coordinates = [np.array([atom[1] for atom in moved["atoms"]])]
            cupy.cuda.Stream.null.synchronize()
            print(json.dumps({"phase": "execute_begin", "stage": stage}), flush=True)
            started = perf_counter()
            item = owner.execute(
                coordinates, properties=properties, strict=False
            ).items[0]
            cupy.cuda.Stream.null.synchronize()
            elapsed = perf_counter() - started
            record = {
                "stage": stage,
                "prepare_seconds": prepare_seconds if stage == "cold" else 0.0,
                "execute_seconds": elapsed,
                "complete_seconds": elapsed
                + (prepare_seconds if stage == "cold" else 0.0),
                "status": item.status,
                "detail": item.status_message,
                "converged": item.converged,
                "energy": item.energy,
                "forces": item.forces.tolist() if item.forces is not None else None,
                "iterations": item.iterations,
                "fock_builds": item.fock_builds,
                "energy_change": item.energy_change,
                "physical_residual_rms": item.physical_residual_rms,
                "precision": item.precision,
                "diagnostic": item.ks_diagnostic.to_payload()
                if item.ks_diagnostic
                else None,
                "warm_start_used": item.warm_start_used,
                "warm_start_fallback": item.warm_start_fallback,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "execute_complete",
                        "stage": stage,
                        "seconds": elapsed,
                        "status": item.status,
                        "iterations": item.iterations,
                        "fock_builds": item.fock_builds,
                    }
                ),
                flush=True,
            )
            if item.status != 0 or not item.converged:
                break
    finally:
        owner.close()
    library = Path(calc._library._name)
    artifacts = {}
    if args.forces:
        profile = args.method.replace("-", "_")
        for path in (
            library.parent / f"libgenerativeqc_stationary_{profile}_spd.so",
            library.parent / f"generativeqc_stationary_{profile}_spd.json",
        ):
            if path.is_file():
                artifacts[path.name] = digest(path)
    return {
        "role": "native",
        "case": args.case,
        "atom_count": len(geometry["atoms"]),
        "declared_spherical_aos": json.loads(
            (ROOT / "tools/dft_mp_v1/manifest.json").read_text()
        )["cases"][args.case]["ao_count_spherical"],
        "method": args.method,
        "precision": args.precision,
        "switch_residual": args.switch_residual,
        "warmup_max": args.warmup_max,
        "forces_requested": args.forces,
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "library_sha256": digest(library),
        "stationary_artifact_sha256": artifacts,
        "input_sha256": digest(input_path),
        "source_commit": args.source_commit,
        "source_patch_sha256": digest(args.source_patch) if args.source_patch else None,
        "runner_sha256": digest(Path(__file__)),
        "energy_tolerance": args.energy_tolerance,
        "density_tolerance": args.density_tolerance,
        "screening_tolerance": 1e-12,
        "records": records,
    }


def reference_grid_arrays(grid: Any, tile_points: int = 8192) -> tuple[Any, Any]:
    """Export identical quadrature with bounded, larger independent CPU tiles.

    Tiling changes only Python dispatch work, not points, partition arithmetic,
    ordering or SCF targets. Avoid the reference exporter's 256-point default
    when the frozen grid contains millions of points.
    """
    import numpy as np

    if tile_points < 1 or grid.npoint > 20_000_000:
        raise ValueError("independent grid exceeds the bounded export contract")
    points = np.empty((grid.npoint, 3))
    weights = np.empty(grid.npoint)
    for tile in grid.tiles(tile_points):
        end = tile.begin + len(tile.weights)
        points[tile.begin : end] = tile.points
        weights[tile.begin : end] = tile.weights
    return points, weights


def _reference_energy(
    args: argparse.Namespace, geometry: dict[str, Any]
) -> dict[str, Any]:
    import pyscf
    from generativeqc import Atom
    from generativeqc_compiler.dft.grid import MolecularGrid
    from pyscf import dft, gto

    oracle_threads = getattr(args, "oracle_threads", 1)
    pyscf.lib.num_threads(oracle_threads)
    atoms = tuple(Atom.from_value(atom) for atom in geometry["atoms"])
    calc = calculator(args, geometry, "cpu")
    labels = [f"{atom[0]}{index}" for index, atom in enumerate(geometry["atoms"])]
    basis: dict[str, list[Any]] = {label: [] for label in labels}
    for shell in calc._shells_for_atoms(atoms):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(row.exponent, row.coefficient) for row in shell.primitives],
            ]
        )
    molecule = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis,
        cart=False,
        unit="Bohr",
        charge=geometry.get("charge", 0),
        spin=geometry.get("multiplicity", 1) - 1,
        verbose=0,
    )
    grid = MolecularGrid(
        atoms, spec=grid_spec(), charge=molecule.charge, multiplicity=molecule.spin + 1
    )
    print(json.dumps({"phase": "oracle_grid_begin", "points": grid.npoint}), flush=True)
    points, weights = reference_grid_arrays(grid)
    print(json.dumps({"phase": "oracle_scf_begin"}), flush=True)
    engine = dft.UKS(molecule) if args.method.endswith("uks") else dft.RKS(molecule)
    engine.xc = args.method.removesuffix("-rks").removesuffix("-uks")
    engine.grids.coords = points
    engine.grids.weights = weights
    engine.small_rho_cutoff = 0
    engine.direct_scf = False
    engine.direct_scf_tol = 1e-14
    engine.conv_tol = 1e-13
    engine.conv_tol_grad = 1e-10
    engine.max_cycle = 150
    energy = engine.kernel()
    print(
        json.dumps(
            {"phase": "oracle_scf_complete", "converged": bool(engine.converged)}
        ),
        flush=True,
    )
    return {
        "role": "oracle",
        "case": args.case,
        "method": args.method,
        "pyscf_version": pyscf.__version__,
        "oracle_threads": oracle_threads,
        "energy": float(energy),
        "converged": bool(engine.converged),
        "basis_functions": molecule.nao_nr(),
        "grid_points": grid.npoint,
        "grid_export_tile_points": 8192,
    }


def oracle_sample(args: argparse.Namespace) -> dict[str, Any]:
    """Compute independent energy and optional two-step directional differences."""
    import numpy as np

    geometry, input_path = inputs(args.case)
    result = _reference_energy(args, geometry)
    result["input_sha256"] = digest(input_path)
    if getattr(args, "finite_difference", False):
        positions = np.array([atom[1] for atom in geometry["atoms"]])
        direction = np.arange(positions.size, dtype=float).reshape(positions.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        differences = []
        for step in getattr(args, "fd_steps", (1e-4, 5e-5)):
            displaced = []
            for sign in (-1.0, 1.0):
                shifted = positions + sign * step * direction
                value = dict(geometry)
                value["atoms"] = [
                    (atom[0], row.tolist())
                    for atom, row in zip(geometry["atoms"], shifted, strict=True)
                ]
                reference = _reference_energy(args, value)
                if not reference["converged"]:
                    raise RuntimeError("independent displaced SCF did not converge")
                displaced.append(reference["energy"])
            differences.append(
                {
                    "step_bohr": step,
                    "energies": displaced,
                    "directional_derivative": (displaced[1] - displaced[0])
                    / (2 * step),
                }
            )
        result["direction"] = direction.tolist()
        result["finite_differences"] = differences
    return result


def audit_endpoint(record: dict[str, Any], forces: bool) -> float:
    """Require both candidate and comparator to be complete strict endpoints."""
    if record["status"] != 0 or not record["converged"]:
        raise ValueError("the attempted endpoint did not converge successfully")
    residual = record["diagnostic"]["physical_residual_max"]
    if (
        not math.isfinite(residual)
        or residual < 0
        or residual > 1e-9
        or not math.isfinite(record["energy"])
    ):
        raise ValueError("the endpoint did not pass the strict physical audit")
    seconds = record["complete_seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("the endpoint timing must be finite and positive")
    if forces:
        values = record["forces"]
        if not values or any(
            len(row) != 3 or not all(math.isfinite(value) for value in row)
            for row in values
        ):
            raise ValueError(
                "the requested complete force endpoint is missing or invalid"
            )
    return residual


def paired_errors(
    actual: dict[str, Any], expected: dict[str, Any], forces: bool
) -> dict[str, float]:
    """Reject inaccurate or missing endpoint work, not just a favorable median."""
    residual = audit_endpoint(actual, forces)
    baseline_residual = audit_endpoint(expected, forces)
    energy_error = abs(actual["energy"] - expected["energy"])
    force_error = 0.0
    if forces:
        differences = [
            abs(actual_value - expected_value)
            for actual_row, expected_row in zip(
                actual["forces"], expected["forces"], strict=True
            )
            for actual_value, expected_value in zip(
                actual_row, expected_row, strict=True
            )
        ]
        if not differences or not all(math.isfinite(value) for value in differences):
            raise ValueError("the force endpoint contains invalid components")
        force_error = max(differences)
    if energy_error > 1e-8 or force_error > 1e-7:
        raise ValueError("the paired physical energy/force acceptance gate failed")
    return {
        "energy_error": energy_error,
        "force_error": force_error,
        "residual": residual,
        "baseline_residual": baseline_residual,
    }


def run_sample(command: list[str], log_path: Path, timeout: float) -> int | str:
    """Bound the complete subprocess tree, including any compiler children."""
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        ) as process,
    ):
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # Children may survive their parent; terminate the session too.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            return "timeout"


def summarize_attempts(
    attempts: list[dict[str, Any]],
    policies: list[str],
    *,
    forces: bool,
    lifecycle: bool,
) -> dict[str, Any]:
    """One invalid attempt invalidates its policy; incomplete lifetimes fail."""
    baselines = {
        attempt["repeat"]: attempt
        for attempt in attempts
        if attempt["policy"] == "fp64"
    }
    summaries = {}
    expected_stages = ["cold", "warm", "changed_geometry"] if lifecycle else ["cold"]
    for name in policies:
        selected = [attempt for attempt in attempts if attempt["policy"] == name]
        errors, stages, gates = [], {}, []
        if not selected:
            errors.append("the requested policy has no attempted samples")
        for attempt in selected:
            try:
                baseline = baselines[attempt["repeat"]]
                if attempt["exit_code"] != 0 or baseline["exit_code"] != 0:
                    raise ValueError("a paired subprocess failed or timed out")
                actual_records, expected_records = (
                    attempt["result"]["records"],
                    baseline["result"]["records"],
                )
                if any(
                    [record["stage"] for record in records] != expected_stages
                    for records in (actual_records, expected_records)
                ):
                    raise ValueError("the requested lifetime boundaries are incomplete")
                pair_gates = [
                    paired_errors(actual, expected, forces)
                    for actual, expected in zip(
                        actual_records, expected_records, strict=True
                    )
                ]
                gates.extend(pair_gates)
                for actual in actual_records:
                    stages.setdefault(actual["stage"], []).append(
                        actual["complete_seconds"]
                    )
            except (ValueError, KeyError, TypeError) as failure:
                errors.append(str(failure))
        summaries[name] = {
            "errors": errors,
            "seconds_median": {
                stage: statistics.median(values) for stage, values in stages.items()
            }
            if not errors
            else None,
            "paired_gates": gates,
        }
    return summaries


def scan(args: argparse.Namespace) -> dict[str, Any]:
    """Retain alternating-order fresh-process pairs, including failed attempts."""
    destination = args.output.with_suffix("")
    destination.mkdir(parents=True, exist_ok=True)
    policies = [("fp64", 0.0, 0)] + [
        (f"auto-{value:g}", value, args.warmup_max) for value in args.thresholds
    ]
    if args.include_legacy:
        policies.append(("legacy-auto", 0.0, 0))
    attempts = []
    for repeat in range(args.repeats):
        ordered = policies if repeat % 2 == 0 else list(reversed(policies))
        for order, (name, threshold, limit) in enumerate(ordered):
            output = destination / f"{repeat:02}-{order:02}-{name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "native",
                "--method",
                args.method,
                "--case",
                args.case,
                "--precision",
                "fp64" if name == "fp64" else "auto",
                "--switch-residual",
                str(threshold),
                "--warmup-max",
                str(limit),
                "--energy-tolerance",
                str(args.energy_tolerance),
                "--density-tolerance",
                str(args.density_tolerance),
                "--source-commit",
                args.source_commit,
                "--output",
                str(output),
            ]
            if args.source_patch:
                command.extend(("--source-patch", str(args.source_patch)))
            if args.forces:
                command.append("--forces")
            if args.lifecycle:
                command.append("--lifecycle")
            started = perf_counter()
            attempt = {
                "policy": name,
                "repeat": repeat,
                "order": order,
                "command": command,
                "sample": str(output),
            }
            attempt["exit_code"] = run_sample(
                command, output.with_suffix(".log"), args.sample_timeout
            )
            attempt["process_seconds"] = perf_counter() - started
            if output.exists():
                attempt["result"] = json.loads(output.read_text())
            attempts.append(attempt)
            args.output.write_text(json.dumps({"attempts": attempts}, indent=2) + "\n")
            print(
                name,
                repeat,
                attempt["exit_code"],
                attempt["process_seconds"],
                flush=True,
            )
    summaries = summarize_attempts(
        attempts,
        [name for name, _, _ in policies],
        forces=args.forces,
        lifecycle=args.lifecycle,
    )
    return {
        "case": args.case,
        "method": args.method,
        "forces_requested": args.forces,
        "attempts": attempts,
        "summaries": summaries,
    }


def main() -> None:
    """Retain every attempted sample; promotion requires separate paired review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("native", "oracle", "scan"))
    parser.add_argument("--method", default="pbe0-rks")
    parser.add_argument("--case", default="water")
    parser.add_argument("--precision", choices=("fp64", "auto"), default="fp64")
    parser.add_argument("--switch-residual", type=float, default=0.0)
    parser.add_argument("--warmup-max", type=int, default=0)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--density-tolerance", type=float, default=1e-8)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--oracle-threads", type=int, default=1)
    parser.add_argument("--fd-steps", type=float, nargs=2, default=(1e-4, 5e-5))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-patch", type=Path)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-timeout", type=float, default=600.0)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.oracle_threads < 1:
        parser.error("--oracle-threads must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = (
        scan(args)
        if args.role == "scan"
        else (native_sample(args) if args.role == "native" else oracle_sample(args))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("records", "attempts")
            }
        )
    )
    for record in result.get("records", []):
        print(
            record["stage"],
            record["complete_seconds"],
            record["iterations"],
            record["precision"].get("mixed_stage_fock_builds"),
        )
    if args.role == "native" and any(
        record["status"] != 0 or not record["converged"] for record in result["records"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
