"""Fail-closed PBE0 Direct/DF complete-endpoint pilot for issue #2054.

Run each arm in a separate process. Raw attempts live under .artifacts; the
``summarize`` command never turns absent, failed, or mismatched evidence into a
performance claim. This runner does not select a production approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.dft_force_matrix import FORMALDEHYDE
from benchmarks.readme_hf_scaling import scaling_cases

SCHEMA = "generativeqc.hybrid-provider-crossover.v1"
ARMS = {
    "direct": {"density_fitting": "none", "proof": ["exact", "exact"]},
    "df-j-exact-k": {"density_fitting": None, "proof": ["density-fitted", "exact"]},
    "df-jk-occupied": {
        "density_fitting": "cuda",
        "proof": ["density-fitted", "density-fitted"],
    },
}
ENERGY_GATE = 1e-8
FORCE_GATE = 3e-7
PHYSICAL_GATE = 1e-8
DF_CONTROLS = {
    "GENERATIVEQC_DF_EXCHANGE": "occupied",
    "GENERATIVEQC_DF_RESIDENT_EXCHANGE": "full",
    "GENERATIVEQC_DF_VALUE_STORAGE": "packed-single",
    "GENERATIVEQC_DF_FINAL_PROJECTION": "auto",
    "GENERATIVEQC_DF_RESPONSE_SPACE": "auto",
    "GENERATIVEQC_DF_RESPONSE_STORAGE": "auto",
    "GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE": "auto",
}


@dataclass(frozen=True)
class Case:
    name: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    charge: int = 0
    multiplicity: int = 1


def case_named(name: str) -> Case:
    if name == "formaldehyde":
        return Case(name, FORMALDEHYDE)
    if name in {"water-3", "water-48", "water-96"}:
        return Case(name, tuple(scaling_cases()[name].atoms))
    raise ValueError(f"unknown frozen case {name!r}")


def moved(case: Case) -> Case:
    xyz = np.asarray([p for _, p in case.atoms], dtype=np.float64)
    xyz[-1] += (0.001, -0.0005, 0.0003)
    return Case(
        case.name + "-moved",
        tuple((symbol, tuple(map(float, point))) for (symbol, _), point in zip(case.atoms, xyz, strict=True)),
        case.charge,
        case.multiplicity,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def source_identity() -> dict[str, Any]:
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def frozen_problem(case: Case, *, auxiliary: str, budget: int) -> dict[str, Any]:
    from generativeqc_compiler.dft.grid import GridSpec

    grid = GridSpec()
    return {
        "method": "pbe0-rks",
        "geometry_bohr": case.atoms,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "orbital_basis": "def2-svp",
        "representation": "spherical",
        "grid_spec": asdict(grid),
        "precision": "fp64",
        "screening_tolerance": 1e-12,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "max_iterations": 180,
        "products": ["energy", "forces"],
        "auxiliary_basis": auxiliary,
        "df_memory_budget_bytes": budget,
    }


def calculator(problem: dict[str, Any], arm: str) -> Any:
    from generativeqc import Calculator, KsOptions
    from generativeqc_compiler.dft.grid import GridSpec

    mode = ARMS[arm]["density_fitting"]
    if mode is None:
        raise NotImplementedError("public KS contract has no DF-J plus exact-K selector")
    return Calculator(
        method=problem["method"],
        basis=problem["orbital_basis"],
        basis_representation=problem["representation"],
        device="cuda",
        density_fitting=mode,
        auxiliary_basis=problem["auxiliary_basis"] if mode == "cuda" else None,
        density_fitting_memory_budget_bytes=(problem["df_memory_budget_bytes"] if mode == "cuda" else 0),
        ks_options=KsOptions(grid=GridSpec()),
        precision=problem["precision"],
        screening_tolerance=problem["screening_tolerance"],
        energy_tolerance=problem["energy_tolerance"],
        density_tolerance=problem["density_tolerance"],
        max_iterations=problem["max_iterations"],
    )


def _json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_payload"):
        return _json(value.to_payload())
    if hasattr(value, "to_dict"):
        return _json(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _counts(calc: Any, case: Case) -> dict[str, int | None]:
    from generativeqc_compiler.dft import NativeAO

    with NativeAO(case.atoms, basis=calc._basis, representation="spherical", multiplicity=1) as ao:
        nao = ao.nao
    naux = None
    if calc._density_fitting_mode != 0:
        with NativeAO(case.atoms, basis=calc._auxiliary_basis, representation="spherical", multiplicity=1) as aux:
            naux = aux.nao
    electrons = sum({"H": 1, "C": 6, "O": 8}[symbol] for symbol, _ in case.atoms) - case.charge
    return {"nao": nao, "naux": naux, "nocc": electrons // 2}


def _proof(batch: Any, arm: str) -> dict[str, Any]:
    from generativeqc._ks_snapshot import NativeKsSnapshot

    snapshot = NativeKsSnapshot(batch, 0)
    try:
        coulomb, exchange, threshold = snapshot.fock_provider_proof()
    finally:
        snapshot.close()
    expected = ARMS[arm]["proof"]
    if [coulomb, exchange] != expected:
        raise RuntimeError(f"provider mismatch: {[coulomb, exchange]} != {expected}")
    if arm == "direct" and threshold != 0.0:
        raise RuntimeError("exact Direct provider reported fitted metric threshold")
    if arm == "df-jk-occupied" and threshold != 1e-10:
        raise RuntimeError("DF provider metric threshold differs from frozen protocol")
    return {"coulomb": coulomb, "exchange": exchange, "metric_relative_threshold": threshold}


def _save_oracle_grid(batch: Any, calc: Any, case: Case, path: Path) -> dict[str, Any]:
    """Export only grid inputs; PySCF performs independent SCF and gradient work."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    with NativeAO(case.atoms, basis=calc._basis, representation="spherical", multiplicity=1) as basis:
        state = StationaryKsState.from_native(batch, basis)
        try:
            points = np.asarray(state.grid.points)
            weights = np.asarray(state.grid.weights)
            owners = np.asarray(state.grid.owners)
            atomic_weights = np.asarray(state._source.atomic_weights)
            if points.shape != (len(weights), 3) or owners.shape != weights.shape or atomic_weights.shape != weights.shape:
                raise RuntimeError("invalid exported quadrature arrays")
            arrays = (points, weights, owners, atomic_weights)
            digest = hashlib.sha256()
            for array in arrays:
                contiguous = np.ascontiguousarray(array)
                digest.update(str(contiguous.dtype).encode())
                digest.update(str(contiguous.shape).encode())
                digest.update(contiguous.tobytes())
            np.savez_compressed(path, points=points, weights=weights, owners=owners, atomic_weights=atomic_weights)
        finally:
            state._source.close()
    return {"path": str(path), "sha256": sha256(path), "semantic_sha256": digest.hexdigest(), "points": len(points)}


def _sample(batch: Any, geometry: Case, phase: str, *, coordinates: bool) -> dict[str, Any]:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = batch.execute(
        coordinates=[np.asarray([p for _, p in geometry.atoms], dtype=np.float64)] if coordinates else None,
        strict=False,
        properties=("energy", "forces"),
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    item = result.items[0]
    force = None if item.forces is None else np.asarray(item.forces)
    valid = (
        item.succeeded and item.converged and item.executed_backend == "cuda"
        and math.isfinite(item.energy)
        and item.physical_residual_rms is not None
        and math.isfinite(item.physical_residual_rms)
        and item.physical_residual_rms <= PHYSICAL_GATE
        and force is not None and force.shape == (len(geometry.atoms), 3)
        and np.isfinite(force).all()
    )
    return {
        "phase": phase, "status": "PASS" if valid else "FAIL", "seconds": elapsed,
        "energy_hartree": item.energy if math.isfinite(item.energy) else None,
        "forces_hartree_per_bohr": None if force is None else force.tolist(),
        "converged": item.converged, "iterations": item.iterations,
        "status_message": item.status_message, "executed_backend": item.executed_backend,
        "physical_residual_rms": item.physical_residual_rms,
        "density_rms": item.density_rms, "fock_builds": item.fock_builds,
        "warm_start_used": item.warm_start_used,
        "warm_start_fallback": item.warm_start_fallback,
        "ks_diagnostic": item.ks_diagnostic,
    }


def run_native(args: argparse.Namespace) -> None:
    case = case_named(args.case)
    problem = frozen_problem(case, auxiliary=args.auxiliary_basis, budget=args.df_budget_bytes)
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("finite Slurm GPU allocation required")
    if os.environ.get("GENERATIVEQC_DF_TRACE"):
        raise RuntimeError("clean endpoint timing refuses enabled DF trace")
    if args.arm == "df-jk-occupied":
        os.environ.update(DF_CONTROLS)
    record: dict[str, Any] = {
        "schema": SCHEMA, "kind": "native", "case": args.case, "arm": args.arm,
        "problem": problem, "source": source_identity(), "python": sys.version,
        "platform": platform.platform(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attempts": [], "status": "RUNNING",
        "df_controls": DF_CONTROLS if args.arm == "df-jk-occupied" else None,
    }
    root = Path(__file__).resolve().parents[1]
    record["component_sha256"] = {
        str(path): sha256(root / path)
        for path in (
            Path("tools/benchmark_hybrid_provider_crossover.py"),
            Path("python/generativeqc/calculator.py"),
            Path("python/generativeqc_compiler/dft/grid.py"),
            Path("src/scf/cuda_fock_provider.cpp"),
            Path("src/dft/cuda_ks.cpp"),
        )
    }
    record["schema_sha256"] = hashlib.sha256(SCHEMA.encode()).hexdigest()
    if args.arm == "df-j-exact-k":
        record.update(status="UNSUPPORTED", reason="public KS has no DF-J plus exact-K selector")
        write(output, record)
        return
    try:
        import cupy as cp

        calc = calculator(problem, args.arm)
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
        record["device"] = {
            "name": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
            "ordinal": cp.cuda.runtime.getDevice(),
        }
        library = Path(calc._library._name).resolve()
        record["library"] = {"path": str(library), "sha256": sha256(library)}
        record["counts"] = _counts(calc, case)
        record["method_identity"] = calc.method_ir.identity
        record["ks_options_identity"] = calc.ks_options.identity
        write(output, record)
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        with calc.prepare_batch([case.atoms], charges=[case.charge], multiplicities=[case.multiplicity], warm_start=True) as batch:
            prepared = time.perf_counter()
            cold = _sample(batch, case, "cold", coordinates=False)
            cold["seconds"] += prepared - started
            cold["prepare_seconds"] = prepared - started
            record["attempts"].append(cold)
            write(output, record)
            if cold["status"] != "PASS":
                raise RuntimeError("cold endpoint failed")
            record["provider_proof"] = _proof(batch, args.arm)
            record["oracle_grids"] = {"original": _save_oracle_grid(batch, calc, case, output.with_suffix(".original-grid.npz"))}
            record["metric_diagnostics"] = [d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()]
            record["transport_diagnostics"] = [d.to_payload() if d else None for d in batch.ks_transport_diagnostics]
            record["resource_diagnostics"] = batch.resource_diagnostics
            write(output, record)
            batch.set_warm_start_updates(False)
            for repeat in range(args.repeats):
                sample = _sample(batch, case, f"warm-{repeat}", coordinates=False)
                record["attempts"].append(sample)
                write(output, record)
                if sample["status"] != "PASS":
                    raise RuntimeError("warm endpoint failed")
            batch.set_warm_start_updates(True)
            changed = _sample(batch, moved(case), "changed-geometry", coordinates=True)
            record["attempts"].append(changed)
            write(output, record)
            if changed["status"] != "PASS":
                raise RuntimeError("changed-geometry endpoint failed")
            record["oracle_grids"]["moved"] = _save_oracle_grid(batch, calc, moved(case), output.with_suffix(".moved-grid.npz"))
            write(output, record)
            batch.set_warm_start_updates(False)
            moved_warm = _sample(batch, moved(case), "moved-warm", coordinates=False)
            record["attempts"].append(moved_warm)
            write(output, record)
            if moved_warm["status"] != "PASS":
                raise RuntimeError("moved-warm endpoint failed")
        record["status"] = "MEASURED"
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write(output, record)


def run_reference(args: argparse.Namespace) -> None:
    """Run a separately sourced PySCF PBE0 E+force oracle for each geometry."""
    native = json.loads(args.native.read_text())
    if native.get("schema") != SCHEMA or native.get("status") != "MEASURED":
        raise ValueError("reference requires completed native input")
    if args.output.exists():
        raise FileExistsError(args.output)
    arm = native["arm"]
    if arm not in ("direct", "df-jk-occupied"):
        raise ValueError("no matched independent oracle for unsupported arm")
    case = case_named(native["case"])
    record: dict[str, Any] = {
        "schema": SCHEMA, "kind": "pyscf-oracle", "case": native["case"], "arm": arm,
        "native_record_sha256": sha256(args.native), "source": native["source"],
        "library": native["library"], "problem": native["problem"],
        "grid_exports": native["oracle_grids"], "status": "RUNNING", "samples": [],
    }
    write(args.output, record)
    _run_reference_impl(args, native, record, case, arm)


def run_profile(args: argparse.Namespace) -> None:
    """Diagnostic pass; its trace and timings never enter clean statistics."""
    from benchmarks.df_component_ledger import read_trace

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("finite Slurm GPU allocation required")
    if args.output.exists() or args.trace.exists():
        raise FileExistsError("profile output/trace already exists")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    if args.arm == "df-j-exact-k":
        raise ValueError("unsupported arm cannot be profiled as a substitute")
    if args.arm == "df-jk-occupied":
        os.environ.update(DF_CONTROLS)
    os.environ["GENERATIVEQC_DF_TRACE"] = str(args.trace.resolve())
    case = case_named(args.case)
    problem = frozen_problem(case, auxiliary=args.auxiliary_basis, budget=args.df_budget_bytes)
    record: dict[str, Any] = {
        "schema": SCHEMA, "kind": "diagnostic-profile", "case": args.case,
        "arm": args.arm, "source": source_identity(), "problem": problem,
        "status": "RUNNING", "trace": str(args.trace), "df_controls": DF_CONTROLS if args.arm == "df-jk-occupied" else None,
    }
    write(args.output, record)
    _run_profile_impl(args, record, case, problem, read_trace)


def _run_profile_impl(args: argparse.Namespace, record: dict[str, Any], case: Case, problem: dict[str, Any], read_trace: Any) -> None:
    try:
        calc = calculator(problem, args.arm)
        library = Path(calc._library._name).resolve()
        record["library"] = {"path": str(library), "sha256": sha256(library)}
        with calc.prepare_batch([case.atoms], charges=[case.charge], multiplicities=[case.multiplicity], warm_start=False) as batch:
            result = batch.execute(strict=False, properties=("energy", "forces")).items[0]
            record["endpoint"] = {
                "succeeded": result.succeeded, "converged": result.converged,
                "energy_hartree": result.energy, "physical_residual_rms": result.physical_residual_rms,
                "forces_hartree_per_bohr": result.forces,
            }
            if not result.succeeded or not result.converged:
                raise RuntimeError("diagnostic endpoint failed")
            record["provider_proof"] = _proof(batch, args.arm)
            record["metric_diagnostics"] = [d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()]
            record["resource_diagnostics"] = batch.resource_diagnostics
        if args.trace.exists() and args.trace.stat().st_size:
            rows = read_trace(args.trace)
            record["trace_sha256"] = sha256(args.trace)
            record["trace_rows"] = len(rows)
            record["executed_counters"] = [
                {"operation": row.get("operation"), "counters": row.get("counters")}
                for row in rows if row.get("counters")
            ]
            record["occupied_reuse_verified"] = any(
                row.get("operation") == "force_response"
                and row.get("counters", {}).get("response_final_fitted_projection_reused", 0) > 0
                and row.get("counters", {}).get("atom_coordinates", 0) == 3 * len(case.atoms)
                for row in rows
            ) if args.arm == "df-jk-occupied" else None
        else:
            record["trace_sha256"] = None
            record["executed_counters"] = None
            record["occupied_reuse_verified"] = False if args.arm == "df-jk-occupied" else None
        record["status"] = "MEASURED"
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write(args.output, record)


def _run_reference_impl(args: argparse.Namespace, native: dict[str, Any], record: dict[str, Any], case: Case, arm: str) -> None:
    try:
        import pyscf
        from pyscf import dft, gto, lib

        record["pyscf_version"] = pyscf.__version__
        lib.num_threads(args.threads)
        for label, geometry in (("original", case), ("moved", moved(case))):
            grid_info = native["oracle_grids"][label]
            path = Path(grid_info["path"])
            if sha256(path) != grid_info["sha256"]:
                raise ValueError("oracle grid export checksum mismatch")
            with np.load(path) as grid:
                points, weights, owners, atomic_weights = (np.asarray(grid[k]) for k in ("points", "weights", "owners", "atomic_weights"))
            labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(geometry.atoms)]
            mol = gto.M(atom=[(label, xyz) for label, (_, xyz) in zip(labels, geometry.atoms, strict=True)],
                        basis=native["problem"]["orbital_basis"], unit="Bohr", cart=False,
                        charge=geometry.charge, spin=geometry.multiplicity - 1, verbose=0)
            mf = dft.RKS(mol)
            if arm == "df-jk-occupied":
                mf = mf.density_fit(auxbasis=native["problem"]["auxiliary_basis"])
            mf.xc = "PBE0"
            mf.grids.coords = points
            mf.grids.weights = weights
            mf.grids.radii_adjust = None
            mf.small_rho_cutoff = 0
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-10
            mf.max_cycle = native["problem"]["max_iterations"]
            atomic_grid = {
                mol.atom_symbol(i): (points[owners == i] - mol.atom_coord(i), atomic_weights[owners == i])
                for i in range(mol.natm)
            }
            mf.grids.gen_atomic_grids = lambda *unused_args, grid=atomic_grid, **unused_kwargs: grid
            started = time.perf_counter()
            energy = float(mf.kernel())
            gradient = mf.nuc_grad_method()
            gradient.grid_response = True
            forces = -np.asarray(gradient.kernel())
            sample = {
                "geometry": label, "seconds": time.perf_counter() - started,
                "energy_hartree": energy, "forces_hartree_per_bohr": forces.tolist(),
                "converged": bool(mf.converged), "nao": mol.nao_nr(),
                "status": "PASS" if mf.converged and math.isfinite(energy) and np.isfinite(forces).all() else "FAIL",
            }
            record["samples"].append(sample)
            write(args.output, record)
        record["status"] = "MEASURED" if all(s["status"] == "PASS" for s in record["samples"]) else "FAILED"
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write(args.output, record)


def _same_problem(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = dict(left), dict(right)
    for value in (a, b):
        value.pop("auxiliary_basis", None)
        value.pop("df_memory_budget_bytes", None)
    return a == b


def _safe_median(values: list[Any]) -> float | None:
    if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
        return None
    return float(statistics.median(values))


def summarize(records: list[dict[str, Any]], oracles: list[dict[str, Any]] | None = None, profiles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not records or any(row.get("schema") != SCHEMA for row in records):
        raise ValueError("missing or foreign crossover records")
    source = records[0].get("source")
    problem = records[0].get("problem")
    library = records[0].get("library", {}).get("sha256")
    device = records[0].get("device")
    case = records[0].get("case")
    failures: list[str] = []
    by_arm: dict[str, Any] = {}
    for row in records:
        arm = row.get("arm")
        if arm not in ARMS or arm in by_arm:
            failures.append("unknown or repeated arm")
            continue
        if row.get("case") != case or not _same_problem(problem, row.get("problem", {})):
            failures.append(f"{arm}: problem identity mismatch")
        if row.get("source") != source or (row.get("library", {}).get("sha256") != library and row.get("status") != "UNSUPPORTED"):
            failures.append(f"{arm}: source/library mismatch")
        if row.get("status") != "UNSUPPORTED" and row.get("device") != device:
            failures.append(f"{arm}: device mismatch")
        if row.get("status") == "UNSUPPORTED":
            if arm != "df-j-exact-k":
                failures.append(f"{arm}: unexpected unsupported")
            by_arm[arm] = {"status": "UNSUPPORTED"}
            continue
        attempts = row.get("attempts", [])
        phases = [a.get("phase") for a in attempts]
        required = {"cold", "changed-geometry", "moved-warm", "warm-0"}
        if row.get("status") != "MEASURED" or not required.issubset(phases) or any(a.get("status") != "PASS" for a in attempts):
            failures.append(f"{arm}: incomplete or failed endpoint")
        natom = len(problem.get("geometry_bohr", [])) if isinstance(problem, dict) else 0
        for attempt in attempts:
            energy = attempt.get("energy_hartree")
            residual = attempt.get("physical_residual_rms")
            forces = np.asarray(attempt.get("forces_hartree_per_bohr"), dtype=float)
            seconds = attempt.get("seconds")
            if (not isinstance(energy, (int, float)) or not math.isfinite(energy)
                    or not isinstance(residual, (int, float)) or not math.isfinite(residual)
                    or residual > PHYSICAL_GATE or forces.shape != (natom, 3)
                    or not np.isfinite(forces).all() or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds <= 0):
                failures.append(f"{arm}/{attempt.get('phase')}: invalid physical or timed endpoint")
        if row.get("provider_proof", {}).get("coulomb") != ARMS[arm]["proof"][0] or row.get("provider_proof", {}).get("exchange") != ARMS[arm]["proof"][1]:
            failures.append(f"{arm}: provider proof mismatch")
        counts = row.get("counts", {})
        if not isinstance(counts.get("nao"), int) or not isinstance(counts.get("nocc"), int) or (arm == "df-jk-occupied" and not isinstance(counts.get("naux"), int)):
            failures.append(f"{arm}: missing actual dimensions")
        by_arm[arm] = {
            "status": row.get("status"),
            "times": {p: [a["seconds"] for a in attempts if a.get("phase") == p] for p in required},
            "warm_median_seconds": _safe_median([a.get("seconds") for a in attempts if str(a.get("phase", "")).startswith("warm-")]),
        }
    if set(by_arm) != set(ARMS):
        failures.append("three explicit arm dispositions required")
    if source is None or source.get("dirty"):
        failures.append("clean frozen source revision required")
    if library is None:
        failures.append("loaded library hash required")
    oracle_rows = {row.get("arm"): row for row in (oracles or [])}
    if len(oracle_rows) != len(oracles or []):
        failures.append("duplicate oracle arm")
    accuracy: dict[str, Any] = {}
    profile_rows = {row.get("arm"): row for row in (profiles or [])}
    if len(profile_rows) != len(profiles or []):
        failures.append("duplicate diagnostic profile arm")
    for arm in ("direct", "df-jk-occupied"):
        profile = profile_rows.get(arm)
        if profile is None or profile.get("schema") != SCHEMA or profile.get("kind") != "diagnostic-profile" or profile.get("status") != "MEASURED":
            failures.append(f"{arm}: missing or failed diagnostic profile")
            continue
        if profile.get("source") != source or profile.get("library", {}).get("sha256") != library or not _same_problem(problem, profile.get("problem", {})):
            failures.append(f"{arm}: diagnostic source/problem mismatch")
        if arm == "df-jk-occupied" and profile.get("occupied_reuse_verified") is not True:
            failures.append("df-jk-occupied: no completed occupied force-response reuse receipt")
    for arm in ("direct", "df-jk-occupied"):
        native = next((r for r in records if r.get("arm") == arm), None)
        oracle = oracle_rows.get(arm)
        if native is None or oracle is None:
            failures.append(f"{arm}: missing native or independent oracle")
            continue
        if oracle.get("schema") != SCHEMA or oracle.get("kind") != "pyscf-oracle" or oracle.get("status") != "MEASURED":
            failures.append(f"{arm}: invalid oracle status")
            continue
        if oracle.get("source") != source or oracle.get("library") != native.get("library") or oracle.get("problem") != native.get("problem"):
            failures.append(f"{arm}: oracle source/problem mismatch")
        if oracle.get("grid_exports") != native.get("oracle_grids"):
            failures.append(f"{arm}: oracle grid mismatch")
        references = {sample.get("geometry"): sample for sample in oracle.get("samples", [])}
        if set(references) != {"original", "moved"}:
            failures.append(f"{arm}: missing oracle geometry")
            continue
        errors = []
        for sample in native.get("attempts", []):
            phase = sample.get("phase", "")
            label = "moved" if phase in {"changed-geometry", "moved-warm"} else "original"
            reference = references[label]
            actual_forces = np.asarray(sample.get("forces_hartree_per_bohr"), dtype=float)
            expected_forces = np.asarray(reference.get("forces_hartree_per_bohr"), dtype=float)
            if (sample.get("status") != "PASS" or reference.get("status") != "PASS"
                    or actual_forces.shape != expected_forces.shape or actual_forces.ndim != 2
                    or not np.isfinite(actual_forces).all() or not np.isfinite(expected_forces).all()):
                failures.append(f"{arm}/{phase}: invalid force arrays or status")
                continue
            energy_error = abs(float(sample["energy_hartree"]) - float(reference["energy_hartree"]))
            force_error = float(np.max(np.abs(actual_forces - expected_forces)))
            errors.append({"phase": phase, "energy_error": energy_error, "force_max_abs_error": force_error})
            if not (energy_error <= ENERGY_GATE and force_error <= FORCE_GATE):
                failures.append(f"{arm}/{phase}: independent oracle gate failed")
        accuracy[arm] = errors
    return {
        "schema": SCHEMA, "case": case,
        "status": "INCOMPLETE" if failures else "PILOT_ACCEPTED",
        "failures": failures, "arms": by_arm, "independent_oracle_errors": accuracy,
        "gates": {"energy_hartree": ENERGY_GATE, "force_max_abs_hartree_per_bohr": FORCE_GATE, "physical_residual_rms": PHYSICAL_GATE},
        "diagnostic_work": {
            arm: None if profile_rows.get(arm) is None else {
                "executed_counters": profile_rows[arm].get("executed_counters"),
                "metric_diagnostics": profile_rows[arm].get("metric_diagnostics"),
                "resource_diagnostics": profile_rows[arm].get("resource_diagnostics"),
                "occupied_reuse_verified": profile_rows[arm].get("occupied_reuse_verified"),
            } for arm in ("direct", "df-jk-occupied")
        },
        "crossover_claim_eligible": False,
        "reason": "DF-J plus exact K is unsupported; one case does not cover 48/96 atom and non-water crossover requirements",
    }


def run_campaign(args: argparse.Namespace) -> None:
    """Run ABBA as independent processes, retaining every losing attempt."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("finite Slurm GPU allocation required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "kind": "native-campaign", "source": source_identity(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "order": [], "status": "RUNNING",
    }
    manifest_path = args.output / "campaign.json"
    write(manifest_path, manifest)
    for case in args.cases:
        unsupported = args.output / f"{case}-df-j-exact-k.json"
        commands = [("direct", 0), ("df-jk-occupied", 0), ("df-jk-occupied", 1), ("direct", 1), ("df-j-exact-k", 0)]
        for arm, repeat in commands:
            target = unsupported if arm == "df-j-exact-k" else args.output / f"{case}-{arm}-{repeat}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "run-native", "--case", case,
                "--arm", arm, "--repeats", "1", "--auxiliary-basis", args.auxiliary_basis,
                "--df-budget-bytes", str(args.df_budget_bytes), "--output", str(target),
            ]
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            manifest["order"].append({
                "case": case, "arm": arm, "repeat": repeat, "record": str(target),
                "process_returncode": completed.returncode, "process_seconds": time.perf_counter() - started,
            })
            write(manifest_path, manifest)
    manifest["status"] = "RECORDED"
    write(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-native")
    run.add_argument("--case", choices=("water-3", "water-48", "water-96", "formaldehyde"), required=True)
    run.add_argument("--arm", choices=tuple(ARMS), required=True)
    run.add_argument("--auxiliary-basis", default="def2-svp")
    run.add_argument("--df-budget-bytes", type=int, default=1 << 30)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--output", type=Path, required=True)
    reference = sub.add_parser("run-reference")
    reference.add_argument("--native", type=Path, required=True)
    reference.add_argument("--threads", type=int, default=8)
    reference.add_argument("--output", type=Path, required=True)
    profile = sub.add_parser("run-profile")
    profile.add_argument("--case", choices=("water-3", "water-48", "water-96", "formaldehyde"), required=True)
    profile.add_argument("--arm", choices=("direct", "df-jk-occupied"), required=True)
    profile.add_argument("--auxiliary-basis", default="def2-svp")
    profile.add_argument("--df-budget-bytes", type=int, default=1 << 30)
    profile.add_argument("--trace", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    campaign = sub.add_parser("run-campaign")
    campaign.add_argument("--cases", nargs="+", choices=("water-3", "water-48", "water-96", "formaldehyde"), required=True)
    campaign.add_argument("--auxiliary-basis", default="def2-svp")
    campaign.add_argument("--df-budget-bytes", type=int, default=1 << 30)
    campaign.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("records", nargs="+", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--oracle", action="append", type=Path, default=[])
    summary.add_argument("--profile", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "run-native":
        if args.repeats < 1 or args.df_budget_bytes < 1:
            parser.error("positive repeats and DF budget required")
        run_native(args)
    elif args.command == "run-reference":
        if args.threads < 1:
            parser.error("positive PySCF thread count required")
        run_reference(args)
    elif args.command == "run-profile":
        run_profile(args)
    elif args.command == "run-campaign":
        if args.df_budget_bytes < 1:
            parser.error("positive DF budget required")
        run_campaign(args)
    else:
        rows = [json.loads(path.read_text()) for path in args.records]
        oracle_rows = [json.loads(path.read_text()) for path in args.oracle]
        profile_rows = [json.loads(path.read_text()) for path in args.profile]
        write(args.output, summarize(rows, oracle_rows, profile_rows))


if __name__ == "__main__":
    main()
