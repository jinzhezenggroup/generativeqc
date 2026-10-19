"""Live, fail-closed WB97M-V Direct FP32 qualification; frozen receipts stay frozen.

Run reference and native engines in separate finite Slurm allocations. Existing
independent receipts require the exact complete protocol below, not case labels.
Construction, SCF, moving-grid forces, export and teardown are timed; source and
actual loaded-library attestation is excluded. Seconds sum own-work intervals
around the provenance gap; the continuous observed span is retained separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, TextIO

from generativeqc import GridSpec, KsOptions
from generativeqc._model_resolution import snapshot_basis
from generativeqc.autotune import source_identity
from generativeqc.elements import SYMBOLS
from generativeqc.ks import resolve_ks_options
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)

from benchmarks._cases import benchmark_cases
from benchmarks._retention import raw_output_path
from benchmarks.pbe0_fast_fock import portable_report
from benchmarks.pbe0_precision import errors, native_sample, reference

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "generativeqc.shared-vv10-range-fp32.v2"
GATES = {"fp64": [1e-7, 1e-5, 3e-6], "fast": [1e-5, 1e-4, 3e-5]}
ERRORS = ("energy_Eh", "max_force_Eh_per_bohr", "rms_force_Eh_per_bohr")


def protocol(
    case_name: str,
    grid: GridSpec,
    *,
    case_snapshot: Any = None,
    basis_snapshot: Any = None,
) -> dict:
    """Bind full physical inputs, resolved composition and both solver recipes."""
    case = benchmark_cases()[case_name] if case_snapshot is None else case_snapshot
    basis = (
        snapshot_basis(case.generativeqc_basis, case.basis_representation)
        if basis_snapshot is None
        else basis_snapshot
    )
    ks = resolve_ks_options("wb97m-v-rks", KsOptions(grid=grid))
    density_threshold = float(MOLECULAR_VV10_DENSITY_THRESHOLD)
    return json.loads(
        json.dumps(
            {
                "case": case_name,
                "atoms": case.atoms,
                "coordinate_units": "Bohr",
                "charge": case.charge,
                "multiplicity": case.multiplicity,
                "method": "wb97m-v-rks",
                "method_ir": ks.method_ir.semantic_payload(),
                "basis": basis.to_payload(),
                "basis_representation": case.basis_representation,
                "ao_count": sum(
                    len(shell.coefficients)
                    * (
                        2 * shell.angular_momentum + 1
                        if case.basis_representation == "spherical"
                        else (shell.angular_momentum + 1)
                        * (shell.angular_momentum + 2)
                        // 2
                    )
                    for symbol, _ in case.atoms
                    for element in basis.elements
                    if element.atomic_number == SYMBOLS.index(symbol) + 1
                    for shell in element.shells
                ),
                "density_fitting": False,
                "auxiliary_basis": None,
                "metric_threshold": None,
                "ecp": None,
                "properties": ["energy", "forces"],
                "semilocal_grid": asdict(grid),
                "nonlocal_grid": asdict(grid),
                "grid_response": True,
                "vv10_density_policy": MOLECULAR_VV10_DENSITY_POLICY,
                "vv10_density_threshold": {
                    "native_scf_and_forces": density_threshold,
                    "reference_scf_and_forces": density_threshold,
                },
                "scf_domain": ks.scf_domain,
                "xc_schedule": ks.xc_schedule,
                "native_recipe": {
                    "initial_guess": "hcore",
                    "max_iterations": 180,
                    "fp64_controls": [1e-10, 1e-8, 1e-12],
                    "fast_budget_controls": [1e-6, 1e-4, 1e-10],
                    "control_order": ["energy", "density", "screening"],
                    "fast_final_physical_closure": "source-bound native budget policy",
                },
                "reference_recipe": {
                    "backend": "gpu4pyscf",
                    "functional": "WB97M-V",
                    "threads": 4,
                    "initial_guess": "1e",
                    "max_cycle": 180,
                    "conv_tol": 1e-12,
                    "conv_tol_grad": 1e-10,
                    "direct_scf_tol": 1e-14,
                    "small_rho_cutoff": 0,
                    "grid_response": True,
                },
                "gates_Eh_maxF_RMS_Eh_per_bohr": GATES,
            },
            allow_nan=False,
        )
    )


def source_receipt() -> dict:
    """Bind core/compiler/API and the live benchmark helper bytes independently."""
    # A source-tree receipt must never attest API code imported from a wheel or
    # another checkout merely because its requested native binary has this hash.
    for name, module in tuple(sys.modules.items()):
        if name == "generativeqc" or name.startswith(
            ("generativeqc.", "generativeqc_compiler.")
        ):
            location = getattr(module, "__file__", None)
            if location is not None and not Path(location).resolve().is_relative_to(
                (ROOT / "python").resolve()
            ):
                raise ValueError(f"imported {name} is outside the attested source tree")
    return {
        "source_identity": source_identity(ROOT),
        "benchmark_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "benchmarks").glob("*.py"))
        },
    }


def validate_reference(oracle: dict, science: dict) -> None:
    """Reject contradictory inner metadata and malformed/nonfinite observables."""
    expected = {
        "converged": True,
        "reference_backend": "gpu4pyscf",
        "functional": "WB97M-V",
        "density_fitting": False,
        "grid_response": True,
        "threads": 4,
        "conv_tol": 1e-12,
        "conv_tol_grad": 1e-10,
        "direct_scf_tol": 1e-14,
        "vv10_density_threshold": {
            "scf": science["vv10_density_threshold"]["reference_scf_and_forces"],
            "forces": science["vv10_density_threshold"]["reference_scf_and_forces"],
        },
        "aos": science["ao_count"],
        "grid_points": len(science["atoms"])
        * science["semilocal_grid"]["radial_points"]
        * science["semilocal_grid"]["angular_polar"]
        * science["semilocal_grid"]["angular_azimuth"],
    }
    if any(oracle.get(key) != value for key, value in expected.items()):
        raise ValueError("independent reference metadata contradicts protocol")
    if len(oracle.get("forces", [])) != len(science["atoms"]):
        raise ValueError("reference force count differs from molecular geometry")
    errors(oracle, oracle)


def load_oracle(path: Path, science: dict) -> tuple[dict, str]:
    """Parse and hash the same immutable byte snapshot, never reread its path."""
    data = path.read_bytes()
    saved = json.loads(data)
    if saved.get("status") != "reference-completed" or saved.get("protocol") != science:
        raise ValueError("independent reference lacks the exact complete protocol")
    oracle = saved["reference"]
    validate_reference(oracle, science)
    return oracle, hashlib.sha256(data).hexdigest()


def check_sample(row: dict, oracle: dict, mode: str) -> dict:
    """Keep all independent gates and actual mixed-module work requirements."""
    measured = errors(row, oracle)
    if any(
        not math.isfinite(measured[k]) or measured[k] > limit
        for k, limit in zip(ERRORS, GATES[mode], strict=True)
    ):
        raise ValueError(f"{mode} independent energy/force gate failed: {measured}")
    if row.get("precision_evidence_error") is not None:
        raise ValueError("invalid native precision census")
    if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
        raise ValueError("invalid complete endpoint timing")
    if mode == "fast":
        validation = row.get("precision_validation") or {}
        if (
            validation.get("status") != "qualified_budget_fock"
            or len(validation.get("solves", [])) != 1
        ):
            raise ValueError("Fast must execute one qualified budget-Fock endpoint")
        for name in ("coulomb_j", "exchange_k", "nonlocal_correlation"):
            if not any(
                op.get("name") == name
                and op.get("compute") == "fp32"
                and op.get("accumulation") == "fp64"
                and op.get("count", 0) > 0
                for op in row.get("precision", {}).get("operators", [])
            ):
                raise ValueError(f"missing executed FP32 {name}")
    return measured


def check_build(build: dict, sources: dict, previous: dict | None) -> None:
    """Attest each actual candidate, not a separately constructed proxy owner."""
    digest = build.get("library_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("missing loaded-library digest")
    if build.get("probe", {}).get("source_identity") != sources["source_identity"]:
        raise ValueError("loaded library does not match current immutable source")
    if previous is not None and build != previous:
        raise ValueError("loaded native build changed between endpoint samples")


def _run(args: argparse.Namespace, output: TextIO) -> None:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "performance_claim": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "summed own-work: fresh-owner construction+SCF+moving-grid forces+export+teardown; excluded provenance gap; continuous observed span retained separately",
        "performance_acceptance": "unassessed: no preregistered performance gate",
        "records": [],
    }

    def save() -> None:
        output.seek(0)
        output.write(
            json.dumps(portable_report(report), indent=2, allow_nan=False) + "\n"
        )
        output.truncate()
        output.flush()

    save()
    try:
        grid = GridSpec(48, 16, 32)
        case = benchmark_cases()[args.case]
        basis = snapshot_basis(case.generativeqc_basis, case.basis_representation)
        science = protocol(args.case, grid, case_snapshot=case, basis_snapshot=basis)
        report.update(protocol=science, source=source_receipt())
        save()
        if args.engine == "reference":
            attempt = {"engine": "gpu4pyscf", "attempt_status": "running"}
            report["records"].append(attempt)
            save()
            report["reference"] = reference(
                args.case,
                grid,
                4,
                functional="WB97M-V",
                backend="gpu4pyscf",
                atoms=case.atoms,
                basis=basis,
                case_snapshot=case,
                vv10_density_threshold=science["vv10_density_threshold"][
                    "reference_scf_and_forces"
                ],
            )
            save()
            validate_reference(report["reference"], science)
            attempt["attempt_status"] = "completed"
            report["status"] = "reference-completed"
        else:
            oracle, oracle_hash = load_oracle(args.oracle, science)
            report.update(oracle=oracle, oracle_sha256=oracle_hash)
            save()
            options = {
                "method": "wb97m-v-rks",
                "device": "cuda",
                "basis": basis,
                "basis_representation": case.basis_representation,
                "ks_options": KsOptions(grid=grid),
                "initial_guess": None,
                "max_iterations": 180,
            }
            previous = None
            for repeat in range(args.repeats + 1):
                modes = ("fp64", "fast") if repeat % 2 == 0 else ("fast", "fp64")
                for mode in modes:
                    attempt = {
                        "mode": mode,
                        "repeat": repeat,
                        "first_call": repeat == 0,
                        "attempt_status": "running",
                    }
                    report["records"].append(attempt)
                    save()
                    attempt.update(
                        native_sample(
                            mode,
                            options,
                            args.case,
                            include_build_metadata=True,
                            case_snapshot=case,
                        )
                    )
                    save()
                    if attempt["native_build"] != attempt.get("native_build_before"):
                        raise ValueError("loaded native build changed during endpoint")
                    check_build(attempt["native_build"], report["source"], previous)
                    previous = attempt["native_build"]
                    attempt["errors"] = check_sample(attempt, oracle, mode)
                    if source_receipt() != report["source"]:
                        raise ValueError("source changed during qualification")
                    attempt["attempt_status"] = "qualified"
                    save()
            times = {
                mode: median(
                    row["seconds"]
                    for row in report["records"]
                    if row["mode"] == mode and not row["first_call"]
                )
                for mode in ("fp64", "fast")
            }
            report.update(
                status="numerically-qualified",
                medians_seconds=times,
                speedup_fp64_over_fast=times["fp64"] / times["fast"],
            )
        if source_receipt() != report["source"]:
            raise ValueError("source changed during qualification")
        report["performance_observation_available"] = args.engine == "native"
    except BaseException as error:
        report.update(
            status="failed",
            performance_claim=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if report["records"] and report["records"][-1]["attempt_status"] == "running":
            report["records"][-1].update(attempt_status="failed", error=report["error"])
        raise
    finally:
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("reference", "native"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("requires a finite Slurm allocation")
    if args.repeats < 1 or (args.engine == "native" and args.oracle is None):
        parser.error("positive repeats and a native --oracle are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.output.open("x+", encoding="utf-8")
    except FileExistsError:
        parser.error("output must be new, including symlinks")
    with output:
        _run(args, output)


if __name__ == "__main__":
    main()
