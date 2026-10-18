"""Audit a fresh, complete README cohort without modifying measurement journals."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks._readme_grid import grid_directory, parse_grid
from benchmarks.readme_dft import METHODS, SCHEMA, specification
from benchmarks.readme_omol25 import (
    check_record,
    complete_execution,
    reference_baselines,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

PHASES = ("cold", "warm", "moved", "moved-warm")
BASE_GRID = (48, 16, 32)
README_SIZES = (3, 12, 48, 96)


def expected_points(
    candidate_grid: tuple[int, int, int] | None = (48, 12, 24),
) -> Iterator[tuple[str, tuple[int, int, int], str, int]]:
    """Require every original method/size in both Direct and DF populations."""
    for mode in ("none", "cuda"):
        for method in METHODS:
            for atoms in README_SIZES:
                yield mode, BASE_GRID, method, atoms
        if candidate_grid is not None and candidate_grid != BASE_GRID:
            for atoms in README_SIZES:
                yield mode, candidate_grid, "wb97m-v", atoms


def physical_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Ignore only a native workspace hint when admitting an independent oracle.

    The allowance changes resource/work policy, not the Hamiltonian. It remains
    attached to its own engine and must never pool distinct timing populations.
    All basis, geometry, grid, exchange and convergence inputs still match.
    """
    result = copy.deepcopy(protocol)
    allowance = result.pop("density_fitting_memory_budget_bytes", 0)
    if type(allowance) is not int or not 0 <= allowance <= 1 << 40:
        raise ValueError("invalid native DF workspace allowance")
    if allowance and not result.get("density_fitting"):
        raise ValueError("native DF workspace allowance requires DF")
    return result


def _phase_records(raw: dict[str, Any], atoms: int) -> dict[str, list[dict]]:
    if raw.get("schema") != SCHEMA or raw["protocol"].get("atoms") != atoms:
        raise ValueError("measurement schema or atom count differs")
    if raw["protocol"].get("repeats") != 5:
        raise ValueError("publication requires five warm and moved-warm repeats")
    if raw["protocol"].get("force_return") != "host_array_in_timed_endpoint":
        raise ValueError("host forces are not part of the complete timed endpoint")
    required = {
        (phase, geometry, repeat)
        for phase, geometry, count in (
            ("cold", 0, 1),
            ("warm", 0, 5),
            ("moved", 1, 1),
            ("moved-warm", 1, 5),
        )
        for repeat in range(count)
    }
    seen = set()
    phases: dict[str, list[dict]] = {phase: [] for phase in PHASES}
    for row in raw["records"]:
        key = (row["phase"], row["geometry"], row["repeat"])
        if key not in required or key in seen:
            raise ValueError("duplicate or mislabeled endpoint")
        seen.add(key)
        phases[row["phase"]].append(row)
    return phases


def qualify_pair(
    native: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    atoms: int,
) -> dict[str, Any]:
    """Recheck saved native forces against later independent geometry oracles.

    Parallel execution may finish before its oracle. This produces a separate
    qualification view, preserving the original gate and every measured value;
    it does not pretend the journal was independently accepted at measurement.
    A known failed gate is never erased, and absent oracles remain unchecked.
    """
    if (
        native is not None
        and reference is not None
        and physical_protocol(native["protocol"])
        != physical_protocol(reference["protocol"])
    ):
        raise ValueError("independent physical protocols differ")
    scientific = reference if reference is not None else native
    point: dict[str, Any] = {"atoms": atoms, "engines": {}}
    if scientific is not None:
        point["protocol"] = copy.deepcopy(scientific["protocol"])
    baselines = (
        reference_baselines(reference, reference["protocol"], allow_missing=True)
        if reference is not None
        else {}
    )
    for engine, raw in (("native", native), ("reference", reference)):
        entry: dict[str, Any] = {"status": "not_run", "records": []}
        point["engines"][engine] = entry
        if raw is None:
            continue
        if raw.get("engine") != engine:
            raise ValueError("journal engine differs from its requested population")
        phases = _phase_records(raw, atoms)
        entry.update(
            raw_status=raw.get("status"),
            stage=raw.get("stage"),
            resources={
                "density_fitting_memory_budget_bytes": raw["protocol"].get(
                    "density_fitting_memory_budget_bytes", 0
                ),
                "native_experiment": copy.deepcopy(raw.get("native_experiment")),
            },
            phase_status={},
        )
        for phase, rows in phases.items():
            statuses = []
            for original in rows:
                row = copy.deepcopy(original)
                row["measurement_gate"] = original.get("gate")
                if (
                    not complete_execution(row, atoms)
                    or row.get("gate") is False
                    or (engine == "reference" and row.get("gate") is not True)
                    or (
                        engine == "native"
                        and row.get("gate") is None
                        and row.get("execution_gate") is not True
                    )
                ):
                    status = "failed"
                elif (oracle := baselines.get(row["geometry"])) is None:
                    status = "completed_unchecked"
                else:
                    row.update(check_record(row, oracle))
                    row["accuracy_qualification"] = (
                        "post-run-independent-reference"
                        if engine == "native"
                        else "reference-replay-consistency-not-independent-parity"
                    )
                    status = "measured" if row["gate"] else "failed"
                row["qualification_status"] = status
                entry["records"].append(row)
                statuses.append(status)
            expected_count = 5 if phase in ("warm", "moved-warm") else 1
            entry["phase_status"][phase] = (
                "failed"
                if "failed" in statuses
                else "incomplete"
                if len(statuses) != expected_count
                else "completed_unchecked"
                if "completed_unchecked" in statuses
                else "measured"
            )
        statuses = set(entry["phase_status"].values())
        entry["status"] = (
            "failed"
            if "failed" in statuses or raw.get("status") in ("failed", "unsupported")
            else "measured"
            if statuses == {"measured"}
            and raw.get("stage") == "complete"
            and raw.get("status") in ("measured", "completed_unchecked")
            else "incomplete"
        )
    return point


def audit(
    root: Path,
    plan: dict[str, Any],
    *,
    candidate_grid: tuple[int, int, int] | None = (48, 12, 24),
) -> dict[str, Any]:
    """Require fresh campaign job identities as well as complete numerical gates."""
    points = []
    for mode, grid, method, atoms in expected_points(candidate_grid):
        directory = (
            root / "measurements" / mode / grid_directory(grid) / method / str(atoms)
        )
        journals = {}
        receipts = {}
        errors = []
        failed_engines = set()
        pending_processes = set()
        for engine in ("native", "reference"):
            path = directory / f"{engine}.json"
            journals[engine] = None
            if not path.exists():
                continue
            payload = path.read_bytes()
            receipts[engine] = {
                "file": str(path.relative_to(root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            try:
                raw = json.loads(payload)
                job = raw.get("scheduler", {}).get("SLURM_JOB_ID")
                if job not in plan["allowed_job_ids"]:
                    raise ValueError("journal is not from a fresh campaign Slurm job")
                if (
                    plan.get("job_hosts")
                    and raw.get("environment", {}).get("host", {}).get("node")
                    != plan["job_hosts"][job]
                ):
                    raise ValueError("journal is not from the campaign's assigned host")
                if plan.get("started_at_utc"):
                    timestamp = datetime.fromisoformat(
                        raw["environment"]["timestamp_utc"]
                    )
                    if timestamp < datetime.fromisoformat(plan["started_at_utc"]):
                        raise ValueError("journal predates the fresh campaign")
                if (
                    engine == "native"
                    and plan.get("native_library_sha256")
                    and raw.get("native_build", {}).get("library_sha256")
                    != plan["native_library_sha256"]
                ):
                    raise ValueError(
                        "native library differs from the measured source snapshot"
                    )
                protocol = raw["protocol"]
                expected = specification(method)
                if (
                    protocol["atoms"] != atoms
                    or protocol["basis"] != expected.basis
                    or protocol["method"]
                    != ("RHF" if method == "hf" else f"{method.upper()}/RKS")
                    or protocol["density_fitting"] != (mode == "cuda")
                ):
                    raise ValueError("journal is misfiled under a different model")
                dimensions = protocol.get("grid")
                if method == "hf":
                    if dimensions is not None:
                        raise ValueError("HF cannot acquire an XC quadrature")
                else:
                    from generativeqc import GridSpec

                    expected_grid = json.loads(json.dumps(asdict(GridSpec(*grid))))
                    if dimensions != expected_grid:
                        raise ValueError("journal grid differs from its directory")
                outcome = path.with_suffix(".outcome")
                if (
                    outcome.exists()
                    and json.loads(outcome.read_text())["exit_code"] != 0
                ):
                    raise ValueError("measurement process did not exit successfully")
                if not outcome.exists():
                    pending_processes.add(engine)
                journals[engine] = raw
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{engine}: {error}")
                failed_engines.add(engine)
        try:
            point = qualify_pair(journals["native"], journals["reference"], atoms)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            point = {
                "atoms": atoms,
                "engines": {
                    engine: {"status": "failed", "records": []}
                    for engine in ("native", "reference")
                },
            }
        point.update(
            mode=mode,
            grid=list(grid),
            method=method,
            raw_receipts=receipts,
            errors=errors,
        )
        for engine in failed_engines:
            point["engines"][engine]["status"] = "failed"
        for engine in pending_processes:
            if point["engines"][engine]["status"] == "measured":
                point["engines"][engine]["status"] = "incomplete"
        points.append(point)
    statuses = Counter(
        entry["status"] for point in points for entry in point["engines"].values()
    )
    return {
        "schema": "generativeqc.readme-rerun-qualification.v1",
        "qualification_scope": "same-grid-engine-parity-only",
        "grid_convergence": "not_evaluated",
        "endpoint_precision_qualification": "not_evaluated",
        "complete": statuses == {"measured": 2 * len(points)}
        and not any(point["errors"] for point in points),
        "expected_engine_populations": 2 * len(points),
        "expected_endpoint_calls": 24 * len(points),
        "statuses": dict(statuses),
        "plan": copy.deepcopy(plan),
        "points": points,
    }


def main() -> None:
    """Write a separate progress/qualification report; never rewrite raw journals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=parse_grid, default=(48, 12, 24))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.raw, json.loads(args.plan.read_text()), candidate_grid=args.candidate_grid
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "expected_engine_populations",
                    "expected_endpoint_calls",
                    "statuses",
                )
            },
            indent=2,
        )
    )
    if not report["complete"] and not args.allow_incomplete:
        parser.exit(1, "README rerun is incomplete or failed; do not promote it.\n")


if __name__ == "__main__":
    main()
