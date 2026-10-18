"""Compare independent XC/NLC grids, retaining reference-stability uncertainty.

This is an accuracy study, not a timing publication. FP64 and Fast use separate
observable budgets; neither a reference-only pass nor agreement of one grid
with itself certifies the native implementation. Refinement differences are
empirical evidence, never rigorous bounds on the remaining integration error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc.force_aware_numerics import ObservableDelta, TargetErrorBudget

from benchmarks._readme_grid import (
    FP64_GRID_ENERGY_TOLERANCE,
    FP64_GRID_FORCE_TOLERANCE,
    parse_grid,
)

PROFILES = {
    "fp64": TargetErrorBudget(FP64_GRID_ENERGY_TOLERANCE, FP64_GRID_FORCE_TOLERANCE),
    "fast": TargetErrorBudget(1e-5, 1e-4, 3e-5),
}
DIMENSIONS = ("radial_points", "angular_polar", "angular_azimuth")


def dimensions(population: dict[str, Any], axis: str) -> tuple[int, int, int]:
    """Read the complete explicit rule, rather than a point-count alias."""
    grid = population[f"{axis}_grid"]
    return parse_grid(",".join(str(grid[name]) for name in DIMENSIONS))


def geometry_rows(population: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Reject invalid replays too, while selecting independent geometry oracles."""
    result = {}
    atoms = population["atoms"]
    if type(atoms) is not int or atoms < 1:
        raise ValueError("atom count must be a positive integer")
    seen = set()
    accuracy_only = (
        population.get("schema") == "generativeqc.reference-vv10-grid-accuracy.v1"
    )
    for row in population["records"]:
        geometry = row["geometry"]
        phase = row["phase"]
        phases = (
            (("accuracy-original",) if geometry == 0 else ("accuracy-moved",))
            if accuracy_only
            else (("cold", "warm") if geometry == 0 else ("moved", "moved-warm"))
        )
        if geometry not in (0, 1) or phase not in phases:
            raise ValueError("phase and geometry disagree")
        key = (geometry, phase, row["repeat"])
        if key in seen or row["converged"] is not True:
            raise ValueError("duplicate or unconverged endpoint")
        seen.add(key)
        forces = np.asarray(row["forces"])
        if forces.shape != (atoms, 3):
            raise ValueError("force shape differs from atom count")
        ObservableDelta.between(row["energy"], forces, row["energy"], forces)
        if phase in ("cold", "moved", "accuracy-original", "accuracy-moved"):
            if geometry in result:
                raise ValueError("duplicate geometry oracle")
            result[geometry] = row
    if set(result) != {0, 1}:
        raise ValueError("both original and moved geometries are required")
    return result


def load_population(path: Path) -> dict[str, Any]:
    """Bind complete reference endpoints to their successful process and bytes."""
    payload = path.read_bytes()
    raw = json.loads(payload)
    if (
        raw["schema"]
        not in (
            "generativeqc.reference-split-vv10.v1",
            "generativeqc.reference-vv10-grid-accuracy.v1",
        )
        or raw["engine"] != "GPU4PySCF"
        or raw["stage"] != "complete"
    ):
        raise ValueError("not a complete independent split-grid reference")
    if raw["schema"] == "generativeqc.reference-vv10-grid-accuracy.v1" and (
        raw.get("purpose") != "accuracy-only-not-a-latency-population"
        or raw.get("accuracy_protocol", {}).get("density_sources")
        != "independent-GPU4PySCF-only"
        or raw.get("accuracy_protocol", {}).get("cold_timing_qualified") is not False
    ):
        raise ValueError(
            "accuracy-only reference cannot claim cold timing or a native density seed"
        )
    outcome = path.with_suffix(".outcome")
    if not outcome.exists() or json.loads(outcome.read_text())["exit_code"] != 0:
        raise ValueError("successful process outcome is missing")
    geometry_rows(raw)
    raw["receipt"] = {
        "file": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "outcome_sha256": hashlib.sha256(outcome.read_bytes()).hexdigest(),
        "scheduler": raw["scheduler"],
        "accelerator": raw["accelerator"],
        "source_sha256": raw["source_sha256"],
    }
    return raw


def compare_populations(
    candidate: dict[str, Any], reference: dict[str, Any], budget: TargetErrorBudget
) -> dict[str, Any]:
    """Check every saved E+F replay, including RMS forces when configured."""
    for field in ("atoms", "basis", "density_fitting", "geometries_bohr"):
        if candidate[field] != reference[field]:
            raise ValueError(f"grid comparison changes {field}")
    for field in ("vv10_density_threshold", "scf_tolerances"):
        if candidate.get(field) != reference.get(field):
            raise ValueError(f"grid comparison changes {field}")
    for axis in ("xc", "nlc"):
        prescription = [
            {
                key: value
                for key, value in population[f"{axis}_grid"].items()
                if key not in DIMENSIONS
            }
            for population in (candidate, reference)
        ]
        if prescription[0] != prescription[1]:
            raise ValueError(f"grid comparison changes {axis} prescription")
    geometry_rows(candidate)
    oracles = geometry_rows(reference)
    errors = []
    for row in candidate["records"]:
        oracle = oracles[row["geometry"]]
        delta = ObservableDelta.between(
            row["energy"], row["forces"], oracle["energy"], oracle["forces"]
        )
        errors.append(
            {
                "geometry": row["geometry"],
                "phase": row["phase"],
                "repeat": row["repeat"],
                "energy_abs": delta.energy_abs,
                "force_max_abs": delta.force_max_abs,
                "force_rms": delta.force_rms,
                "gate": budget.accepts(delta),
            }
        )
    baseline = all(
        candidate[f"{axis}_grid"] == reference[f"{axis}_grid"] for axis in ("xc", "nlc")
    )
    return {
        "is_baseline": baseline,
        "observed_budget_gate": None
        if baseline
        else all(row["gate"] for row in errors),
        "replay_consistency_gate": all(row["gate"] for row in errors),
        "maximum_errors": {
            norm: max(row[norm] for row in errors)
            for norm in ("energy_abs", "force_max_abs", "force_rms")
        },
        "records": errors,
    }


def analyze_case(
    populations: dict[tuple[int, int, int], dict[str, Any]],
    *,
    axis: str,
    anchor: tuple[int, int, int],
    radial_check: tuple[int, int, int],
    angular_check: tuple[int, int, int],
    budget: TargetErrorBudget,
    anchor_fraction: float = 0.1,
) -> dict[str, Any]:
    """Require independent radial/angular stability before admitting a grid.

    The residual allowance is an empirical refinement allowance, not a proof of
    the infinite-grid limit. Keep the original comparison separate from the
    decision so incomplete anchors cannot silently turn into successful gates.
    """
    if not math.isfinite(anchor_fraction) or not 0 < anchor_fraction < 1:
        raise ValueError("anchor fraction must be between zero and one")
    if axis not in ("xc", "nlc"):
        raise ValueError("axis must be xc or nlc")
    if any(
        grid != dimensions(population, axis) for grid, population in populations.items()
    ):
        raise ValueError("population grid differs from its index")
    if radial_check[1:] != anchor[1:] or not radial_check[0] < anchor[0]:
        raise ValueError("radial check must refine radial resolution only")
    if (
        angular_check[0] != anchor[0]
        or angular_check[1:] == anchor[1:]
        or any(
            actual > target
            for actual, target in zip(angular_check[1:], anchor[1:], strict=True)
        )
    ):
        raise ValueError("angular check must refine angular resolution only")
    fixed_axis = "nlc" if axis == "xc" else "xc"
    fixed_grids = {
        dimensions(population, fixed_axis) for population in populations.values()
    }
    if len(fixed_grids) > 1:
        raise ValueError(f"isolated {axis} comparison changes {fixed_axis}")
    reference = populations.get(anchor)
    stability_budget = TargetErrorBudget(
        budget.energy_abs * anchor_fraction,
        budget.force_max_abs * anchor_fraction,
        None if budget.force_rms is None else budget.force_rms * anchor_fraction,
    )
    candidate_budget = TargetErrorBudget(
        budget.energy_abs * (1 - anchor_fraction),
        budget.force_max_abs * (1 - anchor_fraction),
        None if budget.force_rms is None else budget.force_rms * (1 - anchor_fraction),
    )
    checks = {}
    for name, grid in (("radial", radial_check), ("angular", angular_check)):
        population = populations.get(grid)
        checks[name] = (
            None
            if reference is None or population is None
            else compare_populations(population, reference, stability_budget)
        )
    gates = [
        None if check is None else check["observed_budget_gate"]
        for check in checks.values()
    ]
    replay = (
        None
        if reference is None
        else compare_populations(reference, reference, stability_budget)
    )
    checks["anchor_replay"] = replay
    gates.append(None if replay is None else replay["replay_consistency_gate"])
    stable = False if False in gates else None if None in gates else True
    comparisons = []
    for grid, population in sorted(populations.items()):
        compared = (
            None
            if reference is None
            else compare_populations(population, reference, budget)
        )
        comparisons.append(
            {
                "grid": list(grid),
                "receipt": population.get("receipt"),
                "comparison": compared,
                "reserved_budget_gate": (
                    None
                    if reference is None
                    else compare_populations(population, reference, candidate_budget)[
                        "observed_budget_gate"
                    ]
                ),
                "grid_budget_gate": (
                    None
                    if compared is None or compared["is_baseline"] or stable is None
                    else stable
                    and compare_populations(population, reference, candidate_budget)[
                        "observed_budget_gate"
                    ]
                ),
            }
        )
    return {
        "comparison_scope": f"{axis}-only-fixed-{fixed_axis}",
        "budget": asdict(budget),
        "anchor": list(anchor),
        "anchor_receipt": None if reference is None else reference.get("receipt"),
        "anchor_fraction": anchor_fraction,
        "reference_stability_budget": asdict(stability_budget),
        "candidate_budget": asdict(candidate_budget),
        "reference_stability_gate": stable,
        "reference_checks": checks,
        "error_evidence_kind": "empirical-refinement-differences-not-rigorous-bounds",
        "endpoint_precision_qualification": "not_evaluated",
        "comparisons": comparisons,
    }


def main() -> None:
    """Retain incomplete/failed measurements without admitting a partial cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--axis", choices=("xc", "nlc"), required=True)
    parser.add_argument("--fixed-grid", type=parse_grid, required=True)
    parser.add_argument("--anchor", type=parse_grid, required=True)
    parser.add_argument("--radial-check", type=parse_grid, required=True)
    parser.add_argument("--angular-check", type=parse_grid, required=True)
    parser.add_argument(
        "--sizes", type=int, choices=(3, 12, 48, 96), nargs="+", default=[3, 12]
    )
    parser.add_argument("--anchor-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {(mode, atoms): {} for mode in ("none", "cuda") for atoms in args.sizes}
    pending = []
    duplicates = set()
    fixed_axis = "nlc" if args.axis == "xc" else "xc"
    for path in sorted(args.raw.rglob("*.json")):
        try:
            raw = load_population(path)
            key = (raw["density_fitting"], raw["atoms"])
            if key not in cases or dimensions(raw, fixed_axis) != args.fixed_grid:
                continue
            grid = dimensions(raw, args.axis)
            if (key, grid) in duplicates:
                raise ValueError("ambiguous duplicate grid population")
            if grid in cases[key]:
                cases[key].pop(grid)
                duplicates.add((key, grid))
                raise ValueError(
                    "duplicate grid population; choose an explicit attempt directory"
                )
            cases[key][grid] = raw
        except (KeyError, TypeError, ValueError) as error:
            pending.append({"file": str(path), "reason": str(error)})
    result = {
        "schema": "generativeqc.vv10-grid-accuracy-study.v1",
        "pending_or_invalid": pending,
        "cases": [],
        "endpoint_precision_qualification": "not_evaluated",
    }
    for (mode, atoms), populations in cases.items():
        for profile, budget in PROFILES.items():
            result["cases"].append(
                {
                    "mode": mode,
                    "atoms": atoms,
                    "profile": profile,
                    **analyze_case(
                        populations,
                        axis=args.axis,
                        anchor=args.anchor,
                        radial_check=args.radial_check,
                        angular_check=args.angular_check,
                        budget=budget,
                        anchor_fraction=args.anchor_fraction,
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case in result["cases"]:
        admitted = [
            point["grid"]
            for point in case["comparisons"]
            if point["grid_budget_gate"] is True
        ]
        print(
            case["mode"],
            case["atoms"],
            case["profile"],
            "anchor_stable=",
            case["reference_stability_gate"],
            "grid_candidates=",
            admitted,
        )


if __name__ == "__main__":
    main()
