"""Qualify total WB97M-V E+F error against a joint independent anchor.

XC-only, NLC-only and same-grid arithmetic passes are not endpoint precision
passes. All five reference controls here reconverge at the *same* dense anchor:
two XC refinements, two NLC refinements and a lower density-screening threshold.
Their summed differences reserve an empirical uncertainty allowance; they are
not rigorous bounds on the infinite-grid or unscreened limit. This tool never
qualifies latency populations or rewrites the original measurement gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from generativeqc.force_aware_numerics import ObservableDelta

from benchmarks.readme_dft import SCHEMA
from benchmarks.readme_omol25 import complete_execution
from tools.analyze_vv10_grid_study import (
    DIMENSIONS,
    PROFILES,
    accuracy_contract,
    dimensions,
    geometry_rows,
    load_population,
)

CONTROLS = ("xc_radial", "xc_angular", "nlc_radial", "nlc_angular", "screening")
NORMS = ("energy_abs", "force_max_abs", "force_rms")


def _same_reference_problem(candidate: dict, anchor: dict) -> None:
    """Allow resolution/domain changes, not a different Hamiltonian or solver."""
    for field in (
        "schema",
        "engine",
        "atoms",
        "basis",
        "density_fitting",
        "geometries_bohr",
        "source_sha256",
    ):
        if candidate[field] != anchor[field]:
            raise ValueError(f"reference control changes {field}")
    for axis in ("xc", "nlc"):
        rules = [
            {
                key: value
                for key, value in population[f"{axis}_grid"].items()
                if key not in DIMENSIONS
            }
            for population in (candidate, anchor)
        ]
        if rules[0] != rules[1]:
            raise ValueError(f"reference control changes {axis} prescription")
    contracts = [accuracy_contract(population) for population in (candidate, anchor)]
    for contract in contracts:
        contract.pop("vv10_density_threshold")
    if contracts[0] != contracts[1]:
        raise ValueError("reference control changes SCF/Fock/response policy")


def _check_control(name: str, control: dict, anchor: dict) -> None:
    """Each experiment varies exactly one independent error source."""
    _same_reference_problem(control, anchor)
    thresholds = [
        accuracy_contract(population)["vv10_density_threshold"]
        for population in (control, anchor)
    ]
    if name == "screening":
        if not thresholds[0] < thresholds[1]:
            raise ValueError("screening control must lower the anchor threshold")
        varied_axis = None
    else:
        if thresholds[0] != thresholds[1]:
            raise ValueError("grid control changes density screening")
        varied_axis, direction = name.split("_")
        check, target = (
            dimensions(population, varied_axis) for population in (control, anchor)
        )
        valid = (
            check[1:] == target[1:] and check[0] < target[0]
            if direction == "radial"
            else check[0] == target[0]
            and check[1:] != target[1:]
            and all(
                actual <= final
                for actual, final in zip(check[1:], target[1:], strict=True)
            )
        )
        if not valid:
            raise ValueError(f"invalid {name} refinement")
    for axis in ("xc", "nlc"):
        if axis != varied_axis and control[f"{axis}_grid"] != anchor[f"{axis}_grid"]:
            raise ValueError(f"{name} control changes fixed {axis} grid")


def _differences(candidate: dict, anchor: dict) -> list[dict[str, Any]]:
    oracles = geometry_rows(anchor)
    result = []
    for row in candidate["records"]:
        oracle = oracles[row["geometry"]]
        delta = ObservableDelta.between(
            row["energy"], row["forces"], oracle["energy"], oracle["forces"]
        )
        result.append(
            {
                "geometry": row["geometry"],
                "phase": row["phase"],
                "repeat": row["repeat"],
                **{norm: getattr(delta, norm) for norm in NORMS},
            }
        )
    return result


def _maximum(rows: list[dict]) -> dict[str, float]:
    return {norm: max(row[norm] for row in rows) for norm in NORMS}


def qualify_endpoint(
    candidate: dict,
    anchor: dict,
    controls: dict[str, dict],
    *,
    profile: str,
    anchor_fraction: float = 0.1,
) -> dict[str, Any]:
    """Apply one total observable budget to every cold/warm/moved endpoint.

    Endpoint error is measured directly against the joint anchor, so quadrature,
    screening, SCF and arithmetic cannot each consume an independent full budget.
    The sum of all reference-control norms (including anchor replay drift) must
    fit the reserved fraction and is added to each endpoint error before gating.
    GPU4PySCF timing journals are checked too, without calling their FP64
    arithmetic an execution of the native Fast policy.
    """
    if not math.isfinite(anchor_fraction) or not 0 < anchor_fraction < 1:
        raise ValueError("anchor fraction must be between zero and one")
    if set(controls) != set(CONTROLS):
        raise ValueError("joint XC/NLC/screening reference controls are required")
    if (
        anchor["schema"] != "generativeqc.reference-vv10-grid-accuracy.v1"
        or accuracy_contract(anchor)["fock_policy"] != "full-density-rebuild"
    ):
        raise ValueError(
            "joint reference requires explicit full-Fock accuracy controls"
        )
    tolerances = accuracy_contract(anchor)["scf_tolerances"]
    if any(
        tolerances[name] > limit
        for name, limit in (
            ("energy", 1e-12),
            ("gradient", 1e-10),
            ("direct_scf", 1e-14),
        )
    ):
        raise ValueError(
            "reference SCF tolerances are too loose for endpoint qualification"
        )
    geometry_rows(anchor)
    evidence = {}
    for name, control in controls.items():
        _check_control(name, control, anchor)
        geometry_rows(control)
        evidence[name] = {
            "maximum_errors": _maximum(_differences(control, anchor)),
            "receipt": control.get("receipt"),
        }
    evidence["anchor_replay"] = {
        "maximum_errors": _maximum(_differences(anchor, anchor)),
        "receipt": anchor.get("receipt"),
    }
    uncertainty = {
        norm: sum(check["maximum_errors"][norm] for check in evidence.values())
        for norm in NORMS
    }
    budget = asdict(PROFILES[profile])
    stability_gate = all(
        limit is None or uncertainty[norm] <= limit * anchor_fraction
        for norm, limit in budget.items()
    )
    protocol = candidate["protocol"]
    is_native = candidate["engine"] == "native"
    if (
        candidate["schema"]
        != SCHEMA + (".fast.v1" if is_native and profile == "fast" else "")
        or candidate["engine"] not in ("native", "reference")
        or candidate["stage"] != "complete"
        or protocol["method"] != "WB97M-V/RKS"
        or protocol["basis"] != "def2-SVP"
        or protocol["representation"] != "spherical"
        or anchor["basis"] != "spherical def2-SVP"
        or protocol["atoms"] != anchor["atoms"]
        or protocol["density_fitting"] != (anchor["density_fitting"] == "cuda")
        or protocol["geometries_bohr"] != anchor["geometries_bohr"]
        or protocol["repeats"] != 5
        or protocol["force_return"] != "host_array_in_timed_endpoint"
    ):
        raise ValueError(
            "candidate/reference method, profile or endpoint contract differs"
        )
    if is_native:
        experiment = candidate["native_experiment"]
        if experiment["precision_policy"] != profile:
            raise ValueError("native execution profile differs")
        if profile == "fast" and (
            candidate["application_budget"] != budget
            or experiment.get("solver_control_policy") != "application-policy-selected"
            or experiment.get("requested_solver_controls") != {"max_iterations": 100}
            or experiment.get("public_initialization_policy") is not True
            or experiment.get("preliminary_provider") != "none"
            or any(
                experiment.get(key) is not None
                for key in (
                    "force_active_ao_cutoff",
                    "force_active_ao_cache_bytes",
                    "force_max_device_bytes",
                    "force_max_host_bytes",
                )
            )
        ):
            raise ValueError("native Fast application policy or public controls differ")
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
    precision_counts: Counter = Counter()
    for row in candidate["records"]:
        key = (row["phase"], row["geometry"], row["repeat"])
        if (
            key not in required
            or key in seen
            or not complete_execution(row, anchor["atoms"])
        ):
            raise ValueError("missing, duplicate, failed or mislabeled endpoint")
        seen.add(key)
        if is_native and profile == "fast":
            bits = row["precision"]["effective_bits"]
            if bits not in (32, 64):
                raise ValueError("actual native Fast precision is missing")
            precision_counts[str(bits)] += 1
    if seen != required:
        raise ValueError("all twelve E+F endpoints are required")
    rows = _differences(candidate, anchor)
    for row in rows:
        row["with_reference_allowance"] = {
            norm: row[norm] + uncertainty[norm] for norm in NORMS
        }
        row["gate"] = stability_gate and all(
            limit is None or row["with_reference_allowance"][norm] <= limit
            for norm, limit in budget.items()
        )
    return {
        "schema": "generativeqc.vv10-endpoint-precision.v1",
        "qualification_scope": "joint-total-E+F-empirical-reference",
        "error_evidence_kind": "empirical-refinement-differences-not-rigorous-bounds",
        "timing_qualification": "not_evaluated",
        "profile": profile,
        "engine": candidate["engine"],
        "budget": budget,
        "reference_allowance_fraction": anchor_fraction,
        "reference_uncertainty": uncertainty,
        "reference_stability_gate": stability_gate,
        "reference_controls": evidence,
        "maximum_errors": _maximum(rows),
        "maximum_errors_with_reference_allowance": {
            norm: max(row["with_reference_allowance"][norm] for row in rows)
            for norm in NORMS
        },
        "total_endpoint_gate": all(row["gate"] for row in rows),
        "candidate_receipt": candidate.get("receipt"),
        "actual_fast_precision_counts": dict(precision_counts),
        "records": rows,
    }


def main() -> None:
    """Keep missing evidence distinct from a measured failing precision gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", "--native", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    for name in CONTROLS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--anchor-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any]
    try:
        payload = args.candidate.read_bytes()
        candidate = json.loads(payload)
        outcome = args.candidate.with_suffix(".outcome").read_bytes()
        if json.loads(outcome)["exit_code"] != 0:
            raise ValueError("candidate process did not complete successfully")
        if not (
            candidate["scheduler"]["SLURM_JOB_ID"] and candidate["source_file_sha256"]
        ):
            raise ValueError("candidate scheduler/source receipts are missing")
        candidate["receipt"] = {
            "file": str(args.candidate),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "outcome_sha256": hashlib.sha256(outcome).hexdigest(),
            "scheduler": candidate["scheduler"],
            "source_file_sha256": candidate["source_file_sha256"],
        }
        if candidate["engine"] == "native":
            if not candidate["native_build"]["library_sha256"]:
                raise ValueError("native build receipt is missing")
            candidate["receipt"]["native_build"] = candidate["native_build"]
        result = qualify_endpoint(
            candidate,
            load_population(args.anchor),
            {name: load_population(getattr(args, name)) for name in CONTROLS},
            profile=args.profile,
            anchor_fraction=args.anchor_fraction,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result = {"total_endpoint_gate": None, "incomplete_or_invalid": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("total_endpoint_gate=", result["total_endpoint_gate"])
    if result["total_endpoint_gate"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
