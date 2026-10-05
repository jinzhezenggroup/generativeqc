"""Recompute acceptance and timing ratios from the retained cold endpoints.

Run after collecting all five JSON records. Optional census availability is
respected: a raw zero without a measured flag is not scientific work evidence.
The two force configurations within each stage share a binary and GPU; the
legacy/two-pass comparison crosses binaries and allocations on the same GPU.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ("legacy", "canonical", "energy", "symmetric", "screened-1e-12")
FD_REFERENCES = (0.01808580748274835, 0.01808581799878084)
PHASES = (
    "setup",
    "reference_audit",
    "weights",
    "solve",
    "independent_audit",
    "one_electron",
    "two_electron",
)


def validate(record):
    """Keep independent FD, physical audits and optional work separate."""
    if not record["forces_requested"]:
        return {"force_validation": None, "orbital_work": None}
    force = record["forces"]
    assert len(force) == 24 and all(math.isfinite(x) for x in force)
    derivative = -(force[2] - force[14]) / math.sqrt(2)
    errors = [abs(derivative - ref) for ref in FD_REFERENCES]
    census = bool(record["orbital_derivative_census_measured"])
    screened = record["orbital_applied_screening"] > 0
    return {
        "force_validation": {
            "oracle": "../df-lambda-gemm-20261004/oracle-energy-fd.json",
            "oracle_recomputed": False,
            "steps_bohr": [1e-4, 3e-5],
            "reference_directional_derivatives": FD_REFERENCES,
            "analytic_directional_derivative": derivative,
            "absolute_errors": errors,
            "fd_gate": 3e-7,
            "fd_passed": all(error <= 3e-7 for error in errors),
            "z_residual": record["z_residual"],
            "z_gate": 1e-10,
            "z_passed": record["z_residual"] <= 1e-10,
            "stationarity": record["stationarity"],
            "stationarity_gate": 1e-8,
            "stationarity_passed": record["stationarity"] <= 1e-8,
            "force_sum": [sum(force[k::3]) for k in range(3)],
        },
        "orbital_work": {
            "derivative_passes": record["orbital_derivative_passes"],
            "shell_passes": record["orbital_shell_derivative_passes"],
            "generic_passes": record["orbital_generic_derivative_passes"],
            "shell_internal_quartet_visits": None,
            "canonical_quartet_visits": (
                record["orbital_derivative_quartet_visits"] if census else None
            ),
            "canonical_three_axis_jets": (
                record["orbital_derivative_jet_evaluations"] if census else None
            ),
            "jk_census_actions": record["orbital_jk_census_actions"],
            "jk_canonical_quartet_visits": record["orbital_jk_quartet_visits"],
            "jk_eri_evaluations": record["orbital_jk_eri_evaluations"],
            "hardware_flops": None,
        },
        "screening": {
            "requested_threshold": record["orbital_requested_screening"],
            "applied_threshold": record["orbital_applied_screening"],
            "available": bool(record["orbital_linear_screening_available"]),
            "provisional_converged": (
                bool(record["orbital_screened_converged"]) if screened else None
            ),
            "provisional_physical_residual": (
                record["orbital_screened_residual"]
                if screened and record["orbital_screened_converged"]
                else None
            ),
            "provisional_iterations": (
                record["orbital_screened_iterations"] if screened else None
            ),
            "screened_jk_actions": record["orbital_screened_jk_actions"],
            "exact_jk_actions_including_audits_and_weights": record["jk_actions"]
            - record["orbital_screened_jk_actions"],
            "exact_corrective_solves": record["orbital_exact_refinements"],
        },
        "orbital_unassigned_seconds": record["orbital_seconds"]
        - sum(record[f"orbital_{phase}_seconds"] for phase in PHASES),
    }


def compare(before, after, *, same_binary_and_allocation):
    """Report every cold pair, retaining failed gates without filtering."""
    error = max(
        abs(a - b) for a, b in zip(before["forces"], after["forces"], strict=True)
    )
    times = ("native", "reference", "orbital", "orbital_two_electron", "orbital_solve")
    return {
        "same_gpu_uuid": True,
        "same_binary_and_allocation": same_binary_and_allocation,
        "cold_force_max_abs_difference": error,
        "cold_force_gate": 3e-9,
        "cold_force_passed": error <= 3e-9,
        "energy_abs_difference": abs(before["total_energy"] - after["total_energy"]),
        "speedup_ratios": {
            phase: before[f"{phase}_seconds"] / after[f"{phase}_seconds"]
            for phase in times
        },
        "seconds_saved": {
            phase: before[f"{phase}_seconds"] - after[f"{phase}_seconds"]
            for phase in times
        },
    }


def main():
    """Write a deterministic summary, preserving raw evidence separately."""
    records = {name: json.loads((ROOT / f"{name}.json").read_text()) for name in NAMES}
    legacy = records["legacy"]
    total = legacy["native_seconds"]
    force_names = [name for name in NAMES if records[name]["forces_requested"]]
    pair_errors = {
        f"{first}_vs_{second}": max(
            abs(a - b)
            for a, b in zip(records[first]["forces"], records[second]["forces"], strict=True)
        )
        for first, second in itertools.combinations(force_names, 2)
    }
    result = {
        "schema": "generativeqc.rhf-orbital-response.v2",
        "validation": {name: validate(record) for name, record in records.items()},
        "comparisons": {
            "legacy_to_canonical": compare(
                legacy, records["canonical"], same_binary_and_allocation=True
            ),
            "legacy_to_symmetric": compare(
                legacy, records["symmetric"], same_binary_and_allocation=False
            ),
            "symmetric_to_screened": compare(
                records["symmetric"],
                records["screened-1e-12"],
                same_binary_and_allocation=True,
            ),
        },
        "legacy_amdahl_ideal_upper_bounds": {
            "remove_all_nuclear_two_electron": total
            / (total - legacy["orbital_two_electron_seconds"]),
            "remove_entire_z_solve": total / (total - legacy["orbital_solve_seconds"]),
            "remove_all_jk_including_audits_and_weights": total
            / (total - legacy["orbital_jk_seconds"]),
            "note": "Idealized bounds, not achieved speedups or screening predictions.",
        },
        "all_cold_force_pairs": {
            "maximum_component_errors": pair_errors,
            "maximum": max(pair_errors.values()),
            "gate": 3e-9,
            "passed": all(error <= 3e-9 for error in pair_errors.values()),
            "note": "Includes rejected canonical and all retained force schedules; no sample filtered by iteration count or timing.",
        },
        "limitations": [
            "One cold sample per configuration; no timing confidence interval.",
            "Independent RHF/source/CCSD solves introduce force and timing variability.",
            "Cold energy/force total differences are not an exact incremental force cost.",
            "J/K times overlap enclosing phases and must not be added to them.",
            "Unmeasured shell work and hardware FLOPs remain null.",
            "Large source-factor atol=rtol=3e-10 qualification remains outstanding.",
            "Global RHF stability remains uncertified.",
            "Earlier #1829 cold-pair discrepancy remains a separate unresolved record.",
        ],
        "raw_sha256": {
            f"{name}.json": hashlib.sha256((ROOT / f"{name}.json").read_bytes()).hexdigest()
            for name in NAMES
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
