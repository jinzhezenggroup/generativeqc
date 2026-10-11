"""Independent E+F acceptance and same-device complete-endpoint A/B gates.

The oracle subcommand runs CPU PySCF/libcint, outside production wall timing.
The compare subcommand refuses incomplete, mismatched or changed-work receipts;
diagnostic bucket cost models alone never qualify a production speedup.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.selective_eri_rhf_endpoint import digest

PHASES = ("cold", "warm", "moved", "moved-warm")
ENERGY_ATOL = 1e-8
FORCE_ATOL = 1e-6
PAIRED_ENERGY_ATOL = 1e-9
PAIRED_FORCE_ATOL = 1e-7


def make_oracle(input_path: Path) -> dict:
    """Independently converge both geometries, including all nuclear forces."""
    import numpy as np
    import pyscf
    from pyscf import scf

    from benchmarks.rhf_resident_jk import read_input
    from tools.generate_validation_references import pyscf_molecule

    report = {
        "schema": "selective-eri-rhf-oracle-v1",
        "input_sha256": digest(input_path),
        "pyscf_version": pyscf.__version__,
        "energy_tolerance": 1e-12,
        "gradient_tolerance": 1e-9,
        "screening_tolerance": 0.0,
        "moved_displacement_bohr": 0.05,
        "samples": {},
    }
    for phase, displacement in [("base", 0.0), ("moved", 0.05)]:
        inputs = read_input(input_path, displacement)["inputs"]
        inputs["multiplicity"] = 1
        # libcint groups shells by atom/angular momentum; complete molecular
        # energies/forces are invariant to this AO permutation (no matrix reuse).
        inputs["shells"] = sorted(
            inputs["shells"],
            key=lambda shell: (shell["atom_index"], shell["angular_momentum"]),
        )
        molecule, _, _ = pyscf_molecule(inputs)
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = report["energy_tolerance"]
        mean_field.conv_tol_grad = report["gradient_tolerance"]
        mean_field.direct_scf_tol = 0.0
        mean_field.max_cycle = 100
        mean_field.chkfile = None
        mean_field.kernel()
        forces = -mean_field.nuc_grad_method().kernel()
        if (
            not mean_field.converged
            or not math.isfinite(mean_field.e_tot)
            or not np.isfinite(forces).all()
        ):
            raise ValueError(f"independent {phase} RHF E+F did not converge")
        report["samples"][phase] = {
            "energy": float(mean_field.e_tot),
            "forces": forces.ravel().tolist(),
        }
    report["status"] = "complete"
    return report


def max_difference(first: list, second: list) -> float:
    """Reject shape drift and NaNs before applying absolute scientific gates."""
    if len(first) != len(second) or not first:
        raise ValueError("force destinations differ or are empty")
    if not all(math.isfinite(value) for value in [*first, *second]):
        raise ValueError("nonfinite force destination")
    return max(abs(left - right) for left, right in zip(first, second, strict=True))


def compare(baseline: list[dict], candidate: list[dict], oracle: dict) -> dict:
    """Require three fresh pairs for a scoped experimental PR, not global promotion.

    Admission is conservative: median Cold E+F must improve by at least 2%,
    and no median phase may regress by more than 3%. Raw per-pair timings,
    numerical errors and exact work counts remain visible even after refusal.
    """
    if len(baseline) != len(candidate) or len(baseline) < 3:
        raise ValueError("requires at least three equally sized fresh-process pairs")
    if oracle.get("status") != "complete":
        raise ValueError("independent oracle is incomplete")
    identity_keys = (
        "input_sha256",
        "geometry_hash",
        "basis_hash",
        "nbf",
        "budget_bytes",
        "max_iterations",
        "tolerance",
        "precision",
        "screening_tolerance",
        "method",
        "moved_displacement_bohr",
        "device",
        "slurm_job_id",
        "timing_scope",
    )
    first = baseline[0]
    if (
        first["input_sha256"] != oracle["input_sha256"]
        or first["moved_displacement_bohr"] != oracle["moved_displacement_bohr"]
        or first["precision"] != "fp64"
        or first["screening_tolerance"] != 0.0
        or first["method"] != "bounded-physical-reference-RHF"
    ):
        raise ValueError("oracle/endpoint scientific identity mismatch")
    baseline_native, candidate_native = (
        first["native_sha256"],
        candidate[0]["native_sha256"],
    )
    if baseline_native == candidate_native:
        raise ValueError("baseline and candidate use the same native provider")
    rows, failures = [], []
    for pair, (old, new) in enumerate(zip(baseline, candidate, strict=True)):
        for receipt, native in [(old, baseline_native), (new, candidate_native)]:
            if receipt.get("status") != "complete":
                raise ValueError("incomplete endpoint receipt")
            if any(receipt[key] != first[key] for key in identity_keys):
                raise ValueError("paired endpoint identity mismatch")
            if receipt["native_sha256"] != native:
                raise ValueError("provider changed between repetitions")
            route_first = first if native == baseline_native else candidate[0]
            if receipt["controls"] != route_first["controls"]:
                raise ValueError("resident policy changed between repetitions")
            if receipt["controls"]["GENERATIVEQC_RHF_RESIDENT_VALUES"] != "auto":
                raise ValueError(
                    "requires actual automatic baseline routing, not ablation"
                )
            if [sample["phase"] for sample in receipt["samples"]] != list(PHASES):
                raise ValueError(
                    "requires one complete four-phase solve per fresh process"
                )
        if any(
            old["controls"].get(name) is not None
            for name in (
                "GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK",
                "GENERATIVEQC_RHF_RESIDENT_MAX_BYTES",
            )
        ):
            raise ValueError(
                "baseline does not retain unchanged master residency controls"
            )
        for old_sample, new_sample in zip(old["samples"], new["samples"], strict=True):
            phase = old_sample["phase"]
            reference = oracle["samples"][
                "moved" if phase.startswith("moved") else "base"
            ]
            if not math.isfinite(reference["energy"]):
                raise ValueError("nonfinite independent oracle energy")
            errors = []
            for sample in [old_sample, new_sample]:
                if not math.isfinite(sample["energy"]):
                    raise ValueError("nonfinite endpoint energy")
                if not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
                    raise ValueError("invalid complete endpoint timing")
                energy_error = abs(sample["energy"] - reference["energy"])
                force_error = max_difference(sample["forces"], reference["forces"])
                errors.append({"energy": energy_error, "force_max": force_error})
                if energy_error > ENERGY_ATOL or force_error > FORCE_ATOL:
                    failures.append(f"pair {pair} {phase}: independent E+F gate")
            if (
                abs(old_sample["energy"] - new_sample["energy"]) > PAIRED_ENERGY_ATOL
                or max_difference(old_sample["forces"], new_sample["forces"])
                > PAIRED_FORCE_ATOL
            ):
                failures.append(f"pair {pair} {phase}: paired E+F gate")
            counts = (
                "iterations",
                "physical_fock_builds",
                "execution_plan_reused",
                "response_jk_actions",
                "response_derivative_passes",
                "response_orbital_actions",
            )
            if any(old_sample[key] != new_sample[key] for key in counts):
                failures.append(f"pair {pair} {phase}: changed semantic work")
            rows.append(
                {
                    "pair": pair,
                    "phase": phase,
                    "baseline_seconds": old_sample["seconds"],
                    "candidate_seconds": new_sample["seconds"],
                    "gain_fraction": 1 - new_sample["seconds"] / old_sample["seconds"],
                    "baseline_work": {key: old_sample[key] for key in counts},
                    "candidate_work": {key: new_sample[key] for key in counts},
                    "independent_errors": errors,
                }
            )
    phases = {}
    for phase in PHASES:
        samples = [row for row in rows if row["phase"] == phase]
        old_median = statistics.median(row["baseline_seconds"] for row in samples)
        new_median = statistics.median(row["candidate_seconds"] for row in samples)
        gain = 1 - new_median / old_median
        phases[phase] = {
            "baseline_median_seconds": old_median,
            "candidate_median_seconds": new_median,
            "gain_fraction": gain,
            "pair_gain_min": min(row["gain_fraction"] for row in samples),
            "pair_gain_max": max(row["gain_fraction"] for row in samples),
        }
        if gain < -0.03 or (phase == "cold" and gain < 0.02):
            failures.append(f"{phase}: complete-endpoint performance gate")
    return {
        "schema": "selective-eri-rhf-acceptance-v1",
        "qualification_scope": "local experimental mask; no shared formal/global default promotion",
        "qualified": not failures,
        "failures": failures,
        "baseline_native_sha256": baseline_native,
        "candidate_native_sha256": candidate_native,
        "identity": {key: first[key] for key in identity_keys},
        "numerical_gates": {
            "independent_energy_atol": ENERGY_ATOL,
            "independent_force_atol": FORCE_ATOL,
            "paired_energy_atol": PAIRED_ENERGY_ATOL,
            "paired_force_atol": PAIRED_FORCE_ATOL,
        },
        "phases": phases,
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    oracle_parser = commands.add_parser("oracle")
    oracle_parser.add_argument("--input", type=Path, required=True)
    oracle_parser.add_argument("--output", type=raw_output_path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", nargs="+", type=Path, required=True)
    comparison.add_argument("--candidate", nargs="+", type=Path, required=True)
    comparison.add_argument("--oracle", type=Path, required=True)
    comparison.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.command == "oracle":
        report = make_oracle(args.input)
    else:
        load = lambda path: json.loads(path.read_text())
        report = compare(
            list(map(load, args.baseline)),
            list(map(load, args.candidate)),
            load(args.oracle),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
