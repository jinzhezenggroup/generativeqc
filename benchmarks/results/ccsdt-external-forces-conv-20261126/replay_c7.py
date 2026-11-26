"""Replay the C7 H200 endpoints and independently extrapolate the atom-4 z force."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from c7_fd import ATOM, AXIS, SETTINGS, STEPS
from replay import LIMITS, completed, errors
from run import LIBRARY_SHA, PROTOCOL_SHA

ROOT = Path(__file__).resolve().parent
REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"
NATIVE_ID = "3f4f4a5ff958ee10dd67138eba54369fecc55563f07147fc3b2def229c984eb2"
FD_STABILITY_LIMIT = 1e-8
Record = dict[str, Any]


def verify_record_bytes(attempts: list[Record]) -> None:
    for attempt in attempts:
        raw = (json.dumps(attempt["record"], indent=2, allow_nan=False) + "\n").encode()
        if hashlib.sha256(raw).hexdigest() != attempt["record_sha256"]:
            raise ValueError(f"Record checksum differs: {attempt['source']}")


def h200_result(attempts: list[Record], references: list[Record]) -> Record:
    accepted = []
    for attempt in attempts:
        row = attempt["record"]
        if row["status"] != "completed":
            continue
        hardware = attempt["hardware"]
        gpu = hardware["gpu"]
        if (
            "H200" not in gpu["model"]
            or gpu["compute_capability"] != "9.0"
            or hardware["source_revision"] != REVISION
            or hardware["source_identity"] != NATIVE_ID
            or hardware["library_sha256"] != LIBRARY_SHA
            or hardware["protocol_sha256"] != PROTOCOL_SHA
            or hardware["cublas_workspace_config"] != ":16:8"
            or row["work"]["executed_backend"] != "cuda"
        ):
            raise ValueError(f"Unexpected measured H200 identity: {attempt['source']}")
        accepted.append(
            {
                "source": attempt["source"],
                "trial": row["trial"],
                "endpoint_seconds": row["endpoint_seconds"],
                "energy_hartree": row["energy_hartree"],
                "force_c4z": row["force_hartree_per_bohr"][ATOM][AXIS],
                "sampled_device_peak_mib": attempt["gpu_telemetry"]["peak_used_mib"],
            }
        )
    native = completed(attempts, "C7", "native-conv")
    reference_rows = {
        arm: completed(references, "C7", arm) for arm in ("psi4-conv", "pyscf-conv")
    }
    if any(len(rows) != 1 for rows in reference_rows.values()):
        raise ValueError("Expected one qualified analytic record per CPU reference")
    pairs = {
        arm: [
            {
                "h200_trial": trial["trial"],
                "reference_trial": ref["trial"],
                **errors(trial, ref),
            }
            for trial, ref in itertools.product(native, rows)
        ]
        for arm, rows in reference_rows.items()
    }
    violations = any(
        pair[metric] > threshold
        for ref_pairs in pairs.values()
        for pair in ref_pairs
        for metric, threshold in LIMITS.items()
    )
    return {
        "status": "outside_tolerance"
        if violations
        else "passed"
        if len(native) == 2
        else "incomplete",
        "median_endpoint_seconds": statistics.median(
            sample["endpoint_seconds"] for sample in accepted
        )
        if len(accepted) == 2
        else None,
        "samples": accepted,
        "pairs": pairs,
        "reference_atom4_z": {
            arm: rows[0]["force_hartree_per_bohr"][ATOM][AXIS]
            for arm, rows in reference_rows.items()
        },
        "limits": LIMITS,
    }


def finite_difference(attempts: list[Record]) -> Record:
    energies: dict[tuple[str, float, int], float] = {}
    for attempt in attempts:
        row = attempt["record"]
        if row["status"] != "completed":
            continue
        level, step, sign = row["convergence"], row["step_bohr"], row["sign"]
        key = (level, step, sign)
        options = SETTINGS[level]
        settings = row["settings"]
        if (
            key in energies
            or (row["atom_index"], row["axis"]) != (ATOM, AXIS)
            or step not in STEPS.values()
            or sign not in {-1, 1}
            or row["psi4_version"] != "1.10.2"
            or row["fixtures_sha256"] != PROTOCOL_SHA["fixtures.json"]
            or (row["orbitals"], row["electrons"]) != (178, 58)
            or settings["scf_type"] != "pk"
            or settings["cc_type"] != "conv"
            or settings["reference"] != "rhf"
            or settings["freeze_core"]
            or not settings["puream"]
            or settings["e_convergence"] != options["energy"]
            or settings["ccenergy__e_convergence"] != options["energy"]
            or settings["d_convergence"] != options["residual"]
            or settings["ccenergy__r_convergence"] != options["residual"]
            or not math.isfinite(row["energy_hartree"])
        ):
            raise ValueError(f"Invalid FD sample: {attempt['source']}")
        energies[key] = row["energy_hartree"]

    estimates = []
    for level in SETTINGS:
        for step in STEPS.values():
            plus = energies.get((level, step, 1))
            minus = energies.get((level, step, -1))
            if plus is not None and minus is not None:
                estimates.append(
                    {
                        "convergence": level,
                        "step_bohr": step,
                        "energy_minus_hartree": minus,
                        "energy_plus_hartree": plus,
                        "force_hartree_per_bohr": (minus - plus) / (2 * step),
                    }
                )
    force = {
        (row["convergence"], row["step_bohr"]): row["force_hartree_per_bohr"]
        for row in estimates
    }
    result: Record = {"status": "incomplete", "central_differences": estimates}
    keys = [("tight", step) for step in STEPS.values()] + [("check", STEPS["h1"])]
    if all(key in force for key in keys):
        coarse, middle, fine, repeat = (force[key] for key in keys)
        richardson_coarse = (4 * middle - coarse) / 3
        richardson_fine = (4 * fine - middle) / 3
        extrapolated = (16 * richardson_fine - richardson_coarse) / 15
        spacing = abs(richardson_coarse - richardson_fine)
        energy_convergence_sensitivity = abs(fine - repeat)
        result.update(
            status="stable"
            if max(spacing, energy_convergence_sensitivity) < FD_STABILITY_LIMIT
            else "unresolved",
            richardson_h4_h2=richardson_coarse,
            richardson_h2_h1=richardson_fine,
            richardson_2level=extrapolated,
            richardson_spacing=spacing,
            tight_to_check_fine_difference=energy_convergence_sensitivity,
            observed_stability_limit=FD_STABILITY_LIMIT,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=ROOT / "c7-addendum-records.json"
    )
    parser.add_argument("--references", type=Path, default=ROOT / "records-C7.json")
    args = parser.parse_args()
    measurements = json.loads(args.records.read_text())
    references = json.loads(args.references.read_text())["attempts"]
    verify_record_bytes(
        measurements["h200"] + measurements["finite_difference"] + references
    )
    h200 = h200_result(measurements["h200"], references)
    fd = finite_difference(measurements["finite_difference"])
    if "richardson_2level" in fd:
        extrapolated = fd["richardson_2level"]
        fd["signed_analytic_minus_fd"] = {
            arm: force - extrapolated
            for arm, force in h200["reference_atom4_z"].items()
        }
        for sample in h200["samples"]:
            fd["signed_analytic_minus_fd"][f"h200-trial-{sample['trial']}"] = (
                sample["force_c4z"] - extrapolated
            )
    print(
        json.dumps({"h200": h200, "finite_difference": fd}, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
