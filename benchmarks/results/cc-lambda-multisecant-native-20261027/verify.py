"""Replay frozen native multisecant evidence without executing a GPU endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]


def read_json(path: Path) -> dict:
    """Decode the original receipt bytes, including compressed publications."""
    raw = path.read_bytes()
    return json.loads(lzma.decompress(raw) if path.suffix == ".xz" else raw)


def reconstruct_sources(identity: dict) -> None:
    """Optionally check measured historical sources, never current source bodies."""
    with tempfile.TemporaryDirectory(prefix="qc-multisecant-evidence-") as temporary:
        directory = Path(temporary)
        for name in identity["variants"]["strict"]:
            if name.endswith("multisecant_probe.hpp"):
                continue
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(
                    ["git", "show", identity["base_revision"] + ":" + name], cwd=ROOT
                )
            )
        for variant, filename in (
            ("strict", "strict-candidate.patch.xz"),
            ("warmup", "warmup-addition.patch.xz"),
        ):
            patch = directory / "candidate.patch"
            patch.write_bytes(lzma.decompress((DIRECTORY / filename).read_bytes()))
            subprocess.run(
                ["patch", "--batch", "-p1", "-i", str(patch)],
                cwd=directory,
                check=True,
                capture_output=True,
            )
            for name, expected in identity["variants"][variant].items():
                if (
                    hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError(f"historical source reconstruction failed: {name}")


def verify() -> dict:
    """Check provenance, unchanged controls, complete endpoints and observables."""
    qualification = read_json(DIRECTORY / "qualification.json")
    for name, expected in qualification["member_sha256"].items():
        if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"retained member checksum mismatch: {name}")
    reference_path = ROOT / qualification["independent_reference"]["path"]
    if (
        hashlib.sha256(reference_path.read_bytes()).hexdigest()
        != qualification["independent_reference"]["sha256"]
    ):
        raise ValueError("independent reference checksum mismatch")
    reference = read_json(reference_path)
    populations = []
    energy_errors, force_errors, independent_errors = [], [], []
    lambda_errors = []
    observations = 0
    for binding in qualification["records"]:
        record = read_json(DIRECTORY / binding["path"])
        if (
            record["status"] != "completed"
            or record["library_sha256"] != binding["library_sha256"]
        ):
            raise ValueError("incomplete or incorrectly bound native endpoint record")
        driver = lzma.decompress((DIRECTORY / binding["driver"]).read_bytes())
        if hashlib.sha256(driver).hexdigest() != record["driver_sha256"]:
            raise ValueError("measured driver identity mismatch")
        if (
            record["base_source"] != qualification["base_revision"]
            or not record["slurm_job_id"]
            or record["slurm_job_id"] != binding["slurm_job_id"]
            or len(record["samples"]) != binding["samples"]
        ):
            raise ValueError("incorrect source/scheduler/population provenance")
        if (
            record["arguments"]["policy"] != "standard"
            or record["arguments"]["preconvergence"] != "auto"
        ):
            raise ValueError("unexpected convergence/arithmetic configuration")
        for carbon in sorted({row["carbon"] for row in record["samples"]}):
            rows = [row for row in record["samples"] if row["carbon"] == carbon]
            baseline = next(row for row in rows if row["rank"] == 0)
            protocol = record["protocols"][str(carbon)]
            for row in rows:
                observations += 1
                if not row["converged"] or row["executed_backend"] != "cuda":
                    raise ValueError("incomplete/non-CUDA endpoint")
                if (
                    row["convergence"] != baseline["convergence"]
                    or row["convergence"]
                    != qualification["current_standard_controls"]
                ):
                    raise ValueError("candidate changed convergence/precision controls")
                energy_error = abs(row["energy"] - baseline["energy"])
                force_error = float(
                    np.max(np.abs(np.asarray(row["forces"]) - baseline["forces"]))
                )
                if (
                    not np.isfinite([energy_error, force_error]).all()
                    or energy_error > 1e-8
                    or force_error > 1e-8
                ):
                    raise ValueError("full-vector baseline observable parity failed")
                energy_errors.append(energy_error)
                force_errors.append(force_error)
                reference_components = {
                    (c["method"], c["basis_identity"]): c["result"]
                    for c in baseline["components"]
                }
                for component in row["components"]:
                    control = reference_components[
                        (component["method"], component["basis_identity"])
                    ]
                    value = component["result"]
                    energy = abs(value["energy"] - control["energy"])
                    force = float(
                        np.max(np.abs(np.asarray(value["forces"]) - control["forces"]))
                    )
                    if (
                        not np.isfinite([energy, force]).all()
                        or energy > 1e-8
                        or force > 1e-8
                    ):
                        raise ValueError("component full-vector baseline parity failed")
                    energy_errors.append(energy)
                    force_errors.append(force)
                probe = row["probe"]
                if row["rank"]:
                    if (
                        probe["accepted_rank"] != row["rank"]
                        or not probe["applications"]
                    ):
                        raise ValueError(
                            "candidate did not exercise the requested inverse"
                        )
                    if (
                        probe["estimated_probe_peak_bytes"]
                        > probe["available_numeric_bytes"]
                    ):
                        raise ValueError(
                            "candidate exceeded the admitted numeric estimate"
                        )
                    norm = probe["independent_residual_norm"]
                    limit = row["convergence"]["limits"]["lambda_residual_tolerance"]
                    solve_limit = row["convergence"]["limits"][
                        "response_absolute_tolerance"
                    ]
                    if (
                        not np.isfinite(
                            [
                                norm,
                                probe["independent_residual_max"],
                                probe["true_residual_norm"],
                            ]
                        ).all()
                        or max(norm, probe["independent_residual_max"]) > limit
                        or probe["true_residual_norm"] > solve_limit
                    ):
                        raise ValueError("unchanged physical Lambda gate failed")
                    lambda_errors.append(norm)
                if carbon != 2:
                    continue
                # This reuses only a numerical oracle with identical inputs;
                # none of the historical native timings enter the comparison.
                for key in (
                    "atoms_bohr",
                    "orbital_identities",
                    "auxiliary_identities",
                    "hf_reference",
                    "dataset_sha256",
                ):
                    if protocol[key] != reference["protocol"][key]:
                        raise ValueError(
                            f"independent reference identity mismatch: {key}"
                        )
                values = {"composite": row}
                for component in row["components"]:
                    level = (
                        "low"
                        if component["basis_identity"]
                        == protocol["orbital_identities"][0]
                        else "high"
                    )
                    values[level + "/" + component["method"]] = component["result"]
                for name, value in values.items():
                    energy = abs(value["energy"] - reference["energies"][name])
                    derivative = -float(
                        np.sum(np.asarray(value["forces"]) * reference["direction"])
                    )
                    force = max(
                        abs(derivative - entry["values"][name])
                        for entry in reference["derivatives"]
                    )
                    if (
                        not np.isfinite([energy, force]).all()
                        or energy > 1e-8
                        or force > 3e-6
                    ):
                        raise ValueError(f"independent {name} observable gate failed")
                    independent_errors.append((energy, force))
            for rank in sorted({row["rank"] for row in rows}):
                selected = [row for row in rows if row["rank"] == rank]
                performance = [
                    next(
                        component["result"]["cc_performance"]
                        for component in row["components"]
                        if component["method"] == "df-ccsd(t)"
                    )
                    for row in selected
                ]
                populations.append(
                    {
                        "record": binding["path"],
                        "carbon": carbon,
                        "rank": rank,
                        "observations": len(selected),
                        "complete_seconds": [
                            row["complete_seconds"] for row in selected
                        ],
                        "median_complete_seconds": statistics.median(
                            row["complete_seconds"] for row in selected
                        ),
                        "lambda_seconds": [
                            item["lambda_seconds"] for item in performance
                        ],
                        "fp32_actions": [
                            item["lambda_fp32_actions"] for item in performance
                        ],
                        "fp64_actions": [
                            item["lambda_fp64_actions"] for item in performance
                        ],
                        "q_batch": [
                            item["lambda_auxiliary_batch_size"] for item in performance
                        ],
                        "callback_seconds": [
                            row["probe"]["callback_seconds"] for row in selected
                        ],
                    }
                )
    if observations != qualification["completed_observations"]:
        raise ValueError("incomplete retained population")
    return {
        "schema": "generativeqc.multisecant-native-verification.v1",
        "completed_observations": observations,
        "maximum_baseline_energy_change": max(energy_errors),
        "maximum_baseline_force_change": max(force_errors),
        "maximum_independent_lambda_residual": max(lambda_errors),
        "maximum_independent_reference_energy_error": max(
            error[0] for error in independent_errors
        ),
        "maximum_independent_reference_directional_force_error": max(
            error[1] for error in independent_errors
        ),
        "populations": populations,
        "default_promotion": False,
    }


def main() -> None:
    """Write an optional machine-readable replay without updating frozen data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-source", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.reconstruct_source:
        reconstruct_sources(read_json(DIRECTORY / "source-identities.json"))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
