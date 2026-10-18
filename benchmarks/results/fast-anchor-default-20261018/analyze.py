"""Reduce retained production endpoint receipts without discarding work or gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    """Fingerprint raw evidence rather than relying on its local pathname."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduce_sample(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain endpoint scope, semantic work, precision and independent acceptance."""
    record = json.loads(path.read_text())
    precision = record["precision"]
    diagnostic = record["ks_diagnostic"]
    before = (path.parent / "library-before.sha256").read_text().split()[0]
    after = (path.parent / "library-after.sha256").read_text().split()[0]
    exit_code = int((path.parent / "exit-code.txt").read_text())
    sample = {
        name: record.get(name)
        for name in (
            "atoms",
            "policy",
            "delta",
            "slurm_job_id",
            "energy",
            "iterations",
            "fock_builds",
            "construction_seconds",
            "prepare_seconds",
            "execute_seconds",
            "export_seconds",
            "teardown_seconds",
            "cold_endpoint_seconds",
            "initial_guess",
            "density_rms",
            "physical_residual_rms",
            "energy_change",
            "warm_start_used",
            "warm_start_fallback",
            "candidate_controls",
            "independent_gate",
            "error",
        )
    }
    sample.update(
        receipt=path.parent.name,
        receipt_sha256=digest(path),
        source_authentication=json.loads(
            (path.parent / "source-authentication.json").read_text()
        ),
        library_sha256=before,
        library_unchanged=before == after,
        exit_code=exit_code,
        nonlocal_work=diagnostic["nonlocal_work"],
        coulomb_exchange_work=diagnostic["coulomb_exchange_work"],
        nonlocal_history_scopes=diagnostic["nonlocal_history_scopes"],
        operator_inventory=precision["operators"],
    )
    for name in (
        "mixed_stage_fock_builds",
        "strict_stage_fock_builds",
        "post_scf_fock_builds",
        "execution_retries",
        "fallback_count",
        "final_residual_audits",
    ):
        sample[name] = precision[name]
    sample["gate"] = bool(
        record.get("succeeded")
        and record.get("converged")
        and record.get("finite_forces")
        and not record.get("warm_start_used")
        and not record.get("error")
        and record["independent_gate"]["gate"]
        and before == after
        and exit_code == 0
        and precision["complete"]
        and precision["operator_inventory_complete"]
    )
    return sample, record


def compare_pair(
    off: tuple[dict[str, Any], dict[str, Any]],
    default: tuple[dict[str, Any], dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    """Compare identical-controls cold endpoints and their independently checked states."""
    off_sample, off_record = off
    on_sample, on_record = default
    forces = np.asarray(on_record["forces"]) - np.asarray(off_record["forces"])
    densities = [
        np.load(root / sample["receipt"] / Path(record["final_density"]["path"]).name)
        for sample, record in (off, default)
    ]
    density_delta = densities[1] - densities[0]
    off_seconds, on_seconds = (
        off_sample["cold_endpoint_seconds"],
        on_sample["cold_endpoint_seconds"],
    )
    return {
        "atoms": off_record["atoms"],
        "policy": off_record["policy"],
        "off_receipt": off_sample["receipt"],
        "default_receipt": on_sample["receipt"],
        "off_seconds": off_seconds,
        "default_seconds": on_seconds,
        "speedup_off_over_default": off_seconds / on_seconds,
        "elapsed_reduction_percent": 100 * (1 - on_seconds / off_seconds),
        "energy_abs_difference_hartree": abs(
            on_record["energy"] - off_record["energy"]
        ),
        "force_max_difference_hartree_per_bohr": float(np.max(np.abs(forces))),
        "density_max_difference": float(np.max(np.abs(density_delta))),
        "same_controls": off_record["candidate_controls"]
        == on_record["candidate_controls"],
        "same_geometry": off_record["atoms_bohr"] == on_record["atoms_bohr"],
        "same_basis": off_record["basis_identity"] == on_record["basis_identity"],
        "both_independently_qualified": off_sample["gate"] and on_sample["gate"],
    }


def main() -> None:
    """Write compact public evidence; retain full arrays and raw receipts locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--build-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reduced = [reduce_sample(path) for path in sorted(args.root.glob("*/result.json"))]
    pairs = []
    for sample, record in reduced:
        if record["delta"] != "0":
            continue
        counterpart = next(
            item
            for item in reduced
            if item[1]["delta"] == "auto"
            and item[1]["atoms"] == record["atoms"]
            and item[1]["policy"] == record["policy"]
        )
        pairs.append(compare_pair((sample, record), counterpart, args.root))
    build = args.build_receipts
    summary = {
        "schema": "generativeqc.fast-anchor-default-endpoints.v1",
        "base_revision": "5d1dbbcaf2f04fb83980c8f988795df28b41e392",
        "warning": "One cold observation per arm, not repeated statistics or a universal speed claim; all first-use costs retained.",
        "build": json.loads((build / "authentication.json").read_text()),
        "rebuilt_inputs": json.loads((build / "rebuilt-inputs.json").read_text()),
        "compiler_cache_version": (build / "cache-version.txt").read_text().strip(),
        "compiler_cache_before": (build / "cache-before.txt").read_text(),
        "compiler_cache_after": (build / "cache-after.txt").read_text(),
        "samples": [sample for sample, _ in reduced],
        "pairs": pairs,
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for pair in pairs:
        print(json.dumps(pair))
    if not reduced or not all(sample["gate"] for sample, _ in reduced):
        raise SystemExit("At least one retained endpoint failed qualification")
    if not pairs or not all(
        pair[key]
        for pair in pairs
        for key in (
            "same_controls",
            "same_geometry",
            "same_basis",
            "both_independently_qualified",
        )
    ):
        raise SystemExit(
            "Paired endpoints do not share the qualified scientific contract"
        )


if __name__ == "__main__":
    main()
