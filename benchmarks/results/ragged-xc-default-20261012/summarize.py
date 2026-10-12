"""Recheck independent full-vector gates and summarize retained E+F samples.

Run from the measured checkout with PYTHONPATH=python:. No GPU is needed.
Diagnostic samples are checked for accuracy but never enter timing medians.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from generativeqc_compiler.common.performance import assess_comparison

from benchmarks.readme_omol25 import check_record


def summarize(directory: Path, atoms: int) -> dict:
    """Keep all samples, executed work, noise gates and source provenance."""
    root = directory.resolve().parents[2]
    path = directory / f"ragged-default-{atoms}.json.gz"
    data = json.loads(gzip.decompress(path.read_bytes()))
    reference_path = (
        root
        / "benchmarks/results/pbe0-xc-tiles-20261006"
        / f"reference-{atoms}.json.gz"
    )
    reference_bytes = reference_path.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == data["reference_sha256"]
    reference = json.loads(gzip.decompress(reference_bytes))
    assert data["stage"] == "complete" and not data["feasibility"]
    assert reference["protocol"] == data["protocol"]
    assert len(data["samples"]) == 48
    for source, digest in data["source_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == digest, (
            source
        )
    validations = []
    for sample in data["samples"]:
        geometry = 0 if sample["phase"] in ("cold", "warm") else 1
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
        )
        validation = check_record(sample, oracle)
        assert validation["gate"] and sample["gate"]
        assert sample["status"] == 0 and sample["converged"]
        validations.append(validation)
    clean = [row for row in data["samples"] if not row["diagnostic"]]
    performance = assess_comparison(
        [
            {
                "workload": row["phase"],
                "selection": "baseline" if row["arm"] == "exact" else "candidate",
                "seconds": row["endpoint_seconds"],
                "inputs_hash": data["reference_sha256"],
                "synchronized": True,
            }
            for row in clean
        ]
    )
    work = [
        {
            key: row[key]
            for key in (
                "arm",
                "repeat",
                "phase",
                "iterations",
                "fock_builds",
                "warm_start_used",
                "warm_start_fallback",
            )
        }
        for row in clean
    ]
    diagnostics = []
    for row in data["samples"]:
        if not row["diagnostic"]:
            continue
        diagnostics.append(
            {
                "arm": row["arm"],
                "phase": row["phase"],
                "batch_receipts": row["batch_receipts"],
                "operators": row["component_profile"]["groups"],
            }
        )
    return {
        "atoms": atoms,
        "aos": data["protocol"]["aos"],
        "slurm_job_id": data["slurm_job_id"],
        "native_binary_sha256": data["native_binary_sha256"],
        "independent_records": len(validations),
        "energy_error_max": max(row["energy_error"] for row in validations),
        "force_component_error_max": max(row["force_error"] for row in validations),
        "performance": performance,
        "clean_work": work,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    result = {str(atoms): summarize(directory, atoms) for atoms in (96, 48)}
    print(json.dumps(result, indent=2))
