"""Recheck independent full-vector gates and summarize retained E+F samples.

Run from this checkout with local Git history and PYTHONPATH=python:.
No GPU or network is needed.
Diagnostic samples are checked for accuracy but never enter timing medians.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import os
import subprocess
from pathlib import Path

from generativeqc_compiler.common.performance import assess_comparison

from benchmarks.readme_omol25 import check_record


def read_git_blob(root: Path, revision: str, path: str) -> bytes:
    """Read an existing local object without lazy fetch or interactive login."""
    environment = {
        **os.environ,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_ALLOW_PROTOCOL": "",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{path}"],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "historical object unavailable; fetch the recorded revision "
            f"with git fetch origin {revision} and retry"
        )
    return result.stdout


def summarize(directory: Path, atoms: int) -> dict:
    """Keep all samples, executed work, noise gates and source provenance."""
    root = directory.resolve().parents[2]
    path = directory / f"endpoints-{atoms}.json.xz"
    compact = json.loads(lzma.decompress(path.read_bytes()))
    recovery = compact.pop("raw_recovery")
    stored = read_git_blob(root, recovery["revision"], recovery["path"])
    raw = gzip.decompress(stored)
    assert hashlib.sha256(raw).hexdigest() == recovery["uncompressed_sha256"]
    data = json.loads(raw)
    reduced = json.loads(raw)
    for sample in reduced["samples"]:
        sample.pop("forces")
        sample["ks_diagnostic"].pop("history", None)
    assert compact == reduced
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
        measured = read_git_blob(
            root, "ce01eaf5a760f22cc59e58315319b61c6798dae2", source
        )
        assert hashlib.sha256(measured).hexdigest() == digest, source
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
