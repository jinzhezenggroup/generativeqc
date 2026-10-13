"""Lossless storage and reader contracts for retained work/endpoint records."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]


def verify(directory: Path, entry: dict) -> object:
    """Stored and decoded identities protect every historical scientific byte."""
    path = directory / entry["path"]
    stored = path.read_bytes()
    assert len(stored) == entry["bytes"]
    assert hashlib.sha256(stored).hexdigest() == entry["sha256"]
    decoded = gzip.decompress(stored)
    assert len(decoded) == entry["decoded_bytes"]
    assert hashlib.sha256(decoded).hexdigest() == entry["decoded_sha256"]
    result = load_json(path)
    assert result == json.loads(decoded)
    return result


def test_warm_df_work_record_retains_complete_workloads() -> None:
    directory = ROOT / "benchmarks/results/df-one-step-warm-20260926"
    summary = load_json(directory / "summary.json")
    entry = next(part for part in summary["parts"] if part["path"] == "work.json.gz")
    record = verify(directory, entry)
    assert len(record) == 8
    assert all(row["diagnostics"] for row in record)


def test_one_electron_records_retain_source_and_all_samples() -> None:
    directory = ROOT / "benchmarks/results/one-electron-values-rtx5090"
    storage = load_json(directory / "storage.json")
    assert len(storage["files"]) == 2
    for entry in storage["files"]:
        record = verify(directory, entry)
        assert record["samples"]
        assert record["source_identity"]
        assert record["binary_hash"]
        assert record["slurm_job_id"]


def test_one_electron_ownership_evidence_resolves_through_shared_reader() -> None:
    owner = load_json(ROOT / "docs/cuda_ownership/subsystems/one_electron_values.json")
    paths = [path for path in owner["evidence"] if path.endswith(".json.gz")]
    assert paths
    for path in paths:
        assert load_json(ROOT / path)["samples"]
