"""Retain complete Fast-anchor scientific records under bounded storage."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("family", "name", "count"),
    [
        ("fast-anchor-default-20261018", "qualification", 6),
        ("wb97mv-anchor-delta-20261018", "evidence", 8),
    ],
)
def test_complete_original_records_round_trip(
    family: str, name: str, count: int
) -> None:
    directory = ROOT / "benchmarks/results" / family
    storage = json.loads((directory / f"{name}.storage.json").read_text())
    stored = ROOT / storage["stored_path"]
    encoded = stored.read_bytes()
    raw = lzma.decompress(encoded)
    assert len(encoded) == storage["stored_bytes"]
    assert hashlib.sha256(encoded).hexdigest() == storage["stored_sha256"]
    assert len(raw) == storage["original_bytes"]
    assert hashlib.sha256(raw).hexdigest() == storage["original_sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == storage["original_git_blob"]
    )
    record = load_json(stored)
    assert record == json.loads(raw)
    assert len(record["samples"]) == count
    assert all(sample["gate"] for sample in record["samples"])
    small_fast = next(
        pair
        for pair in record["pairs"]
        if pair["atoms"] == 12 and pair["policy"] == "fast"
    )
    assert small_fast["elapsed_reduction_percent"] < 0
    assert all(sample["independent_gate"]["gate"] for sample in record["samples"])
    assert all(sample["receipt_sha256"] for sample in record["samples"])
    assert all(
        sample["source_authentication"]["native_identity"]
        for sample in record["samples"]
    )
