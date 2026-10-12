"""Historical evidence compaction must preserve the complete original bytes."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest


def test_metric_gemv_summary_restores_its_hash_bound_original() -> None:
    root = (
        Path(__file__).resolve().parents[2] / "benchmarks/results/issue206-metric-gemv"
    )
    manifest = json.loads((root / "summary-retention.json").read_text())
    compressed = (root / manifest["archive"]).read_bytes()
    original = gzip.decompress(compressed)
    assert len(compressed) == manifest["archive_bytes"]
    assert hashlib.sha256(compressed).hexdigest() == manifest["archive_sha256"]
    assert len(original) == manifest["original_bytes"]
    assert hashlib.sha256(original).hexdigest() == manifest["original_sha256"]
    assert json.loads(original)["schema"] == "vibeqc.issue206.metric-gemv.v1"


@pytest.mark.parametrize(
    ("aos", "original_bytes", "expected_sha256"),
    [
        (
            192,
            166345,
            "502d55783e90f97419d96ab93491873653a74594fa94201a262cc7bbeec474a2",
        ),
        (
            384,
            267196,
            "4afc40f038cc07ff8fc22a7ef52d73ea948ffb84bc0329a9666b052d33609e37",
        ),
    ],
)
def test_metric_gemv_force_record_restores_its_hash_bound_original(
    aos: int, original_bytes: int, expected_sha256: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads(
        (
            root
            / f"benchmarks/results/issue206-metric-gemv/measurements/{aos}ao-b4-forces-blas.storage.json"
        ).read_text()
    )
    stored = (root / metadata["stored_path"]).read_bytes()
    original = gzip.decompress(stored)
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert len(original) == metadata["original_bytes"] == original_bytes
    assert hashlib.sha256(original).hexdigest() == metadata["original_sha256"]
    assert metadata["original_sha256"] == expected_sha256
    value = json.loads(original)
    assert value["workload"]["ao_count"] == aos
    assert value["workload"]["batch_size"] == 4
