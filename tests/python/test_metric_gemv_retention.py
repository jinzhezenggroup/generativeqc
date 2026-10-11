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


@pytest.mark.parametrize(
    ("aos", "original_bytes", "expected_sha256"),
    [
        (
            19,
            66103,
            "d1bb61f21f7b79346851bbd0c3d3173bd2bf8d6762c0d65fb7134061ca2a92f2",
        ),
        (
            384,
            99289,
            "84dd083b991d8539fa2f8b0ff8dfa6bb89c5cf32e55f9869d594f28a0cbba9ff",
        ),
    ],
)
def test_metric_gemv_energy_record_restores_its_hash_bound_original(
    aos: int, original_bytes: int, expected_sha256: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/results/issue206-metric-gemv/measurements"
    metadata = json.loads(
        (directory / f"{aos}ao-b4-energy-blas.storage.json").read_text()
    )
    assert metadata["original_revision"] == ("e7abb59975bd3c0a9efdd8f82eeb1b74854a4f31")
    stored = (root / metadata["stored_path"]).read_bytes()
    original = gzip.decompress(stored)
    assert not (root / metadata["original_path"]).exists()
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert len(original) == metadata["original_bytes"] == original_bytes
    assert hashlib.sha256(original).hexdigest() == metadata["original_sha256"]
    assert metadata["original_sha256"] == expected_sha256
    header = f"blob {len(original)}\0".encode()
    blob = hashlib.sha1(header + original, usedforsecurity=False).hexdigest()
    assert blob == metadata["original_git_blob"]
    record = json.loads(original)
    assert record["workload"]["ao_count"] == aos
    assert record["workload"]["batch_size"] == 4
    assert record["workload"]["properties"] == ["energy"]
