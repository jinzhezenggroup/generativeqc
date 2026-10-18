"""Historical evidence compaction must preserve the complete original bytes."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_record


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
    ("aos", "variant", "original_bytes", "expected_sha256"),
    [
        (
            192,
            "blas",
            166345,
            "502d55783e90f97419d96ab93491873653a74594fa94201a262cc7bbeec474a2",
        ),
        (
            384,
            "blas",
            267196,
            "4afc40f038cc07ff8fc22a7ef52d73ea948ffb84bc0329a9666b052d33609e37",
        ),
        (
            192,
            "serial",
            166398,
            "61b865a38908cbcfcc224f8bfb7870fc8b2296c746a489deed6c9d08bed0a59e",
        ),
    ],
)
def test_metric_gemv_force_record_restores_its_hash_bound_original(
    aos: int, variant: str, original_bytes: int, expected_sha256: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads(
        (
            root
            / f"benchmarks/results/issue206-metric-gemv/measurements/{aos}ao-b4-forces-{variant}.storage.json"
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
    assert load_record(root / metadata["stored_path"]) == value
    assert value["workload"]["ao_count"] == aos
    assert value["workload"]["batch_size"] == 4


def test_metric_gemv_energy_record_restores_its_hash_bound_original() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/results/issue206-metric-gemv/measurements"
    metadata = json.loads((directory / "384ao-b4-energy-blas.storage.json").read_text())
    stored_path = directory / "384ao-b4-energy-blas.json.gz"
    original_path = directory / "384ao-b4-energy-blas.json"
    assert metadata["stored_path"] == stored_path.relative_to(root).as_posix()
    assert metadata["original_path"] == original_path.relative_to(root).as_posix()
    stored = stored_path.read_bytes()
    original = gzip.decompress(stored)
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert len(original) == metadata["original_bytes"] == 99289
    expected_sha256 = "84dd083b991d8539fa2f8b0ff8dfa6bb89c5cf32e55f9869d594f28a0cbba9ff"
    assert hashlib.sha256(original).hexdigest() == expected_sha256
    assert metadata["original_sha256"] == expected_sha256
    assert metadata["original_git_blob"] == "85eab5969287b529d7526a58fcb4170c31846561"
    value = json.loads(original)
    assert load_record(stored_path) == value
    assert value["workload"]["ao_count"] == 384
    assert value["workload"]["batch_size"] == 4
    assert value["workload"]["properties"] == ["energy"]


def test_metric_gemv_direct_gate_restores_its_hash_bound_original() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/results/issue206-metric-gemv/direct-gate"
    name = "water-octamer-s4-def2-svp-spherical-b4-none"
    metadata = json.loads((directory / f"{name}.storage.json").read_text())
    stored_path = directory / f"{name}.json.gz"
    original_path = directory / f"{name}.json"
    assert metadata["stored_path"] == stored_path.relative_to(root).as_posix()
    assert metadata["original_path"] == original_path.relative_to(root).as_posix()
    stored = stored_path.read_bytes()
    original = gzip.decompress(stored)
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert len(original) == metadata["original_bytes"] == 142797
    expected_sha256 = "257b3a976fc8f73675bd0cb25c6be79913b25c85fb0abf3dd5ccf73ed53fd134"
    assert hashlib.sha256(original).hexdigest() == expected_sha256
    assert metadata["original_sha256"] == expected_sha256
    expected_blob = "5bdd0708d3948dd40cd985c88965859d47e76db7"
    assert metadata["original_git_blob"] == expected_blob
    assert hashlib.sha1(b"blob 142797\0" + original).hexdigest() == expected_blob
    value = json.loads(original)
    assert load_record(stored_path) == value
    assert value["workload"]["ao_count"] == 192
    assert value["workload"]["batch_size"] == 4
    assert value["workload"]["density_fitting"] == "none"
    assert (
        value["accuracy"]["gate_selection"]["selection"]
        == "final_pair_unmatched_labeled"
    )
    pairs = value["accuracy"]["paired_warm_repeats"]
    assert len(pairs) == 5
    assert all(pair["iteration_branches_match"] is False for pair in pairs)
    assert value["timing_summary"]["iteration_matched"] is None
    assert value["timing_summary"]["speed_claim_uses"] == "unmatched_labeled"
    assert all(
        len(value[engine]["warm_samples"]) == 5 for engine in ("vibeqc", "gpu4pyscf")
    )
