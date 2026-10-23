"""Additional historical receipts retain their exact authenticated bytes."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_record

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "expected_sha"),
    [
        (
            "direct-gate/water-tetramer-def2-svp-spherical-b4-none.json",
            "fe44a583a5baa93b8a980e27a35cfc8bc1e60f26ce403e747148334d7ca74930",
        ),
        (
            "historical-prechange-direct/water-tetramer-def2-svp-spherical-b4-none.json",
            "cebcb412587854ca16114770de5f22b59c7ef16484f40c2988d98dd7a846c650",
        ),
        (
            "measurements/192ao-b4-energy-blas.json",
            "4f8b7f0893ca2e9cac6b079b50893a9216a8c9442553d459823b790003e97f13",
        ),
        (
            "measurements/96ao-b4-energy-blas.json",
            "3db3cab6e5a21e6c9497c62ac30941bd20454074306ea855b5958d258fa51cdb",
        ),
        (
            "measurements/19ao-b4-forces-blas.json",
            "b96627e6e8d86b2865797ecd61f91246fc664535e23f10a751fc78c071aa8130",
        ),
    ],
)
def test_additional_receipt_retains_exact_bytes(name: str, expected_sha: str) -> None:
    logical = ROOT / "benchmarks/results/issue206-metric-gemv" / name
    metadata = json.loads(logical.with_suffix(".storage.json").read_text())
    stored = (ROOT / metadata["stored_path"]).read_bytes()
    raw = gzip.decompress(stored)
    assert metadata["original_path"] == logical.relative_to(ROOT).as_posix()
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert stored[4:8] == bytes(4)
    assert len(raw) == metadata["original_bytes"]
    assert (
        hashlib.sha256(raw).hexdigest() == metadata["original_sha256"] == expected_sha
    )
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == metadata["original_git_blob"]
    )
    assert load_record(ROOT / metadata["stored_path"]) == json.loads(raw)
