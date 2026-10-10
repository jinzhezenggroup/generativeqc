"""Historical DIIS compaction retains every byte of the accepted summary."""

import gzip
import hashlib
import json
from pathlib import Path


def test_device_diis_summary_restores_its_hash_bound_original() -> None:
    root = Path(__file__).resolve().parents[2] / "benchmarks/results/issue308-device-diis"
    manifest = json.loads((root / "summary-retention.json").read_text())
    compressed = (root / manifest["archive"]).read_bytes()
    original = gzip.decompress(compressed)
    assert len(compressed) == manifest["archive_bytes"]
    assert hashlib.sha256(compressed).hexdigest() == manifest["archive_sha256"]
    assert len(original) == manifest["original_bytes"]
    assert hashlib.sha256(original).hexdigest() == manifest["original_sha256"]
    assert json.loads(original)["schema"] == "vibeqc.issue308.device-diis.v1"
