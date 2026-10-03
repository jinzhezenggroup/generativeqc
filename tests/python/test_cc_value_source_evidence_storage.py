"""Preserve every original CC source receipt byte through bounded compression."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/cc-value-source-20261003"
EXPECTED = {
    "summary.json": "0a45bc1242705e9ca6d0faf8a6c0d4fd7ecc3c3f",
    "large-energy.json": "93d9208e73c6d975494fe96561ce49a710888024",
}


def test_cc_value_source_receipts_preserve_original_git_bytes() -> None:
    manifest = json.loads((FAMILY / "storage.json").read_text())
    assert manifest["source_revision"] == "cacc6cf0f410bff187c6d064118e954614c99193"
    assert len(manifest["records"]) == len(EXPECTED)
    for row in manifest["records"]:
        name = row["original_path"]
        assert row["stored_path"] == name + ".gz"
        assert not (FAMILY / name).exists()
        stored = (FAMILY / row["stored_path"]).read_bytes()
        assert len(stored) == row["stored_bytes"]
        assert hashlib.sha256(stored).hexdigest() == row["stored_sha256"]
        raw = gzip.decompress(stored)
        assert len(raw) == row["original_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["original_sha256"]
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert blob == EXPECTED[name] == row["original_git_blob"]
        assert load_json(FAMILY / row["stored_path"]) == json.loads(raw)
