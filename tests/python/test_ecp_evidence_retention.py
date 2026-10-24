"""Keep historical heavy-element endpoint evidence exactly recoverable."""

import gzip
import hashlib
import json
import lzma
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_record


@pytest.mark.parametrize(
    "name",
    [
        "ecp-policy-171/baseline-source-identity",
        "ecp-policy-171/candidate-source-identity",
        "ecp-policy-171/compatibility/candidate-source-identity",
        "ecp-real-f-171/source-identity",
        "ecp-halogen-171/source-identity",
        "ecp-halogen-171/refresh/source-identity",
    ],
)
def test_compressed_source_inventory_preserves_original_bytes(name: str) -> None:
    """Preserve pinned historical source inputs without requiring Git history."""
    root = Path(__file__).resolve().parents[2]
    prefix = root / "benchmarks/results" / name
    metadata = json.loads(prefix.with_suffix(".storage.json").read_bytes())
    stored = (root / metadata["stored_path"]).read_bytes()
    bundle = json.loads(lzma.decompress(stored))
    original = bundle["files"][metadata["member"]].encode("utf-8")
    manifest = json.loads((prefix.parent / "raw-evidence-manifest.json").read_bytes())
    assert len(original) == metadata["original_bytes"]
    assert hashlib.sha256(original).hexdigest() == metadata["original_sha256"]
    assert (
        hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
        == metadata["original_git_blob"]
    )
    if metadata["historical_manifest_entry"] is not None:
        entry = next(
            row
            for row in manifest["files"]
            if row["path"] == metadata["historical_manifest_entry"]
        )
        assert len(original) == entry["bytes"]
        assert hashlib.sha256(original).hexdigest() == entry["sha256"]
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert (
        load_record(root / metadata["stored_path"])["files"][metadata["member"]].encode(
            "utf-8"
        )
        == original
    )


def test_heavy_ecp_cpu_record_preserves_original_bytes_and_reader() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/results/ecp-heavy-171"
    metadata = json.loads((directory / "cpu-endpoints.storage.json").read_bytes())
    stored = (root / metadata["stored_path"]).read_bytes()
    original = gzip.decompress(stored)
    historical = json.loads((directory / "raw-evidence-manifest.json").read_bytes())
    entry = next(x for x in historical["files"] if x["path"] == "cpu-endpoints.json")

    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert stored[4:8] == bytes(4)  # Deterministic zero-mtime gzip header.
    assert len(original) == metadata["original_bytes"] == entry["bytes"] == 12047
    assert (
        hashlib.sha256(original).hexdigest()
        == metadata["original_sha256"]
        == entry["sha256"]
        == "f489eeb8f7fc3da6f3f612da639df9ea24d07cc06c739cc34e33459059455893"
    )
    git_object = f"blob {len(original)}\0".encode() + original
    assert (
        hashlib.sha1(git_object).hexdigest()
        == metadata["original_git_blob"]
        == "cce846a8b983a3806b56774d495c7faed03eff22"
    )
    restored = load_record(root / metadata["stored_path"])
    assert restored == json.loads(original)
    assert restored["device"] == "cpu"
    assert len(restored["cases"]) == 4
