"""Keep historical heavy-element endpoint evidence exactly recoverable."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import load_record


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
