"""Lossless historical evidence remains independently inspectable after storage changes."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "benchmarks/results/default-huckel-df-20261012/storage.json"


@pytest.mark.parametrize(
    "entry", json.loads(INVENTORY.read_text())["files"], ids=lambda row: row["path"]
)
def test_retained_evidence_encoded_and_decoded_identity(entry: dict) -> None:
    """Bind archived payloads, never current production source or function bodies."""
    encoded = (ROOT / entry["path"]).read_bytes()
    assert len(encoded) == entry["bytes"]
    assert hashlib.sha256(encoded).hexdigest() == entry["sha256"]
    decoded = gzip.decompress(encoded)
    assert len(decoded) == entry["decoded_bytes"]
    assert hashlib.sha256(decoded).hexdigest() == entry["decoded_sha256"]
