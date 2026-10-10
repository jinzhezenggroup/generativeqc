"""Lossless storage keeps every historical final-occupied endpoint observation."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import decode_record, load_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-final-occupied-endpoint-20260926"
ORIGINAL_SHA256 = "709b72edc673227844ed7ad0a38fe8c572e1d0ebbe39dadf36005fbf95e5d88a"
STORED_SHA256 = "3a61f3392852c1e90c11af1648615b1c0dcd95f8e42f52f5e22b090a7271b1f6"


def test_final_occupied_receipt_preserves_exact_original_bytes() -> None:
    packed = (BUNDLE / "receipt.json.gz").read_bytes()
    assert len(packed) == 16_870
    assert hashlib.sha256(packed).hexdigest() == STORED_SHA256
    assert packed[4:8] == b"\0" * 4  # No wall-clock timestamp in storage.
    original = gzip.decompress(packed)
    assert len(original) == 324_689
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_SHA256
    assert not (BUNDLE / "receipt.json").exists()


def test_final_occupied_receipt_uses_shared_record_readers() -> None:
    path = BUNDLE / "receipt.json.gz"
    packed = path.read_bytes()
    original = gzip.decompress(packed)
    expected = json.loads(original)
    assert decode_record(packed, {}, path=path.name) == expected
    assert decode_record(original, {}, path="receipt.json") == expected
    assert load_record(path) == expected
