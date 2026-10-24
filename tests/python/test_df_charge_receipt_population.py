"""Retained qualification must reject rehashed duplicate populations."""

import hashlib
import json
import lzma
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "benchmarks/results/df-charge-partitions-20261024/reproduce.py"
PACKET = VERIFIER.parent / "endpoints.json.xz"


def test_complete_population() -> None:
    runpy.run_path(str(VERIFIER))


@pytest.mark.parametrize(
    "campaign",
    [
        "screen-df",
        "screen-direct",
        "gemm-matrix-df",
        "gemm-matrix-direct",
        "moved-df",
        "moved-direct",
        "larger",
        "cold-nsys",
    ],
)
def test_rehashed_duplicate_population_is_rejected(
    monkeypatch: pytest.MonkeyPatch, campaign: str
) -> None:
    packet = json.loads(lzma.decompress(PACKET.read_bytes()))
    keys = [
        key for key in packet["records"] if key.startswith("stage25-" + campaign + "/")
    ]
    packet["records"][keys[1]] = packet["records"][keys[0]]
    packet["raw_record_sha256"][keys[1]] = hashlib.sha256(
        packet["records"][keys[1]].encode()
    ).hexdigest()
    altered = lzma.compress(json.dumps(packet).encode())
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: altered if path == PACKET else original(path)
    )
    with pytest.raises(AssertionError, match="endpoint population"):
        runpy.run_path(str(VERIFIER))
