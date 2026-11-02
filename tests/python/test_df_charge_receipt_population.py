"""Retained qualification must reject rehashed duplicate populations."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from benchmarks import _exact_fp64_evidence as evidence

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "benchmarks/results/df-charge-partitions-20261024/reproduce.py"


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
    bundle, files = evidence.load(VERIFIER.parent)
    packet = json.loads(files["original-receipts.json"])
    keys = [
        key for key in packet["records"] if key.startswith("stage25-" + campaign + "/")
    ]
    packet["records"][keys[1]] = packet["records"][keys[0]]
    packet["raw_record_sha256"][keys[1]] = hashlib.sha256(
        packet["records"][keys[1]].encode()
    ).hexdigest()
    # The codec's integrity is tested independently. Inject a fully decoded,
    # rehashed packet here to exercise scientific population validation itself.
    files["original-receipts.json"] = json.dumps(packet)
    monkeypatch.setattr(evidence, "load", lambda path: (bundle, files))
    with pytest.raises(AssertionError, match="endpoint population"):
        runpy.run_path(str(VERIFIER))
