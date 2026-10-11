"""Resident receipts preserve exact recorded bytes, independently of encoding."""

import hashlib
import json
import lzma
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "benchmarks/results/response-contract/2026-10-11"


@pytest.mark.parametrize(
    ("name", "original_bytes", "sha256", "source", "tree"),
    [
        (
            "lambda",
            13857,
            "1512ba57906183327477f328ef9f92d18a68781b31c284b30923693961fc1c53",
            "8e8cf9aab32c22d876aaef09c0c97ff02f66a3c9",
            "7f7c0c883bd4f00b23092a71d1fc68ab3ea4ef2e",
        ),
        (
            "integration",
            11428,
            "f9ce6f73769ab7017f543172215407c560940e202e5655685f1a4cbbfe7d307b",
            "b9617de02ab3664fc24532b742c88ac647e4fb0b",
            "a08b2070fbc0a3e7a6bf432a5216de810bcbf816",
        ),
    ],
)
def test_resident_receipt_preserves_original_bytes_and_source(
    name: str, original_bytes: int, sha256: str, source: str, tree: str
) -> None:
    path = RECEIPTS / f"qualification-resident-{name}.json.xz"
    original = lzma.decompress(path.read_bytes())
    assert len(original) == original_bytes
    assert hashlib.sha256(original).hexdigest() == sha256
    payload = json.loads(original)
    assert payload["source"]["commit"] == source
    assert payload["source"]["tree"] == tree
    assert "not default promotion" in payload["scope"] or (
        payload["promotion"]["default_promoted"] is False
    )
    assert not path.with_suffix("").exists()
    assert path.name in (RECEIPTS / "README.md").read_text()
    ledger = (
        ROOT / "manifests/maintenance/electronic_structure_production_paths.json"
    ).read_text()
    assert str(path.relative_to(ROOT)) in ledger


def test_response_readme_history_preserves_every_original_byte() -> None:
    """Compaction must retain narrative provenance, not discard it for capacity."""
    original = lzma.decompress((RECEIPTS / "README-history.md.xz").read_bytes())
    assert len(original) == 7861
    assert hashlib.sha256(original).hexdigest() == (
        "c8ae3ed93eb0b9ce0e41c5087fe9793af5f4a244fa6a948019c24d554171366f"
    )
    assert "README-history.md.xz" in (RECEIPTS / "README.md").read_text()


def test_df_resident_receipt_keeps_scoped_frozen_source_and_no_promotion() -> None:
    """Later evidence commits must not relabel the measured production tree."""
    original = lzma.decompress(
        (RECEIPTS / "qualification-resident-df.json.xz").read_bytes()
    )
    assert len(original) == 6645
    assert hashlib.sha256(original).hexdigest() == (
        "5953710db9a0f408172232c168dfdf28a73e5aadf3fea711c4f7821b26238069"
    )
    payload = json.loads(original)
    assert payload["source"]["commit"] == "de67ead96da0c319cb5ea57ce9bd8fbb5fc18836"
    assert payload["source"]["tree"] == "2c9dc04eaaf0aa91419fc3f66664aff72e2780ca"
    assert payload["validation"]["gpu_skipped_or_failed"] == 0
    assert payload["validation"]["prepared_reuse"] is False
    assert payload["promotion"]["default_promoted"] is False
    assert payload["instrumentation"]["compute_sanitizer"]["errors"] == 0
    assert payload["instrumentation"]["compute_sanitizer"]["leaked_bytes"] == 0
    assert (
        payload["instrumentation"]["resident"][9]
        > payload["instrumentation"]["baseline"][9]
    )
    assert "qualification-resident-df.json.xz" in (RECEIPTS / "README.md").read_text()
