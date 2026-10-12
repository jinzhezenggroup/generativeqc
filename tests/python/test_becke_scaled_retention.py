"""Retained Becke evidence keeps exact inputs and unsuccessful measurements."""

import gzip
import json
import lzma
from pathlib import Path

from tools.generativeqc_validation.record import load_json, load_publication_record
from tools.generativeqc_validation.retention import digest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/becke-binary-scaled-20261012"
HISTORICAL = ROOT / "benchmarks/results/becke-normalize-20261006"


def test_receipts_preserve_failed_small_owner_and_qualified_large_owner() -> None:
    """Storage updates do not reclassify the failed moved 48-atom portfolio."""
    receipts = load_json(BUNDLE / "raw-receipts.json.xz")
    assert set(receipts) == {"48", "96"}
    small, large = receipts["48"], receipts["96"]
    assert small["status"] == small["stage"] == "failed"
    assert small["error"] and small["failed_command"]
    assert not small["assessments"]["moved-warm"]["significant"]
    assert large["status"] == "qualified" and large["stage"] == "complete"
    for record in (small, large):
        assert record["repeats"] == 5
        assert len(record["samples"]) == 20
        for phase in ("warm", "moved-warm"):
            assessment = record["assessments"][phase]
            assert assessment["significant"] == (
                assessment["relative_improvement"] > assessment["robust_noise_floor"]
            )
    assert all(row["significant"] for row in large["assessments"].values())


def test_historical_becke_publication_resolves_lossless_companions() -> None:
    """Logical reader names survive storage changes and retain decoded identity."""
    manifest = load_json(HISTORICAL / "publication.json")
    for entry in manifest["files"]:
        if "decoded_sha256" not in entry:
            continue
        stored = (HISTORICAL / entry["path"]).read_bytes()
        decoded = (
            gzip.decompress(stored)
            if entry["path"].endswith(".gz")
            else lzma.decompress(stored)
        )
        assert len(decoded) == entry["decoded_bytes"]
        assert digest(decoded) == entry["decoded_sha256"]
        if entry["role"] == "samples":
            name = entry["path"].removesuffix(".xz").removesuffix(".gz")
            assert load_publication_record(
                HISTORICAL, role="samples", name=name
            ) == json.loads(decoded)
