"""Lossless transport must preserve historical DF gates and original identities."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.summarize_df_promotion import summarize

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/results/generated-df-values-142"


def test_lossless_records_keep_stored_and_original_identities() -> None:
    """Authenticate both transport and decoded bytes, not only parsed flags."""
    storage = json.loads((CAMPAIGN / "storage.json").read_text())
    promotion = json.loads((CAMPAIGN / "promotion.json").read_text())
    for entry in storage["files"]:
        packed = (ROOT / entry["stored_path"]).read_bytes()
        raw = gzip.decompress(packed)
        assert len(packed) == entry["stored_bytes"]
        assert hashlib.sha256(packed).hexdigest() == entry["stored_sha256"]
        assert len(raw) == entry["original_bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["original_sha256"]
        name = Path(entry["original_path"]).stem
        assert entry["original_sha256"] == promotion["input_hashes"][name]


def test_complete_promotion_is_identical_for_plain_and_gzip(tmp_path: Path) -> None:
    """Recompute all original gates on both representations without any GPU."""
    for name in ("isolated", "source", "endpoints", "automatic"):
        plain = CAMPAIGN / f"{name}.json"
        raw = (
            plain.read_bytes()
            if plain.exists()
            else gzip.decompress(plain.with_suffix(".json.gz").read_bytes())
        )
        (tmp_path / plain.name).write_bytes(raw)
    packed_result = summarize(CAMPAIGN)
    assert packed_result == summarize(tmp_path)
    promotion = json.loads((CAMPAIGN / "promotion.json").read_text())
    assert packed_result["input_hashes"] == promotion["input_hashes"]
    assert packed_result["passed"]
    assert packed_result["isolated_count"] == 162
    assert packed_result["source_count"] == 32
    assert packed_result["endpoint_count"] == 20
