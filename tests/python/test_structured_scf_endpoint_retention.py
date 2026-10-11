"""Exact-byte storage and complete decoded endpoint records remain recoverable."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/structured-scf-routes-20261011"


@pytest.mark.parametrize(
    ("name", "revision", "original_bytes", "expected_sha256"),
    [
        (
            "endpoint-history-607d47b1e.json",
            "bb535d14e7056400dfbc5bbe8ad27754e804324d",
            28173,
            "042ddf5c2ac4583aea3c8398099b11e108b3a0adc4d093bcd700ab3e7537359a",
        ),
        (
            "endpoint-performance-607d47b1e.json",
            "e7abb59975bd3c0a9efdd8f82eeb1b74854a4f31",
            11862,
            "90899eadbc2d0b3971ca310bfd1a2f1ad8427c64800d43a9fc98f4bb63055089",
        ),
        (
            "endpoint-performance.json",
            "dd50437a47c000f1c023e52786956b8f8c8a487f",
            99039,
            "ba59c84f457a5e1207d9500fb46cfc96b6693287a687d0b7fe2a57f97644bbc5",
        ),
        (
            "endpoint-science-607d47b1e.json",
            "e7abb59975bd3c0a9efdd8f82eeb1b74854a4f31",
            108027,
            "1dcf7f4dc351f8993712c30874f5e84139e0adbddfe9cf797651c4874ed97895",
        ),
        (
            "endpoint-science.json",
            "dd50437a47c000f1c023e52786956b8f8c8a487f",
            77292,
            "be1302b8063616c25e9bdd93beb77bdff556c97395dec9feafeb770f6dc39cc5",
        ),
        (
            "endpoint-validation-607d47b1e.json",
            "e7abb59975bd3c0a9efdd8f82eeb1b74854a4f31",
            10685,
            "ad1d6be21a6ee690c552a712cfa065534242eee281f55ef81d76390bed4a2129",
        ),
        (
            "endpoint-validation.json",
            "dd50437a47c000f1c023e52786956b8f8c8a487f",
            11077,
            "8ba9da75b3fb46d9204fce6ee7f44338c93571b2b8764cb87a834d132065c79a",
        ),
    ],
)
def test_endpoint_receipt_restores_its_hash_bound_original(
    name: str, revision: str, original_bytes: int, expected_sha256: str
) -> None:
    manifest = json.loads((FAMILY / "endpoint-storage.json").read_text())
    entries = manifest["files"]
    assert len(entries) == 7
    assert len({entry["original_path"] for entry in entries}) == 7
    assert len({entry["stored_path"] for entry in entries}) == 7
    original_path = FAMILY / name
    entry = next(
        entry
        for entry in entries
        if entry["original_path"] == original_path.relative_to(ROOT).as_posix()
    )
    assert entry["original_revision"] == revision
    assert entry["stored_path"] == entry["original_path"] + ".gz"
    stored_path = ROOT / entry["stored_path"]
    stored = stored_path.read_bytes()
    original = gzip.decompress(stored)
    assert not original_path.exists()
    assert len(stored) == entry["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    assert len(original) == entry["original_bytes"] == original_bytes
    assert hashlib.sha256(original).hexdigest() == entry["original_sha256"]
    assert entry["original_sha256"] == expected_sha256
    header = f"blob {len(original)}\0".encode()
    blob = hashlib.sha1(header + original, usedforsecurity=False).hexdigest()
    assert blob == entry["original_git_blob"]
    record = json.loads(original)
    assert load_json(stored_path) == record
    assert record["source_commit"] == (
        "607d47b1e2383bdd7618ddc9264f9d7a8205d6b2"
        if "607d47b1e" in name
        else "7d8f4ba22883f9c3accfec70a3e8640510ffbe2c"
    )
    assert record["default_promoted"] is False
    assert record["production_library_qualified"] is False


def expand_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = table["columns"]
    assert len(columns) == len(set(columns))
    rows = [
        {**table["defaults"], **dict(zip(columns, row, strict=True))}
        for row in table["rows"]
    ]
    for field, indices in table["absent"].items():
        for index in indices:
            del rows[index][field]
    return rows


def logical_science_reference(record: dict[str, Any]) -> dict[str, Any]:
    name = record["shared_scientific_inputs"]
    manifest = json.loads((FAMILY / "endpoint-storage.json").read_text())
    matching = [
        entry
        for entry in manifest["files"]
        if entry["original_path"] == (FAMILY / name).relative_to(ROOT).as_posix()
    ]
    assert len(matching) == 1
    return load_json(ROOT / matching[0]["stored_path"])


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_current_tables_restore_complete_pre_compaction_values() -> None:
    science = load_json(FAMILY / "endpoint-science.json.gz")
    restored = copy.deepcopy(science)
    restored["runs"] = expand_table(restored.pop("run_table"))
    restored.pop("compact_table_encoding")
    reference = restored["independent_reference"]
    reference["records"] = expand_table(reference.pop("record_table"))
    # Full parsed-value digests of the publications at e7abb59975, independently
    # computed before the author's lossless table/identity deduplication.
    assert (
        canonical_digest(restored)
        == "ee1205e77c4a6b9089682e49eae67879eebed4126300e5b50502980ed10f1d3f"
    )

    performance = load_json(FAMILY / "endpoint-performance.json.gz")
    assert logical_science_reference(performance) == science
    assert (
        logical_science_reference(
            load_json(FAMILY / "endpoint-history-607d47b1e.json.gz")
        )
        == science
    )
    for profile in performance["profiles"]:
        identity = {**performance["identity_defaults"], **profile["identity"]}
        key = identity.pop("model_key")
        model = science["models"][key]
        identity.update(
            model=model,
            reference_sha256=science["reference_sha256_by_model"][key],
            artifact_sha256=science["artifact_sha256_by_method"][model["method"]],
        )
        profile["identity"] = identity
    for field in (
        "shared_scientific_inputs",
        "identity_defaults",
        "profile_identity_encoding",
    ):
        performance.pop(field)
    assert (
        canonical_digest(performance)
        == "e99e28b55ca2d85dfa768e863863e94f69954ec13f44f2e7d3971eee622d58e9"
    )


def test_historical_summary_keeps_its_complete_original_decisions() -> None:
    original = load_json(FAMILY / "endpoint-performance-607d47b1e.json.gz")
    decisions = original["historical_decisions"]
    # This pins every field from the prior history publication. The author's
    # latest summary is intentionally non-reversible; the complete original
    # preserves all provider controls, policies and comparison details.
    assert (
        canonical_digest(decisions)
        == "8f1690bdb4617bc8e370129bea2a0815a1efbf7896e9c73618e369a45c6abfe8"
    )
    summary = load_json(FAMILY / "endpoint-history-607d47b1e.json.gz")
    assert "historical_decisions" not in summary
    assert summary["historical_decision_columns"] == [
        "case",
        "method",
        "disable_md_j",
        "mask",
        "profitable",
        "executed_library_operations",
        "reason",
    ]
    expected_rows = [
        [
            group["case"],
            group["method"],
            group["provider_controls"]["GENERATIVEQC_DISABLE_MD_J"],
            comparison["mask"],
            comparison["profitable"],
            comparison.get("executed_library_operations"),
            comparison.get("reason"),
        ]
        for group in decisions
        for comparison in group["comparisons"]
    ]
    assert summary["historical_decision_rows"] == expected_rows
