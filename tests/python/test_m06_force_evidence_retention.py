"""M06 compaction preserves observations and binds offline full-data recovery."""

import copy
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import pytest

from tools.compact_m06_force_evidence import SNAPSHOT, compact

ROOT = Path(__file__).resolve().parents[2]


def test_projection_keeps_negative_rows_unknown_work_and_endpoint_times() -> None:
    record = {
        "phase": "warm",
        "complete_seconds": 31.703341123456789,
        "seconds": 31.7,
        "prepare_seconds": 0.003341123456789,
        "forces": [[1e-14, -2e-14, 0.0]],
        "energy": -1221.5372969643936,
        "force_error": 1.5727e-10,
        "gate": False,
        "iterations": 4,
        "fock_builds": 4,
        "native_ks_diagnostic": {
            "history": [{"seconds": 1.0}],
            "components": {"hartree_seconds": 2.0},
            "coulomb_exchange_work": {"exchange_applications": None},
        },
        "native_force_components": {
            "wall_seconds": {"xc": 0.1},
            "work_counts": {"observed": {"native_launches": 47210}},
            "becke_owners": {"partition_ms": 0.1, "pair_visits": 1330642944},
            "resource_bounds": {"resident_capacity": 4096},
        },
    }
    payload = {
        "schema": "original/v1",
        "records": [record],
        "qualification_errors": ["retained failure"],
        "receipts": {"failed.log": "failure"},
        "retained_strict_m06_stress_failures": [{"baseline": [1e-12]}],
    }
    before = copy.deepcopy(payload)
    result = compact(payload, {"path": "original.xz", "sha256": "a" * 64})
    projected = result["records"][0]
    assert payload == before
    assert len(result["records"]) == 1
    for key in record.keys() - {
        "forces",
        "native_ks_diagnostic",
        "native_force_components",
    }:
        assert projected[key] == record[key]
    assert "forces" not in projected
    assert projected["native_ks_diagnostic"] == {
        "coulomb_exchange_work": {"exchange_applications": None}
    }
    assert projected["native_force_components"] == {
        "work_counts": record["native_force_components"]["work_counts"],
        "becke_owners": {"pair_visits": 1330642944},
        "resource_bounds": {"resident_capacity": 4096},
    }
    assert result["qualification_errors"] == payload["qualification_errors"]
    assert (
        result["retained_strict_m06_stress_failures"]
        == payload["retained_strict_m06_stress_failures"]
    )
    assert result["retention"]["snapshot_manifest"] == SNAPSHOT
    assert result["retention"]["full_only_top_level"] == ["receipts"]


@pytest.mark.parametrize(
    ("campaign", "count"),
    [("mgga-integer-powers-20261014", 108), ("m06-rks-force-powers-20261014", 144)],
)
def test_retained_ledgers_bind_full_snapshot_and_keep_all_calls(
    campaign: str, count: int
) -> None:
    directory = ROOT / "benchmarks/results" / campaign
    manifest = json.loads((directory / "manifest.json").read_bytes())
    packed = (directory / manifest["summary"]["path"]).read_bytes()
    assert len(packed) == manifest["summary"]["bytes"]
    assert hashlib.sha256(packed).hexdigest() == manifest["summary"]["sha256"]
    snapshot_bytes = (ROOT / manifest["full_evidence"]["manifest"]).read_bytes()
    assert len(snapshot_bytes) == manifest["full_evidence"]["bytes"]
    assert (
        hashlib.sha256(snapshot_bytes).hexdigest()
        == manifest["full_evidence"]["sha256"]
    )
    snapshot = json.loads(snapshot_bytes)
    assert snapshot["source_revision"] == manifest["full_evidence"]["source_revision"]
    assert snapshot["file_count"] == len(snapshot["files"])
    assert snapshot["total_bytes"] == sum(entry["bytes"] for entry in snapshot["files"])
    ledger = json.loads(lzma.decompress(packed))
    assert ledger["schema"] == "generativeqc.m06-force-endpoint-ledger.v1"
    assert ledger["retention"]["original_capsule"] in snapshot["files"]
    records = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "phase" in value and "complete_seconds" in value:
                records.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(ledger)
    assert len(records) == count
    assert all(record["gate"] is True and record["converged"] for record in records)
    assert all("forces" not in record for record in records)
    assert all("native_scf_ao_work" in record for record in records)
    assert all("work_counts" in record["native_force_components"] for record in records)
    assert all(
        record["native_ks_diagnostic"]["coulomb_exchange_work"]["exchange_applications"]
        is None
        for record in records
    )
