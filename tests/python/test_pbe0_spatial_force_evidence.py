"""Authenticate source-matched spatial-reuse evidence without GPU re-execution."""

from __future__ import annotations

import hashlib
import json
import lzma
from functools import lru_cache
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import block_error, canonical_hash
from generativeqc_compiler.common.performance import assess_comparison
from generativeqc_compiler.common.timing import interleaved_selection_order

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/pbe0-spd-spatial-reuse-20261011"


@lru_cache(maxsize=1)
def raw_receipts() -> dict:
    """Decode once while retaining exact individual receipt-byte identities."""
    return json.loads(lzma.decompress((BUNDLE / "raw-receipts.json.xz").read_bytes()))


def member(name: str) -> dict:
    return json.loads(raw_receipts()["members"][name]["text"])


def test_publication_keeps_measured_dirty_source_and_scoped_decision() -> None:
    """A later commit must not replace the measured base or inflate formal scope."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    freeze = member("source-freeze.json")
    assert manifest["source"] == {"revision": freeze["base"], "dirty": True}
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert len(evidence["block_errors"]) == 120
    assert len(evidence["timings"]) == 40
    assert evidence["scoped_decision"]["status"] == (
        "qualified-scoped-default-promotion"
    )
    for name, original in (
        ("production.patch.xz", "production.patch"),
        ("qualification-original.patch.xz", "qualification.patch"),
    ):
        assert (
            hashlib.sha256(lzma.decompress(files[name])).hexdigest()
            == (freeze["digests"][original])
        )


def test_raw_members_preserve_bytes_and_do_not_fabricate_npz_reconstruction() -> None:
    """Checkpoint digests authenticate local checks but cannot recreate blobs."""
    payload = raw_receipts()
    for retained in payload["members"].values():
        data = retained["text"].encode("utf-8")
        assert len(data) == retained["bytes"]
        assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    assert not any(name.endswith(".npz") for name in payload["members"])
    assert sum(name.endswith(".npz") for name in payload["omitted"]) == 16


@pytest.mark.parametrize("atoms", [48, 96])
def test_all_complete_calls_and_five_pair_assessments(atoms: int) -> None:
    """Recompute every E/F gate and retain original order, outliers and work."""
    record = member(f"performance{atoms}-receipts/paired-{atoms}.json")
    reference = member(f"reference-{atoms}-retained-perf5.json")
    assert record["stage"] == "complete" and record["status"] == "measured"
    assert record["repeats"] == 5
    assert record["spd_selectors"] == {"baseline": "0", "candidate": None}
    assert record["device_validation_selectors"] == {"baseline": 1, "candidate": 1}
    calls = [arm["construction_initialization"] for arm in record["arms"].values()]
    calls += record["setup"] + record["priming"] + record["samples"]
    assert len(calls) == 30 and len(record["samples"]) == 20
    for call in calls:
        diagnostic = call["diagnostics"]
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == call["geometry"] and row["phase"] in {"cold", "moved"}
        )
        assert diagnostic["status"] == 0 and diagnostic["converged"]
        assert block_error(
            [diagnostic["energy"]], [oracle["energy"]], atol=1e-8, rtol=0
        )["passed"]
        assert block_error(diagnostic["forces"], oracle["forces"], atol=1e-7, rtol=0)[
            "passed"
        ]
    work_by_geometry = {}
    for sample in record["samples"]:
        diagnostic = sample["diagnostics"]
        force = diagnostic["native_force_components"]
        assert sample["synchronized"]
        assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
        work = {
            "scf_ao": {
                key: value
                for key, value in diagnostic["native_scf_ao_work"].items()
                if key != "discovery_seconds"
            },
            "force_ao": {
                key: value
                for key, value in force["resident_ao_selection"]["work"].items()
                if key != "discovery_seconds"
            },
            "force_counts": force["work_counts"],
            "grid": force["grid_work_plan"],
            "point": diagnostic["actual_point_selection"],
        }
        assert work == work_by_geometry.setdefault(sample["geometry"], work)
    for phase in ("warm", "moved-warm"):
        samples = [row for row in record["samples"] if row["workload"] == phase]
        assert [row["selection"] for row in samples] == list(
            interleaved_selection_order(5)
        )
        assessment = assess_comparison(samples)
        assert assessment == record["assessments"][phase]
        assert assessment["status"] == "pass"


def test_safety_covers_both_layouts_without_relabeling_failed_wrappers() -> None:
    """Two non-vacuous probes cover four variants; whole-run failures remain."""
    receipts = raw_receipts()["members"]
    assert receipts["remote-safety-driver-exit.txt"]["text"].strip() == "143"
    assert receipts["remote-scoped-racecheck-driver-exit.txt"]["text"].strip() == "1"
    assert receipts["combined-safety-receipts/exit-status.txt"]["text"].strip() == "0"
    separate = member("scoped-race-v2-receipts/checked-routes.json")
    assert separate["cached_variants"] == [0, 24, 0, 24]
    assert separate["bad_width"] == separate["unknown_variant"] == 0
    for tool in ("memcheck", "racecheck"):
        combined = member(f"combined-safety-receipts/{tool}-checked-routes.json")
        assert combined["cached_variants"] == [24, 0, 24, 0]
        assert combined["bad_width"] == combined["unknown_variant"] == 0
        assert (
            receipts[f"combined-safety-receipts/{tool}-exit-status.txt"]["text"].strip()
            == "0"
        )
    for atoms in (48, 96):
        preload = receipts[f"performance{atoms}-receipts/preload-policy.txt"]["text"]
        assert "clean_timing_ld_preload=unset" in preload


def test_storage_compaction_preserves_canonical_measurements() -> None:
    """Storage savings cannot alter old measurements or discard failed records."""
    metadata = json.loads((BUNDLE / "storage-compaction.json").read_text())
    assert metadata["original_git_revision"] == member("source-freeze.json")["base"]
    assert len(metadata["files"]) == 7
    assert metadata["saved_bytes"] == 668780
    for row in metadata["files"]:
        data = (ROOT / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["stored_sha256"]
        assert len(data) == row["stored_bytes"]
        assert canonical_hash(json.loads(data)) == row["canonical_sha256"]
        assert row["original_bytes"] > row["stored_bytes"]
