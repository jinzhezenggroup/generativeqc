"""Authenticate default-promotion receipts without repeating real-GPU work."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import block_error
from generativeqc_compiler.common.performance import assess_comparison

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/ks-resident-final-validation-default-20261011"


@pytest.fixture(scope="module")
def promotion_record() -> dict:
    """Load the source-frozen, publication-authenticated standard envelope."""
    record = load_publication_record(BUNDLE)
    storage = record["raw_receipts"]
    raw = (BUNDLE / storage["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == storage["sha256"]
    files = json.loads(lzma.decompress(raw))["files"]
    for field, pointers in (
        ("cases", "case_receipts"),
        ("references", "reference_receipts"),
    ):
        record[field] = {
            atoms: json.loads(files[path]["utf8"])
            for atoms, path in record[pointers].items()
        }
    return record


def test_default_scope_keeps_exact_source_and_formal_qualification_boundaries(
    promotion_record: dict,
) -> None:
    """Scoped timing observations do not fabricate broad performance provenance."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    validate_publication(
        publication,
        {
            entry["path"]: (BUNDLE / entry["path"]).read_bytes()
            for entry in publication["files"]
        },
    )
    assert publication["decision"]["scope"] == "numerical"
    assert promotion_record["revision"] == "5a9b107ca8b9a654d45d3dc15fac45e962323f3b"
    assert not promotion_record["dirty"]
    assert promotion_record["default_promoted"]
    assert promotion_record["performance"]["status"] == "not-run"
    assert promotion_record["stages"]["production"]["status"] == "not-run"
    assert promotion_record["settings"]["selectors"] == {
        "baseline": 0,
        "candidate": None,
    }
    assert promotion_record["gates"] == {
        "energy_eh": 1e-8,
        "force_eh_per_bohr": 1e-7,
        "relative_floor": 0.02,
        "relative_mad_multiplier": 2,
    }


@pytest.mark.parametrize("atoms", ["48", "96"])
def test_actual_unset_samples_keep_numerics_work_timing_and_lease_gates(
    promotion_record: dict,
    atoms: str,
) -> None:
    """Recompute each complete cohort without pooling the timed-out 96 attempt."""
    case = promotion_record["cases"][atoms]
    analysis = promotion_record["independent_analyses"][atoms]
    assert case["status"] == "measured" and case["stage"] == "complete"
    assert len(case["samples"]) == 20
    assert case["selectors"] == {"baseline": 0, "candidate": None}
    assert case["slurm_job"] == ("7271" if atoms == "48" else "7276")
    assert all(
        arm["native_build"]["library_sha256"]
        == promotion_record["native_library_sha256"]
        for arm in case["arms"].values()
    )
    checkpoints = {
        (row["selection"], row["phase"], row["label"]): row["blob_sha256"]
        for row in case["checkpoints"]
    }
    assert len(checkpoints) == 8
    for side in case["selectors"]:
        for phase in ("warm", "moved-warm"):
            assert (
                checkpoints[side, phase, "before"] == checkpoints[side, phase, "after"]
            )
    semantic_work = {}
    for sample in case["samples"]:
        diagnostic = sample["diagnostics"]
        oracle = next(
            row
            for row in promotion_record["references"][atoms]["records"]
            if row["geometry"] == sample["geometry"]
            and row["phase"] in ("cold", "moved")
        )
        assert block_error(diagnostic["energy"], oracle["energy"], atol=1e-8, rtol=0)[
            "passed"
        ]
        assert block_error(diagnostic["forces"], oracle["forces"], atol=1e-7, rtol=0)[
            "passed"
        ]
        assert diagnostic["status"] == 0 and diagnostic["converged"]
        assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
        assert diagnostic["warm_start_used"] and not diagnostic["warm_start_fallback"]
        assert (
            sample["frozen_seed_blob_sha256"]
            == checkpoints[sample["selection"], sample["workload"], "before"]
        )
        force = diagnostic["native_force_components"]
        bounds = force["resource_bounds"]
        assert (
            bounds["additional_device_peak_bound"] <= bounds["additional_device_budget"]
        )
        assert (
            bounds["additional_host_numeric_bound"] <= bounds["additional_host_budget"]
        )
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
        assert work == semantic_work.setdefault(sample["geometry"], work)
        exported = diagnostic["native_snapshot_export"]
        legacy_bytes = exported["spins"] * (
            (3 * exported["nao"] ** 2 + exported["nao"]) * 8 + 4
        )
        assert exported["selector"] == case["selectors"][sample["selection"]]
        assert exported["reads"] == 1
        assert exported["d2h_bytes"] - legacy_bytes == (
            112 if sample["selection"] == "candidate" else 0
        )
        assert exported["synchronizations"] == (
            3 if sample["selection"] == "candidate" else 1
        )
    for phase in ("warm", "moved-warm"):
        assessment = assess_comparison(
            [row for row in case["samples"] if row["workload"] == phase]
        )
        assert (
            assessment == case["assessments"][phase] == analysis["assessments"][phase]
        )
        assert assessment["status"] == "pass"
        assert all(
            assessment["workloads"][phase][side]["samples"] == 5
            for side in case["selectors"]
        )
    for setup in case["setup"]:
        original = setup["diagnostics"]["native_snapshot_export"]
        repeated = setup["diagnostics"]["repeated_lease_export"]
        assert repeated["identity"] == original["identity"]
        assert repeated["reads"] == repeated["synchronizations"] == 1
        assert repeated["d2h_bytes"] == repeated["spins"] * (
            (3 * repeated["nao"] ** 2 + repeated["nao"]) * 8 + 4
        )


def test_failed_admission_and_timeout_receipts_remain_lossless(
    promotion_record: dict,
) -> None:
    """A misleading zero trap receipt cannot qualify an unfinished cohort."""
    storage = promotion_record["raw_receipts"]
    raw = (BUNDLE / storage["path"]).read_bytes()
    assert len(raw) == storage["bytes"]
    assert hashlib.sha256(raw).hexdigest() == storage["sha256"]
    files = json.loads(lzma.decompress(raw))["files"]
    for receipt in files.values():
        decoded = receipt["utf8"].encode()
        assert len(decoded) == receipt["bytes"]
        assert hashlib.sha256(decoded).hexdigest() == receipt["sha256"]
    for atoms in (48, 96):
        failed = json.loads(
            files[f"default-promotion-v1/endpoint-failed-receipts/paired-{atoms}.json"][
                "utf8"
            ]
        )
        assert failed["status"] == "failed" and not failed["samples"]
    partial = promotion_record["partial_attempt"]
    assert partial["samples"] == 17
    assert partial["scheduler_terminal"]["state"] == "TIMEOUT"
    interrupted = json.loads(
        files["default-promotion-v2/endpoint-receipts/paired-96.json"]["utf8"]
    )
    assert interrupted["stage"] != "complete" and len(interrupted["samples"]) == 17
    assert (
        files["default-promotion-v2/endpoint-receipts/exit-status.txt"]["utf8"].strip()
        == "0"
    )
    assert "for atoms in 96" in files["default-promotion-v3/run-endpoint.sh"]["utf8"]
    assert (
        "trap 'exit 143' TERM" in files["default-promotion-v3/run-endpoint.sh"]["utf8"]
    )
    assert (
        "ccache version 4.5.1"
        in files["default-promotion-v2/build-receipts/cache-version.txt"]["utf8"]
    )
