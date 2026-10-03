"""Offline guards for the additive UKS correction and historical source binding."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORICAL_PUBLICATION = "1b115455f2d493fb287886730891b9d6c66175d4"


def test_historical_source_map_remains_bound_to_old_publication() -> None:
    note = (ROOT / "README.md").read_text()
    historical = json.loads((ROOT / "provenance.json").read_text())["measured_source"]
    repair = json.loads((ROOT / "uks-primary-eligibility.json").read_text())
    assert HISTORICAL_PUBLICATION in note
    assert "pass that historical\ncheckout as `--repository`" in note
    assert (
        repair["historical_reconstruction"]["public_checkout"] == HISTORICAL_PUBLICATION
    )
    assert repair["historical_reconstruction"]["measured_tree"] == historical["tree"]
    assert (
        repair["historical_reconstruction"]["uks_postimage"]
        == historical["postimages"]["src/dft/uks.cpp"]
    )
    assert (
        historical["postimages"]["src/dft/uks.cpp"]["sha256"]
        != repair["repair_source"]["files"]["src/dft/uks.cpp"]["sha256"]
    )


def test_addendum_preserves_failure_and_distinct_repair_gates() -> None:
    repair = json.loads((ROOT / "uks-primary-eligibility.json").read_text())
    for provider in ("openblas", "scalar"):
        row = repair["diagnosis"][provider]
        assert row["baseline_status"] == 0 and row["candidate_status"] == 4
        assert row["primary_histories_bitwise_equal"]
        assert repair["qualification"][provider]["exact_regression_tests_passed"] == 2
    for row in repair["physical"]["phases"]:
        if row["phase"] == "forced_budget_13":
            assert (
                not row["converged"] and row["iterations"] == row["fock_builds"] == 13
            )
            assert not row["seed_present"] and row["snapshot_rejected"]
        else:
            assert row["converged"] and row["grid_checks"] == 9
            assert row["grid_build_attempts"] == 0
            assert all(
                row["errors"][name] <= limit for name, limit in row["gates"].items()
            )
    assert repair["performance_claim"] is False


def test_all_original_scientific_payloads_are_unchanged() -> None:
    repair = json.loads((ROOT / "uks-primary-eligibility.json").read_text())
    for name, record in repair["historical_payloads"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
