"""Profitability cannot be inferred from failed, mismatched or duplicated timings."""

import copy

import pytest

from benchmarks import structured_scf_endpoint_report as report


def receipt(mask: int, repeat: int, seconds: float = 1.0) -> dict:
    return {
        "model": {"method": "rks"},
        "source": "source",
        "library_sha256": "library",
        "batch": 3,
        "graph": False,
        "host": "node1",
        "gpu_inventory": "GPU-uuid, RTX5090, driver",
        "cuda_visible_devices": "1",
        "mask": mask,
        "repeat": repeat,
        "status": "PASS",
        "instrumented": False,
        "rows": [
            {
                "phase": phase,
                "item": item,
                "assessment": {"accepted": True},
                "complete_seconds": seconds,
                "iterations": 4,
            }
            for phase in range(5)
            for item in range(3)
        ],
    }


def pairs() -> tuple[list[dict], list[dict]]:
    return (
        [receipt(0, repeat, 2.0) for repeat in range(3)],
        [receipt(127, repeat) for repeat in range(3)],
    )


def test_matched_endpoint_pairs_are_profitable_without_promoting_defaults() -> None:
    baseline, candidate = pairs()
    result = report.compare(baseline, candidate, 127)
    assert result["profitable"] is True
    assert result["default_promoted"] is False
    assert result["matched_pairs"] == 3
    assert len(result["phases"]["cold"]["paired_speedups"]) == 3


@pytest.mark.parametrize("field", report.IDENTITY_FIELDS)
def test_identity_mismatch_rejects_profitability(field: str) -> None:
    baseline, candidate = pairs()
    candidate[0][field] = "other"
    assert report.compare(baseline, candidate, 127)["profitable"] is None


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("instrumented", True)])
def test_invalid_runs_cannot_supply_clean_timing(field: str, value: str | bool) -> None:
    baseline, candidate = pairs()
    candidate[0][field] = value
    assert report.compare(baseline, candidate, 127)["profitable"] is None


def test_three_batch_members_are_not_three_repetitions() -> None:
    baseline, candidate = pairs()
    assert report.compare(baseline[:1], candidate[:1], 127)["profitable"] is None


def test_duplicate_pair_cannot_increase_confidence() -> None:
    baseline, candidate = pairs()
    baseline[2] = copy.deepcopy(baseline[1])
    candidate[2] = copy.deepcopy(candidate[1])
    assert report.compare(baseline, candidate, 127)["profitable"] is None


def test_cold_preparation_cost_is_not_hidden_by_fast_execution() -> None:
    baseline, candidate = pairs()
    for value in candidate:
        for row in value["rows"]:
            if row["phase"] == 0:
                row["cold_total_seconds"] = 5.0
    assert report.compare(baseline, candidate, 127)["profitable"] is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_timing_is_rejected(invalid: float) -> None:
    baseline, candidate = pairs()
    candidate[0]["rows"][0]["complete_seconds"] = invalid
    assert report.compare(baseline, candidate, 127)["profitable"] is None


def test_missing_member_is_not_a_successful_complete_batch() -> None:
    baseline, candidate = pairs()
    candidate[0]["rows"].pop()
    assert report.compare(baseline, candidate, 127)["profitable"] is None


def test_failing_scientific_row_cannot_be_cherry_picked() -> None:
    baseline, candidate = pairs()
    candidate[0]["rows"][0]["assessment"]["accepted"] = False
    assert report.compare(baseline, candidate, 127)["profitable"] is None


def test_hf_signed_gram_is_not_claimed_as_executed() -> None:
    baseline, candidate = pairs()
    for generated, offered in zip(baseline, candidate, strict=True):
        generated["model"]["method"] = offered["model"]["method"] = "rhf"
        offered["mask"] = 2
    assert report.compare(baseline, candidate, 2)["profitable"] is None
