"""Profitability cannot be inferred from failed, mismatched or duplicated timings."""

import copy

import pytest

from benchmarks import structured_scf_endpoint_report as report


def receipt(mask: int, repeat: int, seconds: float = 1.0) -> dict:
    return {
        "model": {"method": "rks"},
        "source": "source",
        "library_sha256": "library",
        "artifact_sha256": {"library": "library"},
        "batch": 3,
        "graph": False,
        "host": "node1",
        "gpu_inventory": "GPU-uuid, RTX5090, driver",
        "cuda_visible_devices": "1",
        "reference_sha256": {"0": "reference0", "1": "reference1"},
        "provider_controls": {},
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


def observation(mask: int = 127) -> dict:
    return {**receipt(mask, 0), "library_operations": [0, 1, 2, 3, 4, 5, 6]}


def test_matched_endpoint_pairs_are_profitable_without_promoting_defaults() -> None:
    baseline, candidate = pairs()
    result = report.compare(baseline, candidate, 127, [observation()])
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
    assert (
        report.compare(baseline, candidate, 127, [observation()])["profitable"] is False
    )


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


def test_prepared_candidates_are_not_execution_evidence() -> None:
    baseline, candidate = pairs()
    result = report.compare(baseline, candidate, 127)
    assert result["timing_profitable"] is True
    assert result["profitable"] is None


def test_foreign_trace_cannot_qualify_this_endpoint() -> None:
    baseline, candidate = pairs()
    foreign = observation()
    foreign["source"] = "other"
    assert report.compare(baseline, candidate, 127, [foreign])["profitable"] is None


def trace(execution: str = "stream") -> dict:
    return {
        "operation": "structured_gram",
        "execution": execution,
        "valid": True,
        "cuda_error": 0,
        "counters": {"operation": 0, "provider_cublas": 1, "submitted_states": 3},
    }


def test_stream_library_submissions_are_observed() -> None:
    measured = receipt(127, 0)
    measured["instrumented"] = True
    assert report.execution_observation(measured, [trace()])["library_operations"] == [
        0
    ]


def test_capture_alone_is_not_executed_work() -> None:
    measured = receipt(127, 0)
    measured["instrumented"] = True
    assert not report.execution_observation(measured, [trace("graph_capture")])[
        "library_operations"
    ]


def test_hf_captured_density_requires_successful_post_eigen_graph_launches() -> None:
    measured = receipt(127, 0)
    measured["instrumented"] = True
    measured["model"]["method"] = "rhf"
    assert not report.execution_observation(measured, [trace("graph_capture")])[
        "library_operations"
    ]
    for row in measured["rows"]:
        row["graph_post_launches"] = 1
        row["bindings"] = [{"operation": 0, "provider": "cublas"}]
    assert report.execution_observation(measured, [trace("graph_capture")])[
        "library_operations"
    ] == [0]


@pytest.mark.parametrize("execution", ["stream", "graph_capture"])
def test_failed_hf_library_attempt_cannot_authenticate_generated_retry(
    execution: str,
) -> None:
    measured = receipt(127, 0)
    measured["instrumented"] = True
    measured["model"]["method"] = "rhf"
    for row in measured["rows"]:
        row["graph_post_launches"] = 1
        row["bindings"] = [{"operation": 0, "provider": "generated.cuda"}]
    assert not report.execution_observation(measured, [trace(execution)])[
        "library_operations"
    ]


def test_invalid_trace_is_not_library_execution_evidence() -> None:
    measured = receipt(127, 0)
    measured["instrumented"] = True
    invalid = trace()
    invalid["valid"] = False
    assert not report.execution_observation(measured, [invalid])["library_operations"]


def test_extra_phase_cannot_hide_an_unaccepted_row() -> None:
    baseline, candidate = pairs()
    extra = copy.deepcopy(candidate[0]["rows"][0])
    extra["phase"] = 5
    extra["assessment"]["accepted"] = False
    candidate[0]["rows"].append(extra)
    assert (
        report.compare(baseline, candidate, 127, [observation()])["profitable"] is None
    )
