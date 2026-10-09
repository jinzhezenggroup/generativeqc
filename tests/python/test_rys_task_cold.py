"""Cold acceptance must retain every measured repeat and its numerical gates."""

from copy import deepcopy

import pytest

from benchmarks.rys_task_cold import configure_lowering, summarize


def cohort() -> list[dict]:
    """Return a matched synthetic cohort; no device work occurs in these tests."""
    return [
        {
            "schema": "rys-task-cold-ef.v2",
            "mode": mode,
            "k_task_schedule": "work",
            "status": "measured",
            "pid": pair * 2 + offset,
            "library_sha256": "same-binary",
            "protocol": {"atoms": 96},
            "host": "same-host",
            "slurm_job_id": "same-job",
            "cuda_visible_devices": "assigned",
            "complete_seconds": seconds,
            "fock_builds": 19,
            "converged": True,
            "warm_start_used": False,
            "context_primed": False,
            "acceptance": {"gate": True, "energy_error": 1e-11, "force_error": 1e-10},
        }
        for pair in range(3)
        for offset, (mode, seconds) in enumerate(
            (("incumbent", 100.0), ("rys-task", 95.0))
        )
    ]


def test_complete_matched_cohort_reports_work_and_cold_ratio() -> None:
    result = summarize(cohort())
    assert result["gate"]
    assert result["speedup"] == pytest.approx(100 / 95)
    assert result["fock_builds"] == {"incumbent": [19] * 3, "rys-task": [19] * 3}
    assert result["k_task_schedule"] == "work"


@pytest.mark.parametrize("schedule", [None, "fill", "incumbent", "typo"])
def test_lowering_comparison_cannot_hide_a_changed_queue_schedule(
    schedule: str,
) -> None:
    """A same-binary comparison must freeze the independent scheduling axis too."""
    samples = cohort()
    samples[-1]["k_task_schedule"] = schedule
    assert not summarize(samples)["gate"]


def test_new_records_require_schedule_provenance() -> None:
    samples = cohort()
    for sample in samples:
        del sample["k_task_schedule"]
    result = summarize(samples)
    assert not result["gate"]
    assert "missing or invalid K task schedule" in result["failures"]


def test_historical_records_retain_unspecified_schedule_without_relabeling() -> None:
    samples = cohort()
    for sample in samples:
        sample["schema"] = "rys-task-cold-ef.v1"
        del sample["k_task_schedule"]
    result = summarize(samples)
    assert result["gate"]
    assert result["k_task_schedule"] is None


@pytest.mark.parametrize("error", (float("nan"), float("inf"), 2e-7))
def test_no_inaccurate_repeat_can_be_excluded_from_the_gate(error: float) -> None:
    samples = cohort()
    samples[-1]["acceptance"]["force_error"] = error
    result = summarize(samples)
    assert not result["gate"]
    assert "invalid independent force_error" in result["failures"]


@pytest.mark.parametrize(
    "field,value",
    (
        ("library_sha256", "another-binary"),
        ("cuda_visible_devices", "another-device"),
        ("context_primed", True),
        ("status", "running"),
        ("complete_seconds", float("nan")),
    ),
)
def test_unmatched_or_non_cold_endpoints_fail(field: str, value: object) -> None:
    samples = deepcopy(cohort())
    samples[-1][field] = value
    assert not summarize(samples)["gate"]


def test_a_slow_candidate_is_not_a_performance_success() -> None:
    samples = cohort()
    for sample in samples:
        if sample["mode"] == "rys-task":
            sample["complete_seconds"] = 101.0
    result = summarize(samples)
    assert not result["gate"]
    assert "no complete cold E+F median advantage" in result["failures"]


def test_a_fast_median_cannot_hide_regressing_mean() -> None:
    samples = cohort()
    candidates = [sample for sample in samples if sample["mode"] == "rys-task"]
    for sample, seconds in zip(candidates, (80.0, 80.0, 200.0), strict=True):
        sample["complete_seconds"] = seconds
    result = summarize(samples)
    assert result["median_seconds"]["rys-task"] == 80.0
    assert not result["gate"]
    assert "no complete cold E+F mean advantage" in result["failures"]


def test_default_worker_clears_inherited_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "GENERATIVEQC_DIRECT_K_FOCK_LOWERING"
    monkeypatch.setenv(variable, "rys-task")
    assert configure_lowering("default") is None
    assert configure_lowering("incumbent") == "incumbent"


def test_default_cohort_requires_the_actual_unset_path() -> None:
    samples = cohort()
    for sample in samples:
        if sample["mode"] == "rys-task":
            sample.update(mode="default", k_lowering_environment=None)
    result = summarize(samples)
    assert result["gate"]
    assert result["median_seconds"]["default"] == 95.0
    samples[-1]["k_lowering_environment"] = "rys-task"
    assert not summarize(samples)["gate"]


def test_default_and_explicit_candidates_cannot_be_mixed() -> None:
    samples = cohort()
    samples[-1].update(mode="default", k_lowering_environment=None)
    assert not summarize(samples)["gate"]
