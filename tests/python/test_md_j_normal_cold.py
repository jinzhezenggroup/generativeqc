"""Acceptance is based on complete matched normal/MD-J endpoints, not kernels."""

from copy import deepcopy

import pytest

from benchmarks.md_j_normal_cold import collect


def samples() -> list[dict]:
    """A valid three-pair cohort with an independently qualified energy."""
    return [
        {
            "mode": mode,
            "seconds": seconds,
            "energy": -2441.5,
            "physical_residual": 1e-11,
            "converged": True,
            "backend": "cuda",
            "warm_start_used": False,
            "fock_builds": 24,
            "md_j_calls": calls,
            "pid": 10 + 2 * repetition + index,
            "library_sha256": "same-library",
            "input_sha256": "same-input",
            "normal_revision": "same-master-revision",
            "md_j_default": mode == "md-j",
            "slurm_job_id": "same-job",
            "cuda_visible_devices": "assigned-device",
            "natoms": 96,
            "nao": 768,
            "grid_points": 294912,
            "grid_points_source": "native-diagnostic",
            "precision": "fp64",
            "screening": 1e-12,
            "energy_tolerance": 1e-11,
            "density_tolerance": 1e-9,
            "process_cold": True,
            "context_primed": False,
            "supplied_density": False,
        }
        for repetition in range(3)
        for index, (mode, seconds, calls) in enumerate(
            (("normal", 110.0, 0), ("md-j", 70.0, 24))
        )
    ]


def test_complete_matched_normal_cold_advantage() -> None:
    result = collect(samples(), -2441.5)
    assert result["performance_acceptance_passed"]
    assert result["normal_over_md_speedup"] == pytest.approx(110 / 70)
    assert result["j_only_toggle"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("library_sha256", "different-library"),
        ("input_sha256", "different-input"),
        ("normal_revision", "different-master-revision"),
        ("slurm_job_id", "different-job"),
        ("cuda_visible_devices", "different-device"),
        ("converged", False),
        ("backend", "cpu"),
        ("warm_start_used", True),
        ("physical_residual", 1e-6),
        ("energy", -2441.0),
        ("md_j_calls", 0),
        ("md_j_default", False),
        ("pid", 10),
        ("natoms", 12),
        ("nao", 96),
        ("precision", "auto"),
        ("grid_points", 1000),
        ("grid_points_source", "configured-count"),
        ("screening", 1e-8),
        ("energy_tolerance", 1e-6),
        ("density_tolerance", 1e-4),
        ("context_primed", True),
        ("supplied_density", True),
        ("process_cold", False),
    ],
)
def test_incomplete_or_unmatched_cold_cannot_pass(field: str, value: object) -> None:
    cohort = deepcopy(samples())
    cohort[1][field] = value
    assert not collect(cohort, -2441.5)["performance_acceptance_passed"]


def test_slower_md_and_partial_cohorts_fail() -> None:
    cohort = samples()
    assert not collect(cohort[:2], -2441.5)["performance_acceptance_passed"]
    for row in cohort:
        if row["mode"] == "md-j":
            row["seconds"] = 180.0
    assert not collect(cohort, -2441.5)["performance_acceptance_passed"]


def test_first_partial_report_has_no_nan_speedup() -> None:
    result = collect(samples()[:1], -2441.5)
    assert result["normal_over_md_speedup"] is None
    assert result["median_seconds"]["md-j"] is None
