"""Parameter isolation and independent mathematical checks for refined fitting."""

import copy

import pytest

from tools.cuda_timing_refined import _isotonic, _launch_fit, fit_refined, score_refined


def _samples() -> dict:
    """Independent exact max-plus oracle with no crossover, five occupancy knots."""
    rows = []
    for split, grids in (("train", (1, 2, 4, 8, 24)), ("holdout", (3, 6, 20))):
        for grid in grids:
            occupancy = min(1, grid / 24)
            balance = grid / (4 * ((grid + 3) // 4))
            memory_scale = min(1, occupancy / 0.25)
            for family, intensities in (
                ("fma", (1024,)),
                ("copy", (0,)),
                ("mixed", (2, 8, 32, 64)),
            ):
                for intensity in intensities:
                    traffic = 1_000_000 if family != "fma" else 0
                    operations = (
                        1_000_000 if family == "fma" else traffic / 8 * intensity
                    )
                    seconds = max(
                        operations / 1e9 / balance, traffic / 1e8 / memory_scale
                    )
                    wall = 10 * (seconds + 2e-6) + 8e-6
                    rows.append(
                        {
                            "id": f"{split}-{family}-{grid}-{intensity}",
                            "split": split,
                            "family": family,
                            "grid_blocks": grid,
                            "block_threads": 256,
                            "launch_count": 10,
                            "operations_per_launch": int(operations),
                            "bytes_per_launch": traffic,
                            "iterations": intensity,
                            "local_bytes": 0,
                            "shared_bytes": 0,
                            "registers_per_thread": 16,
                            "max_absolute_error": 0.0,
                            "wall_seconds": [wall] * 3,
                        }
                    )
    for split, counts in (("train", (1, 4, 16, 256, 1024)), ("holdout", (2, 8, 128))):
        for count in counts:
            rows.append(
                {
                    "id": f"{split}-launch-{count}",
                    "split": split,
                    "family": "launch",
                    "launch_count": count,
                    "grid_blocks": 1,
                    "block_threads": 128,
                    "operations_per_launch": 0,
                    "bytes_per_launch": 0,
                    "local_bytes": 0,
                    "shared_bytes": 0,
                    "registers_per_thread": 4,
                    "max_absolute_error": 0.0,
                    "wall_seconds": [count * 2e-6 + 8e-6] * 3,
                }
            )
    return {
        "schema": "generativeqc.cuda-timing-probe.v2",
        "cases": rows,
        "sm_count": 4,
        "architecture": "sm_120",
        "device": "synthetic",
        "max_threads_per_sm": 1536,
    }


def test_isotonic_pooling_preserves_samples_without_selecting_fastest() -> None:
    assert _isotonic([1, 4, 2, 5]) == [1, 3, 3, 5]


def test_launch_fit_separates_fixed_cost_from_per_launch_slope() -> None:
    rows = [row for row in _samples()["cases"] if row["family"] == "launch"]
    slope, fixed = _launch_fit(rows)
    assert slope == pytest.approx(2e-6)
    assert fixed == pytest.approx(8e-6)


def test_refined_fit_uses_only_training_and_reports_changed_holdout_failure() -> None:
    measurement = _samples()
    altered = copy.deepcopy(measurement)
    for row in altered["cases"]:
        if row["split"] == "holdout":
            row["wall_seconds"] = [value * 10 for value in row["wall_seconds"]]
    before = fit_refined(measurement, "synthetic")
    after = fit_refined(altered, "synthetic")
    assert before == after
    assert before["arithmetic"].effective_compute_ops_per_second == pytest.approx(1e9)
    assert before["copy"].effective_memory_bytes_per_second == pytest.approx(1e8)
    assert before["copy"].launch_seconds == pytest.approx(2e-6)
    assert before["copy"].batch_seconds == pytest.approx(8e-6)
    assert not score_refined(altered, after, "synthetic")["qualified"]


def test_refined_training_only_is_never_reported_as_qualified() -> None:
    measurement = _samples()
    measurement["cases"] = [
        row for row in measurement["cases"] if row["split"] == "train"
    ]
    profiles = fit_refined(measurement, "synthetic")
    report = score_refined(measurement, profiles, "synthetic")
    assert not report["qualified"]
    assert report["holdout"] is None
