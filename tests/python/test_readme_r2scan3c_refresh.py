"""Exercise mixed-source publication without pinning generated source spellings."""

from __future__ import annotations

import copy

import pytest

from tools.render_readme_dft import phase_statistics, work_cell
from tools.render_readme_r2scan3c import PHASE_COUNTS, overlay_native


@pytest.fixture
def publication() -> tuple[dict, dict]:
    """Use distinct timings and work so selection cannot hide sample trimming."""
    points = [
        {
            "atoms": atoms,
            "protocol": {"atoms": atoms, "basis": "test", "grid": [48, 16, 32]},
            "engines": {
                "native": {"historical": atoms},
                "reference": {"historical-reference": atoms},
            },
            "independent_references": [{"energy": -atoms}],
        }
        for atoms in (3, 12, 48, 96)
    ]
    summary = {
        "density_fitting": False,
        "methods": {"r2scan-3c": points, "pbe0": copy.deepcopy(points)},
    }
    updates = {}
    for point in points[2:]:
        rows = [
            {
                "phase": phase,
                "complete_seconds": seconds,
                "iterations": 1 if "warm" in phase else 19,
                "fock_builds": 1 if "warm" in phase else 19,
                "gate": True,
                "converged": True,
                "energy_error": 1e-11,
                "force_error": 1e-10,
            }
            for phase, count in PHASE_COUNTS.items()
            for seconds in [7.0, 2.0, 9.0, 3.0, 5.0][:count]
        ]
        updates[point["atoms"]] = {
            "protocol": copy.deepcopy(point["protocol"]),
            "native": {"status": "measured", "records": rows},
        }
    return summary, updates


def test_overlay_preserves_every_other_observation(
    publication: tuple[dict, dict],
) -> None:
    """Only the two requested native engines change, without mutating evidence."""
    summary, updates = publication
    original = copy.deepcopy(summary)
    result = overlay_native(summary, updates)
    assert summary == original
    assert result["methods"]["pbe0"] == summary["methods"]["pbe0"]
    for old, new in zip(
        summary["methods"]["r2scan-3c"], result["methods"]["r2scan-3c"], strict=True
    ):
        comparison = copy.deepcopy(new)
        if old["atoms"] in updates:
            assert new["engines"]["native"] == updates[old["atoms"]]["native"]
            assert phase_statistics(new["engines"]["native"], "warm") == (5.0, 3.0, 4.0)
            assert work_cell(new["engines"]["native"], "cold") == "19/19"
            comparison["engines"]["native"] = old["engines"]["native"]
        assert comparison == old


@pytest.mark.parametrize(
    "failure", ["df", "protocol", "incomplete", "gate", "energy", "force", "time"]
)
def test_overlay_rejects_unqualified_endpoints(
    publication: tuple[dict, dict], failure: str
) -> None:
    """Missing repeats or altered scientific conditions cannot become new points."""
    summary, updates = publication
    update = updates[48]
    if failure == "df":
        summary["density_fitting"] = True
    elif failure == "protocol":
        update["protocol"]["grid"] = [24, 8, 16]
    elif failure == "incomplete":
        update["native"]["records"].pop()
    else:
        field, value = {
            "gate": ("gate", False),
            "energy": ("energy_error", 1e-7),
            "force": ("force_error", 1e-6),
            "time": ("complete_seconds", float("nan")),
        }[failure]
        update["native"]["records"][0][field] = value
    with pytest.raises(ValueError):
        overlay_native(summary, updates)
