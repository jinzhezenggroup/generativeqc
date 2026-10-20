"""Exercise mixed-source publication without pinning generated source spellings."""

from __future__ import annotations

import copy
import json
import lzma
import runpy

import numpy as np
import pytest

from tools.render_readme_dft import phase_statistics, work_cell
from tools.render_readme_r2scan3c import (
    PHASE_COUNTS,
    RESULTS,
    overlay_native,
    small_updates,
)


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


@pytest.fixture
def small_publication() -> tuple[dict, dict, np.ndarray]:
    """Use retained independent arrays, not a candidate-derived accuracy oracle."""
    previous = runpy.run_path(str(RESULTS / "readme-refresh-20261015/refresh.py"))
    summary = previous["combined_summary"]("summary.json.xz")
    packet = RESULTS / "xc-diagonal-potential-20261016"
    with lzma.open(packet / "evidence.json.xz", "rt") as stream:
        data = json.load(stream)
    with np.load(packet / "forces.npz", allow_pickle=False) as archive:
        byteplanes = archive["xor_byteplanes"]
    return summary, data, byteplanes


def test_small_cases_replay_every_returned_force(
    small_publication: tuple[dict, dict, np.ndarray],
) -> None:
    """All five repeats and both geometries must have independently valid forces."""
    summary, data, byteplanes = small_publication
    updates = small_updates(summary, data, byteplanes)
    assert set(updates) == {3, 12}
    overlay_native(summary, updates)
    for atoms, update in updates.items():
        for row in update["native"]["records"]:
            assert np.shape(row["forces"]) == (atoms, 3)
            assert row["energy_error"] <= 1e-8
            assert row["force_error"] <= 1e-7


@pytest.mark.parametrize("failure", ["force-bit", "energy", "protocol", "build"])
def test_small_replay_rejects_corruption_and_false_accuracy_receipts(
    small_publication: tuple[dict, dict, np.ndarray], failure: str
) -> None:
    """Collection-time success flags cannot replace independent gate replay."""
    summary, data, byteplanes = small_publication
    population = data["readme_small"]["populations"]["3"]
    if failure == "force-bit":
        column = population["records"][0]["forces_delta"]
        byteplanes[0, column, 0] ^= 1
    elif failure == "energy":
        population["records"][0]["energy"] += 1e-5
    elif failure == "protocol":
        summary["methods"]["r2scan-3c"][0]["protocol"]["grid"]["radial_points"] = 24
    else:
        population["native_build"]["source_identity"] = "wrong-source"
    with pytest.raises(ValueError):
        small_updates(summary, data, byteplanes)
