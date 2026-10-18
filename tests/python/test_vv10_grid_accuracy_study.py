"""Grid accuracy needs stable independent refinements and every configured norm."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from generativeqc import GridSpec

from benchmarks.vv10_grid_accuracy_sweep import parse_grid_pair
from tools.analyze_vv10_grid_study import (
    PROFILES,
    analyze_case,
    compare_populations,
    geometry_rows,
    load_population,
)

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc.force_aware_numerics import TargetErrorBudget

ANCHOR = (144, 48, 96)
RADIAL = (120, 48, 96)
ANGULAR = (144, 40, 80)
CANDIDATE = (48, 16, 32)


def population(
    grid: tuple[int, int, int], *, energy: float = -76.0, force: float = 0.0
) -> dict:
    return {
        "schema": "generativeqc.reference-split-vv10.v1",
        "engine": "GPU4PySCF",
        "stage": "complete",
        "atoms": 3,
        "basis": "spherical def2-SVP",
        "density_fitting": "none",
        "geometries_bohr": [[[0, 0, 0]], [[0, 0, 0.001]]],
        "xc_grid": asdict(GridSpec(*grid)),
        "nlc_grid": asdict(GridSpec(64, 16, 32)),
        "records": [
            {
                "phase": phase,
                "geometry": geometry,
                "repeat": repeat,
                "converged": True,
                "energy": energy,
                "forces": [[force, 0, 0]] * 3,
            }
            for geometry, phase, repeat in (
                (0, "cold", 0),
                (0, "warm", 1),
                (1, "moved", 0),
                (1, "moved-warm", 1),
            )
        ],
        "scheduler": {"SLURM_JOB_ID": "grid-study"},
        "accelerator": {"name": "independent GPU"},
        "source_sha256": {"reference.py": "f" * 64},
    }


def analyze(populations: dict, budget: TargetErrorBudget | None = None) -> dict:
    return analyze_case(
        populations,
        axis="xc",
        anchor=ANCHOR,
        radial_check=RADIAL,
        angular_check=ANGULAR,
        budget=budget or PROFILES["fp64"],
    )


def test_missing_refinements_never_certify_an_apparently_matching_grid() -> None:
    result = analyze({ANCHOR: population(ANCHOR), CANDIDATE: population(CANDIDATE)})
    assert result["reference_stability_gate"] is None
    assert all(point["grid_budget_gate"] is None for point in result["comparisons"])


def test_both_axes_and_anchor_replays_must_be_stable() -> None:
    populations = {
        grid: population(grid) for grid in (ANCHOR, RADIAL, ANGULAR, CANDIDATE)
    }
    result = analyze(populations)
    assert result["reference_stability_gate"] is True
    assert (
        next(
            point for point in result["comparisons"] if point["grid"] == list(CANDIDATE)
        )["grid_budget_gate"]
        is True
    )
    baseline = next(
        point for point in result["comparisons"] if point["grid"] == list(ANCHOR)
    )
    assert baseline["grid_budget_gate"] is None
    populations[ANCHOR]["records"][1]["energy"] += 5e-9
    assert analyze(populations)["reference_stability_gate"] is False


@pytest.mark.parametrize("grid", (RADIAL, ANGULAR))
def test_unstable_anchor_blocks_grid_admission(grid: tuple[int, int, int]) -> None:
    populations = {
        shape: population(shape) for shape in (ANCHOR, RADIAL, ANGULAR, CANDIDATE)
    }
    populations[grid]["records"][2]["forces"][0][0] = 2e-8
    result = analyze(populations)
    assert result["reference_stability_gate"] is False
    assert all(point["grid_budget_gate"] is not True for point in result["comparisons"])


def test_reference_allowance_is_reserved_from_candidate_budget() -> None:
    populations = {
        grid: population(grid) for grid in (ANCHOR, RADIAL, ANGULAR, CANDIDATE)
    }
    populations[CANDIDATE]["records"][2]["energy"] += 9.5e-9
    result = analyze(populations)
    point = next(
        point for point in result["comparisons"] if point["grid"] == list(CANDIDATE)
    )
    assert point["comparison"]["observed_budget_gate"] is True
    assert point["reserved_budget_gate"] is False
    assert point["grid_budget_gate"] is False


def test_fast_grid_gate_includes_rms_and_does_not_relax_fp64() -> None:
    candidate, reference = (
        population(CANDIDATE, energy=-76.0 + 6.38e-7, force=3.23e-6),
        population(ANCHOR),
    )
    assert (
        compare_populations(candidate, reference, PROFILES["fast"])[
            "observed_budget_gate"
        ]
        is True
    )
    assert (
        compare_populations(candidate, reference, PROFILES["fp64"])[
            "observed_budget_gate"
        ]
        is False
    )
    for row in candidate["records"]:
        row["forces"] = [[5e-5] * 3] * 3
    result = compare_populations(candidate, reference, PROFILES["fast"])
    assert result["maximum_errors"]["force_max_abs"] < 1e-4
    assert result["maximum_errors"]["force_rms"] > 3e-5
    assert result["observed_budget_gate"] is False


@pytest.mark.parametrize(
    "failure", ("missing-moved", "nonfinite", "shape", "unconverged-warm", "duplicate")
)
def test_invalid_endpoints_never_become_geometry_oracles(failure: str) -> None:
    raw = population(CANDIDATE)
    if failure == "missing-moved":
        raw["records"].pop(2)
    elif failure == "nonfinite":
        raw["records"][0]["energy"] = float("nan")
    elif failure == "shape":
        raw["records"][0]["forces"] = [[0, 0, 0]]
    elif failure == "unconverged-warm":
        raw["records"][1]["converged"] = False
    else:
        raw["records"].append(copy.deepcopy(raw["records"][0]))
    with pytest.raises((TypeError, ValueError)):
        geometry_rows(raw)


def test_changed_partition_or_other_grid_is_not_an_isolated_scan() -> None:
    candidate, reference = population(CANDIDATE), population(ANCHOR)
    candidate["xc_grid"]["partition_iterations"] = 4
    with pytest.raises(ValueError, match="prescription"):
        compare_populations(candidate, reference, PROFILES["fp64"])
    populations = {
        grid: population(grid) for grid in (ANCHOR, RADIAL, ANGULAR, CANDIDATE)
    }
    populations[CANDIDATE]["nlc_grid"] = asdict(GridSpec(48, 16, 32))
    with pytest.raises(ValueError, match="changes nlc"):
        analyze(populations)


def test_process_failure_is_retained_even_with_complete_endpoint_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "point.json"
    path.write_text(json.dumps(population(CANDIDATE)))
    with pytest.raises(ValueError, match="outcome"):
        load_population(path)
    path.with_suffix(".outcome").write_text('{"exit_code":1}')
    with pytest.raises(ValueError, match="outcome"):
        load_population(path)
    path.with_suffix(".outcome").write_text('{"exit_code":0}')
    loaded = load_population(path)
    assert len(loaded["receipt"]["sha256"]) == 64


@pytest.mark.parametrize("fraction", (0.0, 1.0, -0.1, float("nan")))
def test_reference_fraction_cannot_invent_or_eliminate_a_budget(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="fraction"):
        analyze_case(
            {},
            axis="xc",
            anchor=ANCHOR,
            radial_check=RADIAL,
            angular_check=ANGULAR,
            budget=PROFILES["fp64"],
            anchor_fraction=fraction,
        )


def test_accuracy_only_cross_grid_seeds_cannot_be_relabelled_as_cold(
    tmp_path: Path,
) -> None:
    raw = population(CANDIDATE)
    raw["schema"] = "generativeqc.reference-vv10-grid-accuracy.v1"
    raw["purpose"] = "accuracy-only-not-a-latency-population"
    raw["accuracy_protocol"] = {
        "density_sources": "independent-GPU4PySCF-only",
        "cold_timing_qualified": False,
    }
    raw["records"] = [raw["records"][0], raw["records"][2]]
    for row, phase in zip(
        raw["records"], ("accuracy-original", "accuracy-moved"), strict=True
    ):
        row["phase"] = phase
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps(raw))
    path.with_suffix(".outcome").write_text('{"exit_code":0}')
    loaded = load_population(path)
    assert set(geometry_rows(loaded)) == {0, 1}
    raw["accuracy_protocol"]["cold_timing_qualified"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cannot claim cold timing"):
        load_population(path)
    raw["accuracy_protocol"]["cold_timing_qualified"] = False
    raw["accuracy_protocol"]["density_sources"] = "native-density"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="native density"):
        load_population(path)


@pytest.mark.parametrize(
    "pair", ("48,16,32", "48,16,32:64,16", "48,16,32:64,16,32:96,32,64")
)
def test_accuracy_sweep_rejects_ambiguous_grid_domains(pair: str) -> None:
    with pytest.raises(ValueError):
        parse_grid_pair(pair)


def test_accuracy_sweep_preserves_distinct_xc_and_nlc_shapes() -> None:
    assert parse_grid_pair("144,48,96:64,16,32") == ((144, 48, 96), (64, 16, 32))
