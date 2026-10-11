"""Host gates for the diagnostic model, not GPU/endpoint performance evidence."""

from __future__ import annotations

import copy
import itertools
import json
import os
import random
import subprocess
import sys
import typing

import pytest

from benchmarks.selective_eri_cache import SCHEMA, SCOPE, select_buckets

if typing.TYPE_CHECKING:
    from pathlib import Path


def measured(first: int, second: int, units: int, gain: float) -> dict:
    """Make an explicitly synthetic one-action scenario with matched work."""
    return {
        "first": first,
        "second": second,
        "status": "measured",
        "source_values": units,
        "cache_bytes": 8 * units,
        "parity": "passed",
        "max_absolute_error": 0.0,
        "max_scaled_error": 0.0,
        "build_seconds": [1.0],
        "direct_seconds": [gain + 2.0],
        "replay_seconds": [1.0],
        "allocation_seconds": 0.0,
        "release_seconds": 0.0,
        "work": {
            "direct_candidates": units,
            "direct_evaluations": units,
            "build_candidates": units,
            "build_evaluations": units,
            "replay_candidates": units,
            "replay_evaluations": 0,
        },
    }


def receipt(*rows: dict) -> dict:
    """Keep absent angular classes in the source inventory as explicit empties."""
    selected = {(row["first"], row["second"]): row for row in rows}
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "status": "complete",
        "identity": {
            "native_sha256": "a" * 64,
            "probe_sha256": "b" * 64,
            "input_sha256": "c" * 64,
            "device_uuid": "synthetic-test-device",
            "geometry_hash": "d" * 64,
            "basis_hash": "e" * 64,
            "operator": "full-range-restricted-JK",
            "precision": "fp64",
            "screening_tolerance": 0.0,
        },
        "buckets": [
            selected.get(
                (first, second),
                {"first": first, "second": second, "status": "empty"},
            )
            for first in range(7)
            for second in range(first + 1)
        ],
    }


def test_exact_selection_not_benefit_per_byte_greedy() -> None:
    result = select_buckets(
        receipt(measured(0, 0, 6, 12), measured(1, 0, 5, 9), measured(1, 1, 5, 9)),
        80,
        1,
    )
    assert result["selected"] == [
        {"first": 1, "second": 0},
        {"first": 1, "second": 1},
    ]
    assert result["selected_bytes"] == 80
    assert result["selected_bucket_mask"] == (1 << 1) | (1 << 2)
    assert result["predicted_gain_seconds"] == 18
    assert result["optimal_for_additive_model"]
    assert result["endpoint_performance"] == "not-measured"
    assert result["production_admission"] == "not-qualified"


@pytest.mark.parametrize("actions,budget", [(0, 80), (1, 0), (1, 7)])
def test_zero_horizon_or_capacity_keeps_direct(actions: int, budget: int) -> None:
    result = select_buckets(receipt(measured(0, 0, 1, 10)), budget, actions)
    assert result["status"] == "no-profitable-cache"
    assert result["selected"] == []
    assert result["selected_bucket_mask"] == 0
    assert result["predicted_gain_seconds"] == 0


def test_allocation_release_and_shared_setup_are_charged() -> None:
    row = measured(0, 0, 1, 10)
    row.update(allocation_seconds=2.0, release_seconds=3.0)
    result = select_buckets(receipt(row), 8, 1, fixed_setup_seconds=4.0)
    assert result["predicted_gain_seconds"] == 1.0
    result = select_buckets(receipt(row), 8, 1, fixed_setup_seconds=5.0)
    assert result["selected"] == []


def test_all_repeats_and_medians_are_retained() -> None:
    row = measured(0, 0, 1, 10)
    row.update(
        build_seconds=[1.0, 100.0, 1.0],
        direct_seconds=[12.0, 50.0, 12.0],
        replay_seconds=[1.0, 90.0, 1.0],
    )
    original = copy.deepcopy(row)
    result = select_buckets(receipt(row), 8, 1)
    assert result["predicted_gain_seconds"] == 10.0
    assert row == original


def test_bounded_exact_search_refuses_instead_of_inventing_optimum() -> None:
    result = select_buckets(receipt(measured(0, 0, 1, 1)), 8, 1, maximum_states=1)
    assert result["status"] == "search-bound-refused"
    assert not result["optimal_for_additive_model"]
    assert result["selected"] == []


def test_deterministic_ties_prefer_smaller_storage_then_bucket_order() -> None:
    rows = [measured(0, 0, 2, 5), measured(1, 0, 1, 5), measured(1, 1, 1, 5)]
    result = select_buckets(receipt(*rows), 8, 1)
    assert result["selected"] == [{"first": 1, "second": 0}]
    shuffled = receipt(*rows)
    shuffled["buckets"].reverse()
    assert select_buckets(shuffled, 8, 1) == result


def test_pareto_search_matches_exhaustive_small_scenarios() -> None:
    generator = random.Random(2243)
    keys = [(first, second) for first in range(4) for second in range(first + 1)]
    for _ in range(60):
        rows = [
            measured(first, second, generator.randint(1, 10), generator.randint(-1, 20))
            for first, second in keys[:8]
        ]
        budget = generator.randint(0, 50) * 8
        optimum = 0
        for flags in itertools.product((False, True), repeat=len(rows)):
            chosen = [row for row, flag in zip(rows, flags, strict=True) if flag]
            if sum(row["cache_bytes"] for row in chosen) <= budget:
                optimum = max(
                    optimum,
                    sum(row["direct_seconds"][0] - 2 for row in chosen),
                )
        result = select_buckets(receipt(*rows), budget, 1)
        assert result["predicted_gain_seconds"] == optimum
        assert result["selected_bytes"] <= budget


@pytest.mark.parametrize("status", ["running", "failed"])
def test_incomplete_receipts_are_not_training_data(status: str) -> None:
    report = receipt(measured(0, 0, 1, 10))
    report["status"] = status
    with pytest.raises(ValueError, match="complete"):
        select_buckets(report, 8, 1)


@pytest.mark.parametrize(
    "field",
    [
        "native_sha256",
        "probe_sha256",
        "input_sha256",
        "device_uuid",
        "geometry_hash",
        "basis_hash",
    ],
)
def test_source_device_identity_is_mandatory(field: str) -> None:
    report = receipt(measured(0, 0, 1, 10))
    del report["identity"][field]
    with pytest.raises(ValueError, match="identity"):
        select_buckets(report, 8, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator", "short-range-K"),
        ("precision", "fp32"),
        ("screening_tolerance", 1e-12),
    ],
)
def test_unsupported_scientific_domains_are_not_relabelled(
    field: str, value: str | float
) -> None:
    report = receipt(measured(0, 0, 1, 10))
    report["identity"][field] = value
    with pytest.raises(ValueError, match="identity"):
        select_buckets(report, 8, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_bytes", 9),
        ("parity", "failed"),
        ("direct_seconds", [float("nan")]),
        ("replay_seconds", [-1]),
        ("allocation_seconds", float("inf")),
        ("build_seconds", []),
        ("replay_seconds", [1, 2]),
    ],
)
def test_bad_bucket_measurements_are_rejected(field: str, value: object) -> None:
    row = measured(0, 0, 1, 10)
    row[field] = value
    with pytest.raises(ValueError):
        select_buckets(receipt(row), 8, 1)


@pytest.mark.parametrize(
    "field",
    [
        "direct_candidates",
        "build_evaluations",
        "replay_candidates",
        "replay_evaluations",
    ],
)
def test_mismatched_work_is_not_credited(field: str) -> None:
    row = measured(0, 0, 1, 10)
    row["work"][field] += 1
    with pytest.raises(ValueError, match="work"):
        select_buckets(receipt(row), 8, 1)


def test_missing_and_duplicate_buckets_are_rejected() -> None:
    report = receipt(measured(0, 0, 1, 10))
    report["buckets"].pop()
    with pytest.raises(ValueError, match="28 buckets"):
        select_buckets(report, 8, 1)
    report = receipt(measured(0, 0, 1, 10))
    report["buckets"].append(report["buckets"][0])
    with pytest.raises(ValueError, match="duplicate"):
        select_buckets(report, 8, 1)


@pytest.mark.parametrize(
    "budget,actions,states", [(-1, 1, 10), (8, -1, 10), (8, True, 10), (8, 1, 0)]
)
def test_invalid_selection_resources_are_rejected(
    budget: int, actions: int, states: int
) -> None:
    with pytest.raises(ValueError):
        select_buckets(receipt(), budget, actions, maximum_states=states)


@pytest.mark.parametrize("status", ["capacity-refused", "allocation-refused"])
def test_capacity_refused_buckets_are_not_assumed_profitable(status: str) -> None:
    row = measured(0, 0, 1, 1000)
    row["status"] = status
    result = select_buckets(receipt(row), 8, 1)
    assert result["selected"] == []


def test_parity_label_cannot_hide_an_inaccurate_repeat() -> None:
    row = measured(0, 0, 1, 10)
    row["max_scaled_error"] = 1e-8
    with pytest.raises(ValueError, match="parity error"):
        select_buckets(receipt(row), 8, 1)


def test_boolean_counters_are_not_semantic_work() -> None:
    row = measured(0, 0, 1, 10)
    row["work"]["direct_candidates"] = True
    with pytest.raises(ValueError, match="work"):
        select_buckets(receipt(row), 8, 1)


def test_gain_and_timing_overflow_are_rejected() -> None:
    with pytest.raises(ValueError, match="overflow"):
        select_buckets(receipt(measured(0, 0, 1, 10)), 8, 10**1000)
    row = measured(0, 0, 1, 10)
    row["allocation_seconds"] = 10**1000
    with pytest.raises(ValueError, match="finite timing range"):
        select_buckets(receipt(row), 8, 1)


def test_cli_selector_is_native_and_gpu_independent(tmp_path: Path) -> None:
    source, output = tmp_path / "receipt.json", tmp_path / "selection.json"
    source.write_text(json.dumps(receipt(measured(0, 0, 1, 10))))
    environment = {**os.environ, "GENERATIVEQC_LIBRARY": "/not/a/native/library"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.selective_eri_cache",
            "select",
            "--input",
            str(source),
            "--output",
            str(output),
            "--cache-budget-bytes",
            "8",
            "--expected-actions",
            "1",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "selected"
    assert len(result["receipt_sha256"]) == 64


def test_profiler_refuses_before_loading_native_without_slurm(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.selective_eri_cache",
            "profile",
            "--input",
            "/missing/input",
            "--probe",
            "/missing/probe",
            "--output",
            str(tmp_path / "profile.json"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "srun" in result.stderr
    assert not (tmp_path / "profile.json").exists()
