"""Missing numerical data must never qualify an incremental Direct-J/K run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools import reduce_issue990_incremental_direct_jk as reducer


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (None, None),
        ([], []),
        ([[]], [[]]),
        ([None], [None]),
        ([1.0], 1.0),
        ([1.0], [1.0, 2.0]),
        (True, True),
        ("1.0", "1.0"),
        (float("nan"), 1.0),
        (float("inf"), float("inf")),
        (1.0e308, -1.0e308),
    ],
)
def test_missing_or_invalid_values_are_not_zero_error(
    first: object, second: object
) -> None:
    with pytest.raises(ValueError):
        reducer._maximum_abs_difference(first, second)


def test_finite_nested_values_preserve_exact_comparison() -> None:
    assert reducer._maximum_abs_difference([[0.0, -1.0]], [[0.0, -0.75]]) == 0.25
    assert reducer._maximum_abs_difference([[[0.0, 0.0, 0.0]]], [[[0, 0, 0]]]) == 0


@pytest.mark.parametrize(
    "value", [None, True, "1", 0.0, -1.0, float("nan"), float("inf")]
)
def test_timing_must_be_measured_and_positive(value: object) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        reducer._positive_seconds(value)


def _records(directory: Path) -> None:
    work = {
        "active": True,
        "quartet_work_counters_valid": True,
        "anchor_full_builds": 1,
        "delta_builds": 1,
        "full_admitted_shell_quartets": 2,
        "delta_admitted_shell_quartets": 1,
        "full_admitted_quartet_tiles": 2,
        "delta_admitted_quartet_tiles": 1,
    }
    for mode in ("baseline", "incremental"):
        (directory / mode).mkdir(parents=True)
        for atoms, aos in reducer.SIZES:
            record = {
                "benchmark": "compare_gpu4pyscf_batch",
                "workload": {"ao_count": aos, "batch_size": 1},
                "native_build": {"library_sha256": "a" * 64},
                "generativeqc": {
                    "warm_median_seconds": 2.0 if mode == "baseline" else 1.0,
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [[[0.0, 0.0, 0.0]] * atoms],
                    "warm_samples": [
                        {
                            "convergence": [
                                {"iterations": 2, "incremental_direct_jk": work}
                            ]
                        }
                    ],
                },
            }
            (directory / mode / f"direct-{atoms}.json").write_text(json.dumps(record))


@pytest.mark.parametrize("missing", [None, []])
def test_cli_cannot_publish_pass_without_force_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: object
) -> None:
    _records(tmp_path)
    for mode in ("baseline", "incremental"):
        path = tmp_path / mode / "direct-12.json"
        record = json.loads(path.read_text())
        record["generativeqc"]["forces_hartree_per_bohr"] = missing
        path.write_text(json.dumps(record))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["reducer", "--input", str(tmp_path), "--output", str(output)]
    )
    with pytest.raises(ValueError):
        reducer.main()
    assert not output.exists()


def test_cli_valid_records_still_qualify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records(tmp_path)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["reducer", "--input", str(tmp_path), "--output", str(output)]
    )
    reducer.main()
    result = json.loads(output.read_text())
    assert result["all_numerical_gates_passed"]
    assert result["all_speed_gates_passed"]
    assert result["all_iteration_branches_match"]
    assert len(result["rows"]) == 4
