"""One-factor evidence must retain failures and the public/internal coupling."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "benchmarks/results/hf48-control-isolation-20261108/replay.py"


@pytest.fixture(scope="module")
def reader() -> dict[str, Any]:
    """Execute the independent standard-library evidence reader."""
    return runpy.run_path(str(READER))


def test_screening_does_not_fix_default_force_failure(reader: dict[str, Any]) -> None:
    summary = reader["replay"]()
    assert not summary["current_head_qualification"]
    assert not summary["default_promotion_qualified"]
    for card in summary["cards"].values():
        for condition in ("default", "screening"):
            assert all(
                not error["passed"] for error in card[condition]["independent_errors"]
            )
        for condition in ("energy", "density"):
            assert all(
                error["passed"] for error in card[condition]["independent_errors"]
            )


def test_only_one_public_control_changes_and_internal_gate_is_disclosed(
    reader: dict[str, Any],
) -> None:
    cohort, raw = reader["load"]()
    for condition, expected in cohort["conditions"].items():
        assert len(expected["explicit_controls"]) == (condition != "default")
    assert all(
        "internal_gate_coupling" in text
        for name, text in raw.items()
        if name.endswith("/default.json")
    )


def test_changed_capsule_fails_closed(reader: dict[str, Any], tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_bytes((READER.parent / "cohort.json").read_bytes())
    payload = (READER.parent / "observations.json.xz").read_bytes()
    (tmp_path / "observations.json.xz").write_bytes(payload + b"altered")
    with pytest.raises(ValueError, match="changed observation capsule"):
        reader["load"](tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_forces_cannot_pass(reader: dict[str, Any], value: float) -> None:
    target = {
        "converged": True,
        "energy": 0.0,
        "forces": [[0.0] * 3 for _ in range(48)],
    }
    actual = {**target, "forces": [[value, 0.0, 0.0] for _ in range(48)]}
    with pytest.raises(ValueError, match="nonfinite force"):
        reader["observable_error"](actual, target)
