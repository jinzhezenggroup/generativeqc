"""One-factor evidence must retain failures and the public/internal coupling."""

from __future__ import annotations

import runpy
import shutil
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


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz"))
def test_changed_capsule_fails_closed(
    reader: dict[str, Any], tmp_path: Path, stream: str
) -> None:
    target = tmp_path / READER.parent.name
    shutil.copytree(READER.parent, target)
    shared = READER.parent.parent / "hf-screened-handoff-rejections-20261108"
    storage = tmp_path / shared.name
    shutil.copytree(shared, storage)
    payload = storage / stream
    payload.write_bytes(payload.read_bytes() + b"altered")
    with pytest.raises(ValueError, match=stream):
        reader["load"](target)


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


@pytest.fixture(scope="module")
def stationarity_reader() -> dict[str, Any]:
    """Load the separately measured default-control CUDA force repair."""
    return runpy.run_path(
        str(ROOT / "benchmarks/results/hf-force-stationarity-20261109/replay.py")
    )


def test_strict_force_owner_retains_default_failure_and_independent_acceptance(
    stationarity_reader: dict[str, Any],
) -> None:
    summary = stationarity_reader["replay"]()
    assert not summary["arms"]["baseline"]["all_passed"]
    for arm in ("candidate-before-format", "candidate"):
        assert summary["arms"][arm]["all_passed"]
        assert all(
            sample["maximum_force"] < 1e-7 and sample["energy"] < 1e-8
            for sample in summary["arms"][arm]["samples"]
        )


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz", "sha256.txt"))
def test_strict_force_capsule_corruption_fails_closed(
    stationarity_reader: dict[str, Any], tmp_path: Path, stream: str
) -> None:
    source = ROOT / "benchmarks/results/hf-force-stationarity-20261109"
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    payload = destination / stream
    payload.write_bytes(payload.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="changed force capsule|" + stream):
        stationarity_reader["load"](destination)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_strict_force_nonfinite_errors_cannot_pass(
    stationarity_reader: dict[str, Any], value: float
) -> None:
    target = {
        "converged": True,
        "energy": 0.0,
        "forces": [[0.0] * 3 for _ in range(48)],
    }
    actual = {**target, "forces": [[value, 0.0, 0.0] for _ in range(48)]}
    with pytest.raises(ValueError, match="nonfinite force"):
        stationarity_reader["observable_error"](actual, target)
