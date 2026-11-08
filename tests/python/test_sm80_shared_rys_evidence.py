"""Historical E/F failures must survive replay and cannot authorize promotion."""

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "benchmarks/results/a100-shared-rys-xy-20261108/replay.py"


@pytest.fixture(scope="module")
def reader() -> dict[str, Any]:
    return runpy.run_path(str(READER))


def test_default_force_failure_is_not_hidden_by_native_parity(
    reader: dict[str, Any],
) -> None:
    summary = reader["replay"]()
    assert not summary["independent_default_promotion_qualified"]
    for card in summary["cards"].values():
        case = card["default_cases"]["48"]
        assert case["all_120_native_pairs_pass"]
        assert len(case["independent_default_observations"]) == 16
        assert all(
            not item["passed"] for item in case["independent_default_observations"]
        )
        assert all(
            item["force"] > 1e-7 for item in case["independent_default_observations"]
        )
        assert all(
            item["passed"]
            for item in card["default_cases"]["96"]["independent_default_observations"]
        )
        assert all(item["passed"] for item in card["moved_and_restored_hf96"])


def test_tight_request_is_reported_separately(reader: dict[str, Any]) -> None:
    summary = reader["replay"]()
    for card in summary["cards"].values():
        for samples in card["strict_request_hf48"].values():
            assert len(samples) == 4
            assert all(item["passed"] for item in samples)
    assert summary["paired_warm_ratio_ceiling"] == 1.03
    assert summary["independent_gates"]["maximum_force"] == 1e-7


def test_require_promotion_fails_closed(
    reader: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", [str(READER), "--require-promotion"])
    with pytest.raises(SystemExit, match="promotion remains blocked"):
        reader["main"]()


@pytest.mark.parametrize("forces", [[[float("nan"), 0, 0]], [[0, 0]]])
def test_invalid_independent_forces_cannot_pass(
    reader: dict[str, Any], forces: list[list[float]]
) -> None:
    actual = {"energy": 0.0, "forces": forces, "converged": True}
    target = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    with pytest.raises(ValueError):
        reader["observable_error"](actual, target)


def test_changed_capsule_is_rejected(reader: dict[str, Any], tmp_path: Path) -> None:
    directory = READER.parent
    (tmp_path / "cohort.json").write_bytes((directory / "cohort.json").read_bytes())
    payload = (directory / "observations.json.xz").read_bytes()
    (tmp_path / "observations.json.xz").write_bytes(payload + b"altered")
    with pytest.raises(ValueError, match="changed observation capsule"):
        reader["load"](tmp_path)
