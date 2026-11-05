"""Host checks for honest selector and complete-population HF qualification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarks import hf_incremental_direct_jk as benchmark

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("strict", [False, True])
def test_explicit_arms_keep_identical_numerical_controls(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    """The measured experimental route must not masquerade as an unset default."""
    import os

    def capture_options(**options: object) -> dict:
        return options

    monkeypatch.setattr(benchmark, "Calculator", capture_options)
    monkeypatch.setenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK", "1")
    options = benchmark.calculator(True, strict)
    assert os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK"] == "1"
    assert options["screening_tolerance"] == 1e-12
    if strict:
        assert options["energy_tolerance"] == 1e-12
        assert options["density_tolerance"] == 1e-9
    else:
        assert "energy_tolerance" not in options
        assert "density_tolerance" not in options
    assert benchmark.calculator(False, strict) == options
    assert os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK"] == "0"


@pytest.fixture
def samples() -> list[dict]:
    """Use declared complete populations, not measured GPU timing in CI."""
    return [
        {
            "aos": 768,
            "phase": phase,
            "incremental": incremental,
            "seconds": 1.0 if incremental else 2.0,
            "passed": True,
            "iterations": 11,
            "fock_builds": 11,
        }
        for phase in ("cold", "warm", "moved")
        for incremental in (False, True)
        for _repeat in range(2)
    ]


def test_complete_qualified_populations(samples: list[dict]) -> None:
    """Only independently qualified, complete arms get a speedup ratio."""
    for row in benchmark.summarize(samples, ["768"], 2):
        assert row["speedup"] == pytest.approx(2.0)
        assert row["off"]["qualified"] and row["on"]["qualified"]


@pytest.mark.parametrize(
    "invalidate",
    [
        lambda samples: samples.pop(0),
        lambda samples: samples[0].update(passed=False),
    ],
    ids=["incomplete", "failed-independent-gate"],
)
def test_unqualified_population_is_not_ranked(
    samples: list[dict], invalidate: Callable[[list[dict]], object]
) -> None:
    """Retain the failed arm's latency without turning it into a speed claim."""
    invalidate(samples)
    cold = next(
        row
        for row in benchmark.summarize(samples, ["768"], 2)
        if row["phase"] == "cold"
    )
    assert not cold["off"]["qualified"]
    assert cold["off"]["median_seconds"] == 2.0
    assert cold["speedup"] is None
