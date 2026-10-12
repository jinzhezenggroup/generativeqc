"""Early approximate KS handoff retains independent strict physical correctness."""

from __future__ import annotations

import os
from argparse import Namespace
from functools import cache
from typing import Any

import pytest

from benchmarks.dft_mixed_thresholds import calculator, inputs, oracle_sample

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly Slurm-allocated GPU",
)


@cache
def _independent_energy(method: str, case: str = "water") -> float:
    pytest.importorskip("pyscf")
    result = oracle_sample(Namespace(method=method, case=case, precision="fp64"))
    assert result["converged"]
    return result["energy"]


def _solve(method: str, precision: str, case: str = "water") -> Any:
    geometry, _ = inputs(case)
    calc = calculator(Namespace(method=method, precision=precision), geometry, "cuda")
    with calc.prepare_batch(
        [geometry["atoms"]],
        charges=[geometry.get("charge", 0)],
        multiplicities=[geometry.get("multiplicity", 1)],
    ) as owner:
        return owner.execute(properties=("energy",), strict=True).items[0]


@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks", "r2scan-rks"])
@pytest.mark.parametrize("threshold", [1e-2, 1e-4, 1e-6])
def test_early_switch_retains_strict_oracle_and_fresh_energy_history(
    method: str, threshold: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", str(threshold))
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "4")
    result = _solve(method, "auto")
    assert result.converged
    assert result.energy == pytest.approx(_independent_energy(method), abs=1e-8)
    assert result.ks_diagnostic.physical_residual_max < 1e-9
    assert 0 < result.precision["mixed_stage_fock_builds"] <= 4
    assert result.precision["strict_stage_fock_builds"] >= 2
    assert result.precision["strict_refinement_applied"]
    assert any(
        row.iteration > 1 and row.energy_change is None
        for row in result.ks_diagnostic.history
    )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks", "r2scan-rks"])
def test_residual_can_handoff_without_an_iteration_cap(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "0")
    result = _solve(method, "auto")
    assert result.converged
    assert result.precision["mixed_stage_fock_builds"] < 100
    assert result.precision["strict_stage_fock_builds"] >= 2
    assert result.ks_diagnostic.physical_residual_max < 1e-9
    assert result.energy == pytest.approx(_independent_energy(method), abs=1e-8)


@pytest.mark.parametrize("method", ["pbe-uks", "pbe0-uks", "r2scan-uks"])
def test_unqualified_open_shell_handoff_is_rejected_explicitly(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "1e-2")
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "4")
    with pytest.raises((ValueError, RuntimeError), match="requires restricted CUDA KS"):
        _solve(method, "auto", "oh")


@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks", "r2scan-rks"])
def test_single_approximate_observation_requires_strict_refinement(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "1")
    result = _solve(method, "auto")
    assert result.converged
    assert result.precision["mixed_stage_fock_builds"] == 1
    assert result.precision["strict_refinement_applied"]
    assert result.ks_diagnostic.physical_residual_max < 1e-9
    assert result.energy == pytest.approx(_independent_energy(method), abs=1e-8)


@pytest.mark.parametrize("precision", ["fp64", "auto"])
def test_unrestricted_incumbent_precision_preparation_is_retained(
    precision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "nan" if precision == "fp64" else "0"
    )
    monkeypatch.setenv(
        "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "-1" if precision == "fp64" else "0"
    )
    geometry, _ = inputs("oh")
    calc = calculator(
        Namespace(method="pbe0-uks", precision=precision), geometry, "cuda"
    )
    with calc.prepare_batch([geometry["atoms"]], charges=[0], multiplicities=[2]):
        pass


def test_explicit_fp64_ignores_approximate_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "nan")
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "-1")
    result = _solve("pbe0-rks", "fp64")
    assert result.energy == pytest.approx(_independent_energy("pbe0-rks"), abs=1e-8)
    assert result.precision["mixed_stage_fock_builds"] == 0
    assert not result.precision["strict_refinement_applied"]


@pytest.mark.parametrize(
    "control,value",
    [
        ("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "nan"),
        ("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "-1"),
        ("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "-1"),
    ],
)
def test_invalid_approximate_controls_fail_explicitly(
    control: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(control, value)
    with pytest.raises((ValueError, RuntimeError), match="GENERATIVEQC_KS_MIXED"):
        _solve("pbe0-rks", "auto")
