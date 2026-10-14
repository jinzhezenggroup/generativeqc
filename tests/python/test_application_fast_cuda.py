"""Scheduled public Fast endpoints retain independent science and seed policy."""

from __future__ import annotations

import ctypes
import os
from functools import cache
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    InitialGuessSpec,
    KsOptions,
    ObservableTarget,
    PrecisionCalculator,
    PrecisionPolicy,
    TargetAccuracy,
    TargetErrorBudget,
    _native,
)
from generativeqc import ks as ks_implementation

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly Slurm-allocated GPU",
)

CASE = "water-def2-svp-spherical"
GRID = GridSpec(48, 16, 32)


def _options(**overrides: Any) -> dict:
    """Match the independent target, including the original strict tolerances."""
    case = benchmark_cases()[CASE]
    return {
        "method": "pbe0-rks",
        "basis": case.generativeqc_basis,
        "basis_representation": case.basis_representation,
        "device": "cuda",
        "ks_options": KsOptions(grid=GRID),
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
        "max_iterations": 180,
        **overrides,
    }


@cache
def _oracle() -> dict:
    """Independent CPU SCF and moving-grid analytic forces, never a native seed."""
    pytest.importorskip("pyscf")
    return reference(CASE, GRID, 1)


def _assert_qualified(result: Any) -> None:
    report = result.precision_validation
    assert result.converged and result.executed_backend == "cuda"
    assert report.status == "qualified_mixed_gradient"
    assert report.solve_count == 1 and report.target is None
    assert report.errors == () and report.budget is not None
    assert report.scope == "qualification_not_per_call_observable_audit"
    assert report.gradient_admission == "pbe0-rks-budget-fp32-v1"
    work = result.precision
    assert work["gradient"]["returned_mode"] == "mixed"
    assert work["gradient"]["attempted_fp32_geometry_points"] > 0
    assert work["gradient"]["attempted_fp32_grid_points"] > 0
    assert work["gradient"]["fp64_gradient_retries"] == 0
    assert work["gradient"]["weighted_fp32_orders"] == list(range(6))
    assert work["complete"] and work["operator_inventory_complete"]
    assert 0 < work["mixed_stage_fock_builds"] <= 12
    assert work["strict_stage_fock_builds"] >= 2
    assert work["final_residual_audits"] > 0
    assert work["strict_refinement_applied"]
    assert any(
        row["name"] == "xc"
        and row["compute"] == "fp32"
        and row["accumulation"] == "fp32"
        and row["count"] > 0
        for row in work["operators"]
    )


@pytest.mark.parametrize("seed", ("automatic", "minao", "huckel", "hcore"))
def test_public_fast_preserves_seed_and_independent_oracle(seed: str) -> None:
    overrides = (
        {}
        if seed == "automatic"
        else {"initial_guess": None if seed == "hcore" else InitialGuessSpec(seed)}
    )
    calculator = PrecisionCalculator(policy="fast", **_options(**overrides))
    strict = Calculator(precision="fp64", **_options(**overrides))
    assert calculator._candidate._initial_guess == strict._initial_guess
    atoms = benchmark_cases()[CASE].atoms
    try:
        observed = calculator.singlepoint(atoms)
        _assert_qualified(observed)
        oracle = _oracle()
        assert abs(observed.energy - oracle["energy"]) < 1e-7
        difference = np.asarray(observed.forces) - oracle["forces"]
        assert np.max(np.abs(difference)) < 1e-5
        assert np.sqrt(np.mean(difference**2)) < 3e-6
        if seed != "hcore":
            diagnostic = observed.initial_guess
            assert diagnostic["outcome"] == "used"
            if calculator._candidate._initial_guess.kind == "minao":
                assert diagnostic["preliminary_fock_builds"] == 0
            else:
                assert diagnostic["preliminary_fock_builds"] > 0
            assert diagnostic["preparation_numeric_capacity"] > 0
        else:
            assert observed.initial_guess is None
    finally:
        calculator.clear_cache()
        strict.clear_cache()


def test_tight_budget_isolated_from_fast_and_retained_geometry() -> None:
    atoms = benchmark_cases()[CASE].atoms
    controls = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("GENERATIVEQC_")
    }
    fast = PrecisionCalculator(policy="fast", **_options())
    tight = PrecisionCalculator(
        policy=PrecisionPolicy("fast", budget=TargetErrorBudget(1e-8, 1e-7, 3e-8)),
        **_options(),
    )
    strict = Calculator(precision="fp64", **_options())
    try:
        original = fast.singlepoint(atoms)
        _assert_qualified(original)
        audited = tight.singlepoint(atoms)
        assert audited.precision_validation.status == "observed_met"
        assert audited.precision_validation.solve_count == 2
        assert audited.precision["mixed_stage_fock_builds"] == 0
        assert all(error.met for error in audited.precision_validation.errors)
        replay = fast.singlepoint(atoms)
        _assert_qualified(replay)
        assert abs(replay.energy - original.energy) < 1e-10
        moved = list(atoms)
        symbol, position = moved[-1]
        moved[-1] = (symbol, (position[0], position[1], position[2] + 0.02))
        changed = fast.singlepoint(moved)
        expected = strict.singlepoint(moved)
        _assert_qualified(changed)
        assert (
            changed.precision_validation.model_identity
            != original.precision_validation.model_identity
        )
        assert abs(changed.energy - expected.energy) < 1e-8
        np.testing.assert_allclose(changed.forces, expected.forces, atol=1e-5, rtol=0)
        fast.clear_cache()
        rebuilt = fast.singlepoint(atoms)
        _assert_qualified(rebuilt)
        assert abs(rebuilt.energy - original.energy) < 1e-10
        assert controls == {
            key: value
            for key, value in os.environ.items()
            if key.startswith("GENERATIVEQC_")
        }
    finally:
        fast.clear_cache()
        tight.clear_cache()
        strict.clear_cache()


def test_explicit_target_keeps_complete_audit() -> None:
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-7),)
    )
    calculator = PrecisionCalculator(
        policy=PrecisionPolicy("fast", target=target), **_options()
    )
    try:
        result = calculator.singlepoint(
            benchmark_cases()[CASE].atoms, properties=("energy",)
        )
        assert result.precision_validation.target == target
        assert result.precision_validation.status == "observed_met"
        assert result.precision_validation.solve_count == 2
        assert result.precision_validation.errors[0].met
        assert result.precision["mixed_stage_fock_builds"] > 0
    finally:
        calculator.clear_cache()


def test_energy_replay_cannot_inherit_gradient_provenance() -> None:
    calculator = PrecisionCalculator(policy="fast", **_options())
    try:
        atoms = benchmark_cases()[CASE].atoms
        _assert_qualified(calculator.singlepoint(atoms))
        energy = calculator.singlepoint(atoms, properties=("energy",))
        assert "gradient" not in energy.precision
        assert energy.precision_validation.status == "qualified_strict_final"
        _assert_qualified(calculator.singlepoint(atoms))
    finally:
        calculator.clear_cache()


def test_gradient_precision_transport_is_bounded_and_execution_owned() -> None:
    calculator = PrecisionCalculator(policy="fast", **_options(initial_guess=None))
    try:
        candidate = calculator._candidate
        getter = candidate._library.generativeqc_batch_gradient_precision_v1
        output = (ctypes.c_uint64 * 6)(*([123] * 6))
        with candidate.prepare_batch([benchmark_cases()[CASE].atoms]) as batch:
            assert getter(batch._batch, 0, output, 4) == _native.STATUS_INVALID_ARGUMENT
            assert getter(batch._batch, 1, output, 5) == _native.STATUS_INVALID_ARGUMENT
            assert (
                getter(batch._batch, 0, output, 5)
                == _native.STATUS_PRECISION_UNAVAILABLE
            )
            assert list(output) == [123] * 6
            batch.execute(properties=("energy", "forces"), strict=True)
            assert getter(batch._batch, 0, output, 5) == _native.STATUS_SUCCESS
            assert output[0] == output[1] == 1 and output[2] > 0 and output[3] > 0
            assert output[4] == 0 and output[5] == 123
            batch.execute(properties=("energy",), strict=True)
            assert (
                getter(batch._batch, 0, output, 5)
                == _native.STATUS_PRECISION_UNAVAILABLE
            )
    finally:
        calculator.clear_cache()


def test_minimum_force_budget_falls_back_without_optional_fp32_grid_storage() -> None:
    """A strict-sized owner still computes forces without exceeding its bound."""
    calculator = PrecisionCalculator(policy="fast", **_options(initial_guess=None))
    try:
        candidate = calculator._candidate
        query = candidate._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
        limit = candidate._library.generativeqc_ks_batch_stationary_cuda_limit_v1
        with candidate.prepare_batch([benchmark_cases()[CASE].atoms]) as batch:
            receipt = (ctypes.c_uint64 * 9)()
            assert (
                query(batch._batch, 0, receipt, len(receipt)) == _native.STATUS_SUCCESS
            )
            assert receipt[5] > receipt[4]
            assert (
                limit(batch._batch, 0, receipt[4] - 1) == _native.STATUS_OUT_OF_MEMORY
            )
            assert limit(batch._batch, 0, receipt[4]) == _native.STATUS_SUCCESS
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            work = result.precision["gradient"]
            assert work["complete"] and work["attempted_fp32_geometry_points"] > 0
            assert work["attempted_fp32_grid_points"] == 0
            assert work["fp64_gradient_retries"] == 0
            np.testing.assert_allclose(
                result.forces, _oracle()["forces"], atol=1e-5, rtol=0
            )
    finally:
        calculator.clear_cache()


def test_explicit_force_target_rejects_mixed_and_verifies_fp64_fallback() -> None:
    """An actual observable target, unlike qualification, gates every endpoint."""
    target = TargetAccuracy(
        (
            ObservableTarget("energy", "absolute", "Eh", absolute=1e-7),
            ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-8),
        )
    )
    calculator = PrecisionCalculator(
        policy=PrecisionPolicy("fast", target=target), **_options()
    )
    try:
        result = calculator.singlepoint(benchmark_cases()[CASE].atoms)
        report = result.precision_validation
        assert report.status == "observed_met" and report.solve_count == 3
        assert report.gradient_admission == "fp64-after-observable-budget-rejection"
        assert "candidate_exceeded_observable_budget" in report.fallback_reasons
        assert any(not error.met for error in report.rejected_errors)
        assert all(error.met for error in report.errors)
        assert "gradient" not in result.precision
    finally:
        calculator.clear_cache()


def test_small_domain_uses_explicit_fp64_fallback() -> None:
    calculator = PrecisionCalculator(policy="fast", **_options(basis="sto-3g"))
    try:
        result = calculator.singlepoint(benchmark_cases()[CASE].atoms)
        assert result.precision_validation.status == "observed_met"
        assert result.precision_validation.solve_count == 2
        assert result.precision["mixed_stage_fock_builds"] == 0
        assert (
            "budget_or_domain_requires_fp64_audit"
            in result.precision_validation.fallback_reasons
        )
    finally:
        calculator.clear_cache()


def test_ordinary_auto_does_not_acquire_application_seed_admission() -> None:
    calculator = Calculator(precision="auto", **_options())
    assert calculator._initial_guess is None
    with pytest.raises(NotImplementedError, match="MINAO requires"):
        Calculator(
            precision="auto", **_options(initial_guess=InitialGuessSpec("minao"))
        )


@pytest.mark.parametrize(
    "malformation,detail",
    (
        ("truncated", "incomplete application precision"),
        ("nonfinite", "finite and nonnegative"),
        ("missing_energy", "requires AUTO and an energy allowance"),
    ),
)
def test_native_rejects_malformed_private_transport(
    malformation: str, detail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise native validation, bypassing Python's intentional input guard."""
    marshal = ks_implementation.native_ks_options

    def malformed(options: KsOptions) -> Any:
        descriptor = marshal(options)
        if options._precision_budget is None:
            return descriptor
        if malformation == "truncated":
            descriptor.struct_size = ctypes.sizeof(_native.KsOptionsDescriptor) + 8
        elif malformation == "nonfinite":
            descriptor._extension_owner.application_energy_budget = float("nan")
        else:
            descriptor._extension_owner.application_energy_budget = 0.0
        return descriptor

    calculator = PrecisionCalculator(policy="fast", **_options(initial_guess=None))
    monkeypatch.setattr(ks_implementation, "native_ks_options", malformed)
    try:
        with pytest.raises(RuntimeError, match=detail):
            calculator.singlepoint(benchmark_cases()[CASE].atoms)
    finally:
        calculator.clear_cache()
