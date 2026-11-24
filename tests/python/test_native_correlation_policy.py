"""Shared correlated policy transport and production-consumer requirements."""

from __future__ import annotations

import ctypes
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import CBSBasis, CCSDTCBSCalculator, TargetErrorBudget, _native
from generativeqc import _correlation_convergence as policy
from generativeqc._cbs_plan import native_cbs_plan
from generativeqc._model_resolution import snapshot_basis

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual adapter and shared policy, with no test arithmetic."""
    directory = tmp_path_factory.mktemp("correlation-policy")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_correlation_convergence.cpp"],
        directory / "policy.so",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    result = ctypes.CDLL(str(path))
    result.generativeqc_status_message = lambda status: b"policy rejected"
    return result


def reference_policy(
    budget: TargetErrorBudget | None,
) -> tuple[float, dict[str, float]]:
    """Freeze the pre-cutover metadata policy, not an observable-error oracle."""
    scale = (
        0.0
        if budget is None
        else min(
            1.0,
            budget.energy_abs / 1e-5,
            budget.force_max_abs / 1e-4,
            budget.force_rms / 3e-5 if budget.force_rms is not None else 1.0,
        )
    )
    if scale < 1e-3:
        scale = 0.0
    return scale, {
        "energy_tolerance": max(1e-11, scale * 1e-8),
        "density_tolerance": max(1e-11, scale * 1e-9),
        "ccsd_energy_tolerance": 1e-8,
        "ccsd_residual_tolerance": max(1e-9, scale * 1e-6),
        "lambda_residual_tolerance": max(1e-9, scale * 1e-7),
        "response_absolute_tolerance": max(1e-12, scale * 1e-8),
        "orbital_residual_tolerance": max(1e-10, scale * 1e-7),
        "stationarity_tolerance": max(1e-8, scale * 1e-6),
    }


def test_public_sdk_contract(required_native_cxx: Any, tmp_path: Path) -> None:
    """Exercise public layouts, atomic failures, overlap and stopping boundaries."""
    executable = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_correlation_policy.cpp",
            ROOT / "src/api/c_api_correlation_convergence.cpp",
        ],
        tmp_path / "policy",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_policy_preserves_legacy_limits(library: Any) -> None:
    """Compare randomized budgets and exact bracket neighbors with the old policy."""
    rng = random.Random(2227)
    budgets = [None, TargetErrorBudget(1e-5, 1e-4, None)]
    for value in (math.nextafter(1e-8, 0), 1e-8, math.nextafter(1e-8, math.inf)):
        budgets.append(TargetErrorBudget(value, 1e-4, None))
    budgets.extend(
        TargetErrorBudget(
            10 ** rng.uniform(-14, 2),
            10 ** rng.uniform(-14, 2),
            None if index % 3 == 0 else 10 ** rng.uniform(-14, 2),
        )
        for index in range(600)
    )
    expected = [reference_policy(budget) for budget in budgets]
    with ThreadPoolExecutor(max_workers=8) as workers:
        actual = list(
            workers.map(
                lambda b: policy.convergence_policy(b, library=library), budgets
            )
        )
    assert actual == expected


def test_cbs_policy_preserves_legacy_allocation(library: Any) -> None:
    """Freeze allocation and all stop limits, including midpoint rounding."""
    rng = random.Random(2227)
    inputs = [
        (2, 3, 3.0, 1.63),
        # A single compensated sum rounds this low-basis case one ULP low.
        (28, 52, 7.860377502539633e-164, 6.250054175082996e-155),
    ]
    for _ in range(600):
        low = rng.randrange(2, 100)
        inputs.append(
            (
                low,
                rng.randrange(low + 1, 101),
                10 ** rng.uniform(-285, 3),
                10 ** rng.uniform(-285, 3),
            )
        )
    default = TargetErrorBudget(1e-5, 1e-4, 3e-5)
    explicit = TargetErrorBudget(1e-16, 2e-4)
    for index, values in enumerate(inputs):
        plan = native_cbs_plan(
            library, *values, "highest" if index % 3 == 2 else "exponential"
        )
        for role in range(3):
            coefficients = list(plan.coefficients)
            if role < 2:
                coefficients[2 + role] += coefficients.pop()
            amplification = math.fsum(map(abs, coefficients))
            for requested, standard in ((None, True), (explicit, True), (None, False)):
                expected_total = requested or (default if standard else None)
                expected_component = None
                if expected_total is not None:
                    rms = expected_total.force_rms
                    expected_component = TargetErrorBudget(
                        expected_total.energy_abs / amplification,
                        expected_total.force_max_abs / amplification,
                        rms / amplification if rms is not None else None,
                    )
                result = policy.cbs_convergence_policy(
                    library, plan, role, requested, default_standard=standard
                )
                assert result.coefficient_l1 == amplification
                assert result.total_budget == expected_total
                assert result.component_budget == expected_component
                assert (result.scale, result.limits) == reference_policy(
                    expected_component
                )


@pytest.mark.parametrize(
    "mode",
    (
        "invalid_role",
        "invalid_default",
        "zero_energy",
        "negative_rms",
        "infinite_budget",
        "underflow_energy",
        "underflow_rms",
        "inconsistent_plan",
        "overflow_l1",
        "abi_mismatch",
    ),
)
def test_cbs_policy_rejects_without_publication(library: Any, mode: str) -> None:
    """Check failures at the public ABI, before any receipt fields change."""
    query = library.generativeqc_cbs_convergence_query_v1
    query.argtypes = [
        ctypes.POINTER(policy._CBSDescriptor),
        ctypes.POINTER(policy._CBSResult),
    ]
    query.restype = ctypes.c_int
    extrapolate = library.generativeqc_cbs_extrapolation_query_v1
    from generativeqc._cbs_plan import _Descriptor, _Plan

    extrapolate.argtypes = [ctypes.POINTER(_Descriptor), ctypes.POINTER(_Plan)]
    extrapolate.restype = ctypes.c_int
    inputs = _Descriptor(
        ctypes.sizeof(_Descriptor), _native.ABI_VERSION, 2, 3, 0, 0, 3.0, 1.63
    )
    plan = _Plan(ctypes.sizeof(_Plan), _native.ABI_VERSION)
    assert (
        extrapolate(ctypes.byref(inputs), ctypes.byref(plan)) == _native.STATUS_SUCCESS
    )
    budget = policy._encode_budget(TargetErrorBudget(1e-5, 1e-4, 3e-5))
    descriptor = policy._CBSDescriptor(
        ctypes.sizeof(policy._CBSDescriptor),
        _native.ABI_VERSION,
        0,
        1,
        ctypes.pointer(plan),
        ctypes.pointer(budget),
    )
    result = policy._CBSResult(
        ctypes.sizeof(policy._CBSResult), _native.ABI_VERSION, -7.0
    )
    expected = _native.STATUS_INVALID_ARGUMENT
    if mode == "invalid_role":
        descriptor.correction_basis = 3
    elif mode == "invalid_default":
        descriptor.default_standard = 2
    elif mode == "zero_energy":
        budget.energy_abs = 0
    elif mode == "negative_rms":
        budget.force_rms = -math.ulp(0.0)
    elif mode == "infinite_budget":
        budget.force_max_abs = math.inf
    elif mode == "underflow_energy":
        budget.energy_abs = math.ulp(0.0)
    elif mode == "underflow_rms":
        budget.force_rms = math.ulp(0.0)
    elif mode == "inconsistent_plan":
        plan.coefficients[0] = 0
    elif mode == "overflow_l1":
        inputs.correlation_exponent = 3e-308
        assert (
            extrapolate(ctypes.byref(inputs), ctypes.byref(plan))
            == _native.STATUS_SUCCESS
        )
    elif mode == "abi_mismatch":
        plan.abi_version += 1
        expected = _native.STATUS_ABI_MISMATCH
    before = bytes(result)
    assert query(ctypes.byref(descriptor), ctypes.byref(result)) == expected
    assert bytes(result) == before


def test_cbs_production_planning_uses_native(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual facade planning retains explicit tighter settings and CBS ceilings."""
    basis = snapshot_basis("sto-3g")
    arguments = ((CBSBasis(basis, 2), CBSBasis(snapshot_basis("def2-svp"), 3)),)
    options = {"energy_tolerance": 1e-13, "ccsd_residual_tolerance": 1e-10}
    monkeypatch.setattr(_native, "load_library", lambda **options: library)
    actual = CCSDTCBSCalculator(*arguments, calculator_options=options)._convergence
    budget = TargetErrorBudget(**actual["component_budget"])
    expected_scale, expected_limits = reference_policy(budget)
    assert actual["scale"] == expected_scale
    assert (
        actual["limits"]["lambda_residual_tolerance"]
        == expected_limits["lambda_residual_tolerance"]
    )
    assert actual["limits"]["energy_tolerance"] == 1e-13
    assert actual["limits"]["ccsd_residual_tolerance"] == 1e-10
    assert actual["limits"]["density_tolerance"] <= 1e-11
    assert CCSDTCBSCalculator(*arguments, policy="tight")._convergence["scale"] == 0


def test_selected_library_never_discovers_another(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calculator-owned transport must use its own library even if discovery differs."""

    def forbidden() -> None:
        raise AssertionError("selected library escaped to default discovery")

    monkeypatch.setattr(_native, "load_library", forbidden)
    budget = TargetErrorBudget(1e-5, 1e-4, 3e-5)
    expected = reference_policy(budget)
    assert policy.convergence_policy(budget, library=library) == expected
    with pytest.raises(
        NotImplementedError, match="required correlated-convergence query"
    ):
        policy.convergence_policy(budget, library=SimpleNamespace())


@pytest.mark.parametrize(
    "mode", ("rejection", "bad_header", "nan_scale", "nan_limit", "zero_limit")
)
def test_admitted_errors_never_retry_python(mode: str) -> None:
    """Native refusal and invalid publication remain visible at the call boundary."""

    def operation(descriptor: Any, output: Any) -> int:
        if mode == "rejection":
            return _native.STATUS_INVALID_ARGUMENT
        result = ctypes.cast(output, ctypes.POINTER(policy._Limits)).contents
        result.scale = 0
        for name in policy._LIMIT_NAMES:
            setattr(result, name, 1e-9)
        if mode == "bad_header":
            result.abi_version += 1
        if mode == "nan_scale":
            result.scale = math.nan
        if mode == "nan_limit":
            result.energy_tolerance = math.nan
        if mode == "zero_limit":
            result.energy_tolerance = 0
        return _native.STATUS_SUCCESS

    selected = SimpleNamespace(
        generativeqc_correlation_convergence_query_v1=operation,
        generativeqc_status_message=lambda status: b"policy rejected",
    )
    with pytest.raises(RuntimeError, match="rejected|invalid receipt"):
        policy.convergence_policy(None, library=selected)


@pytest.mark.parametrize("budget", (None, TargetErrorBudget(1e-5, 1e-4)))
def test_missing_protocol_rejects_discovered_library(
    monkeypatch: pytest.MonkeyPatch, budget: TargetErrorBudget | None
) -> None:
    """Even strict standalone metadata must come from the discovered native owner."""
    monkeypatch.setattr(_native, "load_library", lambda: SimpleNamespace())
    with pytest.raises(
        NotImplementedError, match="required correlated-convergence query"
    ):
        policy.convergence_policy(budget)


def test_cbs_planning_rejects_missing_protocol(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = SimpleNamespace(
        generativeqc_cbs_extrapolation_query_v1=library.generativeqc_cbs_extrapolation_query_v1
    )
    monkeypatch.setattr(_native, "load_library", lambda **options: selected)
    bases = (
        CBSBasis(snapshot_basis("sto-3g"), 2),
        CBSBasis(snapshot_basis("def2-svp"), 3),
    )
    with pytest.raises(
        NotImplementedError, match="selected native CBS convergence query"
    ):
        CCSDTCBSCalculator(bases)


def test_loading_errors_are_not_missing_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken configured library must not silently select compatibility."""
    path = tmp_path / "broken.so"
    path.write_text("not a shared library")
    monkeypatch.setattr(_native, "_candidate_paths", lambda: (path,))
    with pytest.raises(OSError):
        policy.convergence_policy(None)
