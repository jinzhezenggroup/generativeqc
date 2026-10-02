"""Execute the real CUDA derivative-selection block without a device library."""

from __future__ import annotations

import ast
import builtins
from contextlib import nullcontext
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _select(
    source: object,
    *,
    budget: int = 32,
    ecp: bool = False,
    required: bool = False,
    host_reserve: int = 32,
) -> object:
    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    module = ast.parse(path.read_text())
    owner = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_complete_rks_cuda_gradient_diagnostic"
    )
    block = next(
        n
        for n in ast.walk(owner)
        if isinstance(n, ast.With)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "native_integral_components"
                for t in child.targets
            )
            for child in n.body
        )
    )

    def is_assignment(statement: ast.stmt, name: str) -> bool:
        return isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in statement.targets
        )

    start = next(
        i
        for i, n in enumerate(block.body)
        if is_assignment(n, "native_integral_components")
    )
    stop = next(
        i for i, n in enumerate(block.body) if is_assignment(n, "resident_grid_density")
    )
    function = ast.parse("def select():\n    pass").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = (
        block.body[start:stop] + ast.parse("return native_integral_components").body
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(
        c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == "select"
    )
    return FunctionType(
        compiled,
        {
            "__builtins__": builtins.__dict__,
            "state": SimpleNamespace(_source=source),
            "max_device_bytes": budget,
            "peak": 0,
            "ecp": ecp,
            "requires_native_integrals": required,
            "native_integral_host_reserve": host_reserve,
            "na": 2,
            "np": np,
            "MappingProxyType": MappingProxyType,
            "timeline": SimpleNamespace(phase=lambda _: nullcontext()),
        },
    )()


@pytest.mark.parametrize(
    "failure", ["missing", "not-callable", "unavailable", "no-budget", "ecp"]
)
def test_fitted_derivative_failure_never_selects_exact_fallback(failure: str) -> None:
    exact = Mock(
        side_effect=AssertionError("Direct must not execute for a fitted state")
    )
    source = SimpleNamespace(density_fitted=True, cuda_integral_derivatives=exact)
    if failure != "missing":
        source.density_fitted_integral_derivatives = (
            None if failure == "not-callable" else Mock(return_value=None)
        )
    with pytest.raises(NotImplementedError, match="change the Hamiltonian"):
        _select(
            source, budget=0 if failure == "no-budget" else 32, ecp=failure == "ecp"
        )
    exact.assert_not_called()
    if failure in ("no-budget", "ecp"):
        source.density_fitted_integral_derivatives.assert_not_called()


def test_fitted_provider_errors_propagate_without_exact_retry() -> None:
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(
            side_effect=RuntimeError("stale DF response")
        ),
        cuda_integral_derivatives=Mock(),
    )
    with pytest.raises(RuntimeError, match="stale DF response"):
        _select(source)
    source.cuda_integral_derivatives.assert_not_called()


@pytest.mark.parametrize("invalid", [np.zeros((2, 2, 3)), np.full((4, 2, 3), np.nan)])
def test_fitted_provider_rejects_invalid_publication(invalid: np.ndarray) -> None:
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(invalid, {})),
    )
    with pytest.raises(RuntimeError, match="invalid output"):
        _select(source)


def test_complete_fitted_response_and_explicit_exact_fallback_remain_distinct() -> None:
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(output, {})),
        cuda_integral_derivatives=Mock(return_value=None),
    )
    np.testing.assert_array_equal(_select(source), output)
    source.density_fitted_integral_derivatives.assert_called_once_with(2, 32)
    source.density_fitted = False
    assert _select(source) is None
    source.cuda_integral_derivatives.assert_called_once_with(
        2, 32, range_exchange=False
    )


@pytest.mark.parametrize("failure", ["missing", "unavailable", "no-budget"])
def test_enlarged_direct_domain_cannot_select_ao_task_fallback(failure: str) -> None:
    source = SimpleNamespace(density_fitted=False)
    if failure != "missing":
        source.cuda_integral_derivatives = Mock(return_value=None)
    with pytest.raises(NotImplementedError, match="cannot use AO-task fallback"):
        _select(source, budget=0 if failure == "no-budget" else 32, required=True)


def test_enlarged_domain_accepts_complete_native_owner_only() -> None:
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(return_value=(output, {})),
    )
    np.testing.assert_array_equal(_select(source, required=True), output)
    source.cuda_integral_derivatives.assert_called_once_with(
        2, 32, range_exchange=False
    )


@pytest.mark.parametrize("actual", [32, 33])
def test_native_host_staging_must_fit_its_concurrent_reserve(actual: int) -> None:
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(
            return_value=(np.zeros((4, 2, 3)), {"one_electron_host_peak_bytes": actual})
        ),
    )
    if actual == 32:
        assert _select(source, required=True, host_reserve=32) is not None
    else:
        with pytest.raises(RuntimeError, match="host staging exceeds"):
            _select(source, required=True, host_reserve=32)


def _cpu_selection(source: object, execution: str) -> bool:
    path = ROOT / "python/generativeqc/_stationary_cpu.py"
    module = ast.parse(path.read_text())
    owner = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "complete_rks_gradient_diagnostic"
    )

    def assigned(statement: ast.stmt, name: str) -> bool:
        return isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        )

    start = next(
        i for i, n in enumerate(owner.body) if assigned(n, "native_fitted_integrals")
    )
    stop = next(i for i, n in enumerate(owner.body) if assigned(n, "work"))
    function = ast.parse("def select():\n    pass").body[0]
    function.body = (
        owner.body[start:stop] + ast.parse("return native_fitted_integrals").body
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(
        c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == "select"
    )
    return FunctionType(
        compiled,
        {
            "__builtins__": builtins.__dict__,
            "state": SimpleNamespace(_source=source),
            "execution": execution,
        },
    )()


def test_cpu_reference_derivatives_reject_a_fitted_hamiltonian() -> None:
    with pytest.raises(NotImplementedError, match="change the Hamiltonian"):
        _cpu_selection(SimpleNamespace(density_fitted=True), "reference")
    assert _cpu_selection(SimpleNamespace(density_fitted=True), "native")
    assert not _cpu_selection(SimpleNamespace(density_fitted=False), "reference")
    assert not _cpu_selection(SimpleNamespace(), "native")
