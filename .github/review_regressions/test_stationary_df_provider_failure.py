"""Execute the real CUDA derivative-selection block without a device library."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _select(source: object, *, budget: int = 32, ecp: bool = False) -> object:
    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    module = ast.parse(path.read_text())
    owner = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_complete_rks_cuda_gradient_diagnostic")
    block = next(n for n in ast.walk(owner) if isinstance(n, ast.With)
                 and any(isinstance(child, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "native_integral_components"
                                 for t in child.targets) for child in n.body))
    def is_assignment(statement: ast.stmt, name: str) -> bool:
        return isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in statement.targets
        )
    start = next(i for i, n in enumerate(block.body) if is_assignment(n, "native_integral_components"))
    stop = next(i for i, n in enumerate(block.body) if is_assignment(n, "native_complete_integrals"))
    function = ast.parse("def select():\n    pass").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = block.body[start:stop] + ast.parse("return native_integral_components").body
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec")
    compiled = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == "select")
    return FunctionType(compiled, {
        "state": SimpleNamespace(_source=source), "max_device_bytes": budget,
        "peak": 0, "ecp": ecp, "na": 2, "np": np, "MappingProxyType": MappingProxyType,
        "timeline": SimpleNamespace(phase=lambda _: nullcontext()),
    })()


@pytest.mark.parametrize("failure", ["missing", "not-callable", "unavailable", "no-budget", "ecp"])
def test_fitted_derivative_failure_never_selects_exact_fallback(failure: str) -> None:
    exact = Mock(side_effect=AssertionError("Direct must not execute for a fitted state"))
    source = SimpleNamespace(density_fitted=True, cuda_integral_derivatives=exact)
    if failure != "missing":
        source.density_fitted_integral_derivatives = None if failure == "not-callable" else Mock(return_value=None)
    with pytest.raises(NotImplementedError, match="change the Hamiltonian"):
        _select(source, budget=0 if failure == "no-budget" else 32, ecp=failure == "ecp")
    exact.assert_not_called()


def test_fitted_provider_errors_propagate_without_exact_retry() -> None:
    source = SimpleNamespace(density_fitted=True,
        density_fitted_integral_derivatives=Mock(side_effect=RuntimeError("stale DF response")),
        cuda_integral_derivatives=Mock())
    with pytest.raises(RuntimeError, match="stale DF response"):
        _select(source)
    source.cuda_integral_derivatives.assert_not_called()


@pytest.mark.parametrize("invalid", [np.zeros((2, 2, 3)), np.full((4, 2, 3), np.nan)])
def test_fitted_provider_rejects_invalid_publication(invalid: np.ndarray) -> None:
    source = SimpleNamespace(density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(invalid, {})))
    with pytest.raises(RuntimeError, match="invalid output"):
        _select(source)


def test_complete_fitted_response_and_explicit_exact_fallback_remain_distinct() -> None:
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(output, {})),
        cuda_integral_derivatives=Mock(return_value=None))
    np.testing.assert_array_equal(_select(source), output)
    source.density_fitted_integral_derivatives.assert_called_once_with(2, 32)
    source.density_fitted = False
    assert _select(source) is None
    source.cuda_integral_derivatives.assert_called_once_with(2, 32, range_exchange=False)
