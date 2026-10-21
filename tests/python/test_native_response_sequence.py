"""Native sequential RHS ownership, independent oracles and partial-budget/lifetime gates."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _native
from generativeqc.response_solver import (
    DiagonalPreconditioner,
    GMRESOptions,
    solve,
    solve_many,
)


class MatrixOperator:
    """An independent nonsymmetric physical action with observable actual work."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.dimension = len(matrix)
        self.actions = 0
        self.problem = SimpleNamespace(
            validate_rhs=lambda values: np.asarray(values),
            compatibility_identity="sequence-independent-matrix",
        )

    def apply(self, vector: np.ndarray) -> np.ndarray:
        self.actions += 1
        return self.matrix @ vector


@pytest.fixture
def library() -> Any:
    result = _native.optional_response_sequence_library()
    assert result is not None, (
        "build the current native sequence protocol for this required gate"
    )
    return result


@pytest.mark.parametrize("dimension", [3, 11, 25])
@pytest.mark.parametrize("scale", [1e-170, 1.0, 1e170])
@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_sequence_matches_independent_dense_oracle(
    library: Any, dimension: int, scale: float, preconditioned: bool, layout: str
) -> None:
    random = np.random.default_rng(dimension)
    matrix = np.diag(np.linspace(2, 5, dimension)) + 0.03 * random.normal(
        size=(dimension, dimension)
    )
    rhs = scale * random.normal(size=(dimension, 4))
    rhs[:, 1], rhs[:, 2] = -0.4 * rhs[:, 0], 0.0
    if layout == "strided":
        storage = np.full((dimension, 8), 193.0)
        storage[:, ::2] = rhs
        rhs = storage[:, ::2]
    else:
        rhs = np.array(rhs, order=layout)
    operator = MatrixOperator(matrix)
    result = solve_many(
        operator,
        rhs,
        options=GMRESOptions(rtol=1e-11, restart=dimension),
        preconditioner=DiagonalPreconditioner(np.diag(matrix))
        if preconditioned
        else None,
        raise_on_failure=True,
    )
    expected = np.linalg.solve(matrix, rhs / scale)
    np.testing.assert_allclose(
        result.solution / scale, expected, atol=2e-11, rtol=2e-10
    )
    assert (
        result.operator_actions
        == operator.actions
        == sum(item.operator_actions for item in result.results)
    )
    assert result.rhs_rank == np.linalg.matrix_rank(rhs, tol=1e-12)
    assert all(item.relative_residual <= 1e-11 for item in result.results)
    for item in result.results:
        with pytest.raises(ValueError):
            item.solution.flags.writeable = True
        with pytest.raises(ValueError):
            item.basis.flags.writeable = True


def test_native_sequence_preserves_a_budget_admitted_prefix(library: Any) -> None:
    # Leave enough room for the independent RHS-rank diagnostic before testing
    # streaming admission. A tiny solve can need less storage than that stage.
    operator = MatrixOperator(np.eye(8))
    rhs = np.column_stack([np.ones(8), np.arange(1.0, 9.0), np.ones(8)])
    options = GMRESOptions(restart=2, max_iterations=4)
    single = solve(operator, rhs[:, 0], options=options)
    staging_bytes = rhs.shape[0] * rhs.itemsize
    budget = rhs.nbytes + single.workspace_bytes + staging_bytes
    operator.actions = 0
    result = solve_many(
        operator, rhs, options=replace(options, max_workspace_bytes=budget)
    )
    assert result.results[0].converged
    assert all(item.reason == "workspace_limit" for item in result.results[1:])
    assert result.results[1] is result.results[2]
    assert operator.actions == result.results[0].operator_actions
    assert not result.converged


def test_native_sequence_error_never_retries_python_science(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Rejected:
        def __getattr__(self, name: str) -> Any:
            return getattr(library, name)

        def generativeqc_response_sequence_solve_v1(
            self, descriptor: Any, summary: Any
        ) -> int:
            return _native.STATUS_INTERNAL_ERROR

    monkeypatch.setattr(
        _native, "optional_response_sequence_library", lambda: Rejected()
    )
    operator = MatrixOperator(np.eye(4))
    with pytest.raises(RuntimeError):
        solve_many(operator, np.ones((4, 2)))
    assert operator.actions == 0


def test_callback_exception_and_retained_input_survive_sequence_cleanup(
    library: Any,
) -> None:
    operator = MatrixOperator(np.eye(4))
    retained = []
    injected = RuntimeError("sequence action failure")

    def fail(vector: np.ndarray) -> np.ndarray:
        retained.append(vector)
        raise injected

    operator.apply = fail
    with pytest.raises(RuntimeError) as caught:
        solve_many(operator, np.ones((4, 3)))
    assert caught.value is injected
    saved = retained[0].copy()
    recovered = solve_many(MatrixOperator(2 * np.eye(37)), np.ones((37, 4)))
    assert recovered.converged
    np.testing.assert_array_equal(retained[0], saved)
    assert not retained[0].flags.writeable


@pytest.mark.parametrize("initially_stale", [False, True])
def test_stale_reference_is_rejected_even_when_zero_rhs_skips_the_action(
    library: Any,
    initially_stale: bool,
) -> None:
    operator = MatrixOperator(np.eye(4))
    injected = RuntimeError("stale sequence reference")

    first_actions = None

    def validate() -> None:
        nonlocal first_actions
        if initially_stale or first_actions is not None:
            raise injected
        if operator.actions:
            first_actions = operator.actions

    operator.validate_current = validate
    rhs = np.zeros((4, 2))
    if not initially_stale:
        rhs[:, 0] = 1.0
    with pytest.raises(RuntimeError) as caught:
        solve_many(operator, rhs)
    assert caught.value is injected
    assert operator.actions == (0 if initially_stale else first_actions)


def test_older_optional_sequence_protocol_keeps_numerical_compatibility(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OlderLibrary:
        def __getattr__(self, name: str) -> Any:
            if name.startswith("generativeqc_response_sequence_"):
                raise AttributeError(name)
            return getattr(library, name)

    monkeypatch.setattr(_native, "optional_response_library", lambda: OlderLibrary())
    matrix = np.array([[2.0, 0.3], [-0.1, 3.0]])
    rhs = np.array([[0.2, -0.1], [0.8, 0.3]])
    result = solve_many(MatrixOperator(matrix), rhs, raise_on_failure=True)
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), atol=1e-12
    )


@pytest.mark.parametrize("foreign", ["negative-stride", "float32", "device-provider"])
def test_foreign_domains_keep_their_existing_controller(
    monkeypatch: pytest.MonkeyPatch, foreign: str
) -> None:
    operator = MatrixOperator(np.diag([2.0, 3.0, 4.0]))
    rhs = np.arange(1.0, 7.0).reshape(3, 2)
    if foreign == "negative-stride":
        rhs = rhs[::-1]
    elif foreign == "float32":
        rhs = rhs.astype(np.float32)
    else:
        operator.device_workspace_bytes = 1

    def unexpected() -> Any:
        pytest.fail("foreign controller must not discover the CPU sequence protocol")

    monkeypatch.setattr(_native, "optional_response_sequence_library", unexpected)
    result = solve_many(operator, rhs, raise_on_failure=True)
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(operator.matrix, rhs), atol=1e-12
    )
