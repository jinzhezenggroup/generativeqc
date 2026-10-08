"""CPU-only buffer reuse checks for native-XC and Krylov Python orchestration."""

from __future__ import annotations

import inspect
import typing

import numpy as np
from generativeqc import response_solver
from generativeqc._stationary_cpu import _xc_gradient_argument
from generativeqc_compiler.xc.grid_response import partition_response


class _MatrixOperator:
    """Small independent host Krylov problem without native or CUDA backends."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.dimension = len(matrix)
        self.problem = type(
            "Problem",
            (),
            {
                "compatibility_identity": "cpu-scratch",
                "validate_rhs": self._validate_rhs,
            },
        )()

    def _validate_rhs(self, rhs: typing.Any) -> np.ndarray:
        value = np.asarray(rhs, dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape[0] != self.dimension:
            raise ValueError("invalid response RHS dimension")
        return value

    def apply(self, values: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(values, dtype=np.float64)


def test_xc_gradient_present_does_not_allocate_default(monkeypatch: typing.Any) -> None:
    gradient = np.arange(30, dtype=np.float64).reshape(2, 5, 3)
    original = np.zeros
    allocation_shapes: list[object] = []

    def traced_zeros(
        shape: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> np.ndarray:
        allocation_shapes.append(shape)
        return original(shape, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "zeros", traced_zeros)
        assert _xc_gradient_argument({"gradient": gradient}, 5) is gradient
        assert _xc_gradient_argument({"gradient": None}, 5) is None
        assert allocation_shapes == []
        fallback = _xc_gradient_argument({}, 5)
        assert allocation_shapes == [(2, 5, 3)]
    assert fallback.shape == (2, 5, 3)
    assert not fallback.any()


def test_scalar_gmres_allocates_one_arnoldi_matrix_across_restarts(
    monkeypatch: typing.Any,
) -> None:
    matrix = np.diag(np.linspace(1.0, 4.0, 12))
    matrix[0, 1] = 0.2
    matrix[1, 0] = -0.1
    operator = _MatrixOperator(matrix)
    rhs = np.linspace(-1.0, 1.0, 12)
    options = response_solver.GMRESOptions(
        rtol=1e-12, restart=2, max_iterations=100, stagnation_window=100
    )
    original = np.zeros
    arnoldi_allocations = 0

    def traced_zeros(
        shape: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> np.ndarray:
        nonlocal arnoldi_allocations
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_solve_single":
            if shape == (3, 2):
                arnoldi_allocations += 1
        return original(shape, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "zeros", traced_zeros)
        result = response_solver.solve(operator, rhs, options=options)
    assert result.iterations > options.restart
    assert arnoldi_allocations == 1
    assert result.converged
    np.testing.assert_allclose(matrix @ result.solution, rhs, rtol=0, atol=1e-9)


def test_block_gmres_reuses_one_projected_rhs_buffer(
    monkeypatch: typing.Any,
) -> None:
    matrix = np.diag(np.linspace(1.0, 4.0, 12))
    operator = _MatrixOperator(matrix)
    rhs = np.column_stack((np.linspace(0.1, 1.0, 12), np.linspace(-1.0, 0.5, 12)))
    options = response_solver.GMRESOptions(rtol=1e-12, restart=8, max_iterations=100)
    max_columns = min(operator.dimension, rhs.shape[1] + options.max_iterations)
    original = np.zeros
    rhs_allocations: list[int] = []

    def traced_zeros(
        shape: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> np.ndarray:
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_block_solve":
            if isinstance(shape, int):
                rhs_allocations.append(shape)
        return original(shape, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "zeros", traced_zeros)
        result = response_solver.solve_many(
            operator, rhs, strategy="blocked", options=options
        )
    assert rhs_allocations == [max_columns]
    assert result.converged
    np.testing.assert_allclose(matrix @ result.solution, rhs, rtol=0, atol=1e-9)


def test_reused_becke_impulse_matches_fresh_direction_for_all_atoms() -> None:
    """Exact partition JVP parity across reused tiles and axis directions."""
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.1, 1.3, 0.0]],
        dtype=np.float64,
    )
    tiles = (
        (np.array([[0.3, 0.2, 0.7], [0.9, 0.5, 0.1]]), np.array([0, 1])),
        (np.array([[0.1, 0.4, 0.8]]), np.array([2])),
    )
    reusable = np.zeros((len(centers), 3), dtype=np.float64)
    for points, owners in tiles:
        for atom in range(len(centers)):
            for axis in range(3):
                fresh = np.zeros_like(reusable)
                fresh[atom, axis] = 1.0
                expected = partition_response(
                    points, centers, point_motion=fresh[owners], center_motion=fresh
                )
                reusable[atom, axis] = 1.0
                actual = partition_response(
                    points,
                    centers,
                    point_motion=reusable[owners],
                    center_motion=reusable,
                )
                np.testing.assert_array_equal(actual.directional, expected.directional)
                np.testing.assert_array_equal(actual.weights, expected.weights)
                reusable[atom, axis] = 0.0
                assert not reusable.any()
