"""Shared CPU GMRES: independent dense oracles, ABI failures and callback lifetimes."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import _native
from generativeqc.response_solver import (
    DiagonalPreconditioner,
    GMRESOptions,
    solve,
    solve_many,
)

if TYPE_CHECKING:
    from pathlib import Path


class MatrixOperator:
    """A nonsymmetric linear action with public observable work and RHS contracts."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.dimension = len(matrix)
        self.actions = 0
        self.problem = type(
            "Problem",
            (),
            {
                "validate_rhs": staticmethod(lambda values: np.asarray(values)),
                "compatibility_identity": "independent-test-matrix",
            },
        )()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        self.actions += 1
        return self.matrix @ vector


@pytest.fixture
def library() -> Any:
    result = _native.optional_response_library()
    assert result is not None, (
        "build the current native response protocol for this required gate"
    )
    return result


@pytest.mark.parametrize("older_protocol", [False, True])
def test_missing_optional_protocol_preserves_numerical_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, older_protocol: bool
) -> None:
    path = tmp_path / "optional-response.so"
    if older_protocol:
        path.touch()
        monkeypatch.setattr(ctypes, "CDLL", lambda _: object())
    monkeypatch.setattr(_native, "_candidate_paths", lambda: iter([path]))
    matrix = np.array([[3.0, 0.2], [-0.1, 2.0]])
    rhs = np.array([0.5, -0.3])
    result = solve(MatrixOperator(matrix), rhs, raise_on_failure=True)
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), atol=1e-12
    )


def test_selected_library_load_error_never_retries_python_arithmetic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "invalid-response.so"
    path.write_bytes(b"not a shared library")
    monkeypatch.setattr(_native, "_candidate_paths", lambda: iter([path]))
    operator = MatrixOperator(np.eye(2))
    with pytest.raises(OSError):
        solve(operator, np.ones(2))
    assert operator.actions == 0


@pytest.mark.parametrize("dimension", [3, 8, 17])
@pytest.mark.parametrize("scale", [1.0, 1e-170, 1e170])
@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("with_guess", [False, True])
def test_nonsymmetric_native_solve_matches_independent_dense_oracle(
    library: Any,
    dimension: int,
    scale: float,
    preconditioned: bool,
    with_guess: bool,
) -> None:
    random = np.random.default_rng(dimension)
    matrix = np.diag(np.linspace(2, 5, dimension)) + 0.05 * random.normal(
        size=(dimension, dimension)
    )
    operator = MatrixOperator(matrix)
    rhs = scale * random.normal(size=dimension)
    expected = np.linalg.solve(matrix, rhs / scale)
    options = GMRESOptions(rtol=1e-11, restart=dimension)
    preconditioner = DiagonalPreconditioner(np.diag(matrix)) if preconditioned else None
    result = solve(
        operator,
        rhs,
        options=options,
        preconditioner=preconditioner,
        initial_guess=0.2 * scale * np.ones(dimension) if with_guess else None,
        raise_on_failure=True,
    )
    np.testing.assert_allclose(
        result.solution / scale, expected, atol=2e-11, rtol=2e-10
    )
    actual_residual = np.linalg.norm((rhs - matrix @ result.solution) / scale) * scale
    assert result.residual_norm == pytest.approx(
        actual_residual, rel=0.03, abs=2e-15 * scale
    )
    assert result.relative_residual <= options.rtol
    assert result.operator_actions == operator.actions
    assert result.history[-1] == result.residual_norm
    assert np.isfinite(result.history).all()
    assert result.basis.shape[0] == dimension
    with pytest.raises(ValueError):
        result.solution.flags.writeable = True
    with pytest.raises(ValueError):
        result.basis.flags.writeable = True


class ProtocolView:
    """Inject stable public ABI failures without replacing the scientific operator."""

    def __init__(
        self, library: Any, *, reject: bool = False, fail_view: bool = False
    ) -> None:
        self.library, self.reject, self.fail_view = library, reject, fail_view
        self.live: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.library, name)

    def generativeqc_response_solve_v1(self, descriptor: Any, output: Any) -> int:
        if self.reject:
            ctypes.cast(
                descriptor, ctypes.POINTER(_native.ResponseGmresDescriptor)
            ).contents.flags |= 1 << 24
        status = self.library.generativeqc_response_solve_v1(descriptor, output)
        owner = ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value
        if owner:
            self.live.add(owner)
        return status

    def generativeqc_response_result_view_v1_get(self, owner: Any, output: Any) -> int:
        if self.fail_view:
            return _native.STATUS_NUMERICAL_FAILURE
        return self.library.generativeqc_response_result_view_v1_get(owner, output)

    def generativeqc_response_result_destroy(self, owner: Any) -> None:
        self.live.remove(owner.value)
        self.library.generativeqc_response_result_destroy(owner)


@pytest.mark.parametrize("reject", [False, True])
def test_native_failure_does_not_retry_and_releases_published_handles(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    reject: bool,
) -> None:
    protocol = ProtocolView(library, reject=reject, fail_view=not reject)
    monkeypatch.setattr(_native, "optional_response_library", lambda: protocol)
    operator = MatrixOperator(np.eye(4))
    with pytest.raises(RuntimeError):
        solve(operator, np.ones(4))
    assert not protocol.live
    if reject:
        assert operator.actions == 0


def test_callback_exception_is_preserved_and_its_python_vector_outlives_native_cleanup(
    library: Any,
) -> None:
    operator = MatrixOperator(np.eye(4))
    retained = []
    injected = RuntimeError("injected callback failure")

    def action(vector: np.ndarray) -> np.ndarray:
        retained.append(vector)
        raise injected

    operator.apply = action
    with pytest.raises(RuntimeError, match="injected") as failure:
        solve(operator, np.arange(1, 5))
    assert failure.value is injected
    snapshot = retained[0].copy()
    recovered = solve(
        MatrixOperator(2 * np.eye(31)), np.ones(31), raise_on_failure=True
    )
    assert recovered.converged
    np.testing.assert_array_equal(retained[0], snapshot)
    assert not retained[0].flags.writeable


def test_cpu_native_workspace_exact_limit_and_one_byte_below(
    library: Any,
) -> None:
    from generativeqc._response_native import workspace_bytes

    options = GMRESOptions()
    required = workspace_bytes(library, 4, options)
    operator = MatrixOperator(np.eye(4))
    failed = solve(
        operator, np.ones(4), options=replace(options, max_workspace_bytes=required - 1)
    )
    assert not failed.converged and failed.reason == "workspace_limit"
    assert operator.actions == 0
    exact = solve(
        operator,
        np.ones(4),
        options=replace(options, max_workspace_bytes=required),
        raise_on_failure=True,
    )
    assert exact.workspace_bytes == required
    np.testing.assert_allclose(exact.solution, np.ones(4))


def test_foreign_device_provider_keeps_existing_controller_without_cpu_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> Any:
        raise AssertionError(
            "device provider attempted the CPU-only controller cutover"
        )

    monkeypatch.setattr(_native, "optional_response_library", forbidden)
    operator = MatrixOperator(np.diag([2.0, 3.0, 4.0]))
    operator.device_workspace_bytes = 64
    result = solve(operator, np.ones(3), raise_on_failure=True)
    np.testing.assert_allclose(result.solution, [0.5, 1 / 3, 0.25])


@pytest.mark.parametrize("strategy", ["sequential", "recycled"])
def test_native_and_explicit_portable_multirhs_keep_the_same_linear_problem(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    random = np.random.default_rng(41)
    matrix = np.diag(np.arange(2, 10)) + 0.02 * random.normal(size=(8, 8))
    rhs = random.normal(size=(8, 3))
    actual = solve_many(
        MatrixOperator(matrix), rhs, strategy=strategy, raise_on_failure=True
    )
    monkeypatch.setattr(_native, "optional_response_library", lambda: None)
    portable = solve_many(
        MatrixOperator(matrix), rhs, strategy=strategy, raise_on_failure=True
    )
    for column, (native, reference) in enumerate(
        zip(actual.results, portable.results, strict=True)
    ):
        np.testing.assert_allclose(
            native.solution, reference.solution, atol=2e-10, rtol=2e-9
        )
        np.testing.assert_allclose(
            matrix @ native.solution, rhs[:, column], atol=2e-10, rtol=2e-9
        )
