"""Native CPU recycle ownership, independent projectors and bounded failure behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _native
from generativeqc.response_solver import (
    GMRESOptions,
    KrylovRecycleSpace,
    ResponseCompatibilityError,
    solve,
    solve_many,
)


@pytest.fixture
def library() -> Any:
    result = _native.optional_response_recycle_library()
    assert result is not None, "build the current native recycle protocol for this gate"
    return result


def problem() -> Any:
    return SimpleNamespace(compatibility_identity="independent-recycle-v1")


def result(solution: np.ndarray, basis: np.ndarray) -> Any:
    return SimpleNamespace(converged=True, solution=solution, basis=basis)


@pytest.mark.parametrize("dimension", [3, 9, 25])
@pytest.mark.parametrize("scale", [1e-170, 1.0, 1e170])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_native_recycle_matches_independent_qr_projector(
    library: Any, dimension: int, scale: float, layout: str
) -> None:
    random = np.random.default_rng(dimension)
    orthogonal, _ = np.linalg.qr(random.normal(size=(dimension, 3)))
    solution = scale * orthogonal[:, 0]
    basis = scale * np.column_stack(
        [
            -0.4 * orthogonal[:, 0],
            orthogonal[:, 1],
            np.zeros(dimension),
            orthogonal[:, 2],
        ]
    )
    if layout == "negative":
        solution, basis, orthogonal = (
            solution[::-1],
            basis[::-1, ::-1],
            orthogonal[::-1],
        )
    elif layout == "strided":
        storage = np.full((2 * dimension, 8), 917.0)
        storage[::2, ::2] = basis
        basis = storage[::2, ::2]
    else:
        basis = np.array(basis, order=layout)
    current = problem()
    with KrylovRecycleSpace(current, max_vectors=4) as space:
        space.update(current, result(solution, basis))
        rhs = random.normal(size=dimension)
        expected = (
            orthogonal @ (orthogonal.T @ rhs) if scale >= 1e-12 else np.zeros(dimension)
        )
        np.testing.assert_allclose(
            space.initial_guess(current, rhs), expected, atol=3e-12, rtol=3e-12
        )
        assert space.storage_bytes == (3 * dimension * 8 if scale >= 1e-12 else 0)
        assert space.generation == 1


class MatrixOperator:
    """Nonsymmetric independent dense action, never a GPU provider."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.dimension = len(matrix)
        self.actions = 0
        self.problem = SimpleNamespace(
            compatibility_identity="recycle-dense-v1",
            validate_rhs=lambda values: np.asarray(values),
        )

    def apply(self, vector: np.ndarray) -> np.ndarray:
        self.actions += 1
        return self.matrix @ vector


def test_production_recycling_keeps_vectors_native_between_rhs(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoDiagnosticCopies:
        def __getattr__(self, name: str) -> Any:
            return getattr(library, name)

        def generativeqc_response_recycle_copy_v1(self, *arguments: Any) -> int:
            pytest.fail(
                "normal solve projection/update published retained vectors to Python"
            )

    monkeypatch.setattr(
        _native, "optional_response_recycle_library", lambda: NoDiagnosticCopies()
    )
    random = np.random.default_rng(2463)
    matrix = np.diag(np.linspace(2.0, 4.0, 12)) + 0.02 * random.normal(size=(12, 12))
    operator = MatrixOperator(matrix)
    rhs = random.normal(size=(12, 4))
    rhs[:, 1], rhs[:, 3] = -0.4 * rhs[:, 0], 0.0
    answer = solve_many(operator, rhs, strategy="recycled", raise_on_failure=True)
    np.testing.assert_allclose(
        answer.solution, np.linalg.solve(matrix, rhs), atol=1e-10
    )
    assert answer.operator_actions == operator.actions
    assert all(item.recycled_vectors > 0 for item in answer.results[1:])


def test_native_failure_does_not_retry_host_replacement(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Rejected:
        def __getattr__(self, name: str) -> Any:
            return getattr(library, name)

        def generativeqc_response_recycle_update_v1(self, *arguments: Any) -> int:
            return 8

    monkeypatch.setattr(
        _native, "optional_response_recycle_library", lambda: Rejected()
    )
    current = problem()
    with KrylovRecycleSpace(current) as space:
        with pytest.raises(RuntimeError):
            space.update(current, result(np.ones(4), np.empty((4, 0))))
        assert space.generation == space.storage_bytes == 0


def test_whole_solve_preflight_precedes_native_owner_creation(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoCreate:
        def __getattr__(self, name: str) -> Any:
            return getattr(library, name)

        def generativeqc_response_recycle_create_v1(self, *arguments: Any) -> int:
            pytest.fail("native retained owner acquired before whole-solve admission")

    monkeypatch.setattr(
        _native, "optional_response_recycle_library", lambda: NoCreate()
    )
    operator = MatrixOperator(np.eye(4))
    with KrylovRecycleSpace(operator.problem) as space:
        answer = solve(
            operator,
            np.ones(4),
            recycle=space,
            options=GMRESOptions(max_workspace_bytes=1),
        )
        assert answer.reason == "workspace_limit"
        assert operator.actions == space.generation == space.storage_bytes == 0


def test_bad_candidate_rolls_back_and_recovers(library: Any) -> None:
    current = problem()
    with KrylovRecycleSpace(current, max_vectors=3) as space:
        with pytest.raises(ValueError):
            space.update(current, result(np.ones(4), np.full((4, 1), np.nan)))
        assert space.generation == space.storage_bytes == 0
        space.update(current, result(np.arange(1.0, 5.0), np.empty((4, 0))))
        saved = space.initial_guess(current, np.ones(4))
        before = saved.copy()
        space.reset()
        space.update(current, result(np.arange(4.0, 0.0, -1), np.empty((4, 0))))
        np.testing.assert_array_equal(saved, before)


def test_changed_limits_and_empty_reference_keep_public_semantics(library: Any) -> None:
    current = problem()
    with KrylovRecycleSpace(current, max_vectors=4) as space:
        space.update(current, result(np.eye(4)[0], np.eye(4).T))
        space.max_vectors = 1
        assert space.update_bytes(4) == 4 * 4 * 8
        space.update(current, result(np.ones(4), np.empty((4, 0))))
        assert space.storage_bytes == 4 * 8
        with pytest.raises(ValueError):
            space.initial_guess(current, np.empty(0))
        space.update(current, result(np.empty(0), np.empty((0, 0))))
        assert space.storage_bytes == 0
        assert space.initial_guess(current, np.empty(0)).shape == (0,)


def test_foreign_update_uses_explicit_compatibility_not_native_retry(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DomainGuard:
        foreign = False

        def __getattr__(self, name: str) -> Any:
            return getattr(library, name)

        def generativeqc_response_recycle_update_v1(self, *arguments: Any) -> int:
            if self.foreign:
                pytest.fail("foreign dtype entered the FP64 native update protocol")
            return library.generativeqc_response_recycle_update_v1(*arguments)

    guarded = DomainGuard()
    monkeypatch.setattr(_native, "optional_response_recycle_library", lambda: guarded)
    current = problem()
    with KrylovRecycleSpace(current, max_vectors=2) as space:
        space.update(current, result(np.array([1.0, 0.0, 0.0]), np.empty((3, 0))))
        guarded.foreign = True
        space.update(
            current,
            result(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.empty((3, 0))),
        )
        np.testing.assert_allclose(
            space.initial_guess(current, np.ones(3)), [1, 1, 0], atol=1e-14
        )


def test_stale_problem_and_explicit_transport_are_not_native_aliases(
    library: Any,
) -> None:
    current = problem()
    with KrylovRecycleSpace(current) as space:
        space.update(current, result(np.array([1.0, 0.0, 0.0]), np.empty((3, 0))))
        changed = SimpleNamespace(compatibility_identity="changed", dimension=3)
        with pytest.raises(ResponseCompatibilityError):
            space.initial_guess(changed, np.ones(3))
        transported = space.transport(changed, lambda vector: vector)
        space.reset()
        np.testing.assert_allclose(
            transported.initial_guess(changed, np.ones(3)), [1, 0, 0]
        )
        transported.close()


def test_absent_protocol_and_declared_device_domain_preserve_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "optional_response_recycle_library", lambda: None)
    operator = MatrixOperator(np.diag([2.0, 3.0, 4.0]))
    with KrylovRecycleSpace(operator.problem) as space:
        answer = solve(operator, np.ones(3), recycle=space, raise_on_failure=True)
        np.testing.assert_allclose(answer.solution, [0.5, 1 / 3, 0.25], atol=1e-10)

    def unexpected() -> Any:
        pytest.fail("declared device provider discovered the CPU recycle protocol")

    monkeypatch.setattr(_native, "optional_response_recycle_library", unexpected)
    operator.device_workspace_bytes = 1
    with KrylovRecycleSpace(operator.problem) as space:
        answer = solve(operator, np.ones(3), recycle=space, raise_on_failure=True)
        assert answer.converged


def test_foreign_failure_preserves_native_retention_until_success(library: Any) -> None:
    current = problem()
    with KrylovRecycleSpace(current, max_vectors=3) as space:
        space.update(current, result(np.array([1.0, 0.0, 0.0]), np.empty((3, 0))))
        with pytest.raises(ValueError):
            space.update(
                current, result(np.full(3, np.nan, dtype=np.float32), np.empty((3, 0)))
            )
        assert space.generation == 1 and space.storage_bytes == 24
        np.testing.assert_allclose(space.initial_guess(current, np.ones(3)), [1, 0, 0])
        space.max_vectors = 1
        space.update(current, result(np.ones(3, dtype=np.float32), np.empty((3, 0))))
        assert space.generation == 2 and space.storage_bytes == 24
        np.testing.assert_allclose(space.initial_guess(current, np.ones(3)), [1, 0, 0])


def test_selected_library_load_error_is_not_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = OSError("selected recycle library is broken")

    def broken() -> Any:
        raise injected

    monkeypatch.setattr(_native, "optional_response_recycle_library", broken)
    current = problem()
    with KrylovRecycleSpace(current) as space:
        with pytest.raises(OSError) as caught:
            space.initial_guess(current, np.ones(3))
        assert caught.value is injected
