"""Independent primal controls for bounded analytic residual-ECP mixed sources."""

from __future__ import annotations

import ctypes as ct
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_native_ecp_direction import (
    CENTERS,
    MOTION,
    POINTER,
    SIZE_POINTER,
    _oracle,
    _owner,
    _pointer,
)
from test_native_ecp_direction import native as native_fixture

if TYPE_CHECKING:
    from pathlib import Path

RIGHT = np.array([[-0.11, 0.07, 0.16], [0.08, -0.13, 0.04], [-0.03, 0.12, -0.09]])


@pytest.fixture(scope="module")
def mixed_native(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Any:
    """Bind real-core mixed APIs through the shared normalized test owner."""
    loaded = native_fixture.__wrapped__(
        tmp_path_factory, required_native_cxx, policy_library
    )
    loaded.ecp_mixed_plan.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.ecp_mixed_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        SIZE_POINTER,
        POINTER,
        ct.c_size_t,
        ct.c_int,
    ]
    return loaded


def _plan(native: Any, handle: Any) -> np.ndarray:
    result = np.zeros(2, dtype=np.uintp)
    assert native.ecp_mixed_plan(handle, result.ctypes.data_as(SIZE_POINTER)) == 0, (
        native.ecp_direction_error().decode()
    )
    return result


def _weights(count: int) -> np.ndarray:
    """Signed, deliberately nonsymmetric ordered weights have no occupancy factor."""
    return np.random.default_rng(2848 + count).normal(size=(count, count)) * 0.2


def _apply(
    native: Any,
    handle: Any,
    weights: np.ndarray,
    left: np.ndarray = MOTION,
    right: np.ndarray = RIGHT,
    *,
    limits: np.ndarray | None = None,
    output: np.ndarray | None = None,
    counts: tuple[int, int, int] | None = None,
    missing_validator: bool = False,
) -> tuple[int, np.ndarray, tuple[int, ...]]:
    if limits is None:
        limits = _plan(native, handle)
    if output is None:
        output = np.full((2, 4), 817.0)
    if counts is None:
        counts = (weights.size, left.size, right.size)
    status = native.ecp_mixed_apply(
        handle,
        _pointer(weights),
        counts[0],
        _pointer(left),
        counts[1],
        _pointer(right),
        counts[2],
        limits.ctypes.data_as(SIZE_POINTER),
        _pointer(output),
        output.size,
        int(missing_validator),
    )
    work = (ct.c_size_t * 10)()
    native.ecp_direction_work(handle, work)
    return status, output, tuple(work)


@pytest.mark.parametrize("angular", range(4))
@pytest.mark.parametrize("spherical", [False, True])
def test_frozen_mixed_source_against_independent_primal_operators(
    mixed_native: Any,
    angular: int,
    spherical: bool,
    tmp_path: Path,
    record_testsuite_property: Any,
) -> None:
    """Both residual parts use four displaced libcint primals, never native AD."""
    with _owner(mixed_native, angular, spherical) as (handle, sizes, terms):
        weights = _weights(sizes[0])
        limits = _plan(mixed_native, handle)
        status, actual, work = _apply(mixed_native, handle, weights)
        assert status == 0, mixed_native.ecp_direction_error().decode()

        def primal(centers: np.ndarray) -> np.ndarray:
            return np.einsum(
                "pij,ij->p", _oracle(centers, terms, angular, spherical), weights
            )

        baseline = primal(CENTERS)
        np.testing.assert_allclose(actual[:, 0], baseline, atol=2e-11, rtol=2e-10)
        steps = [2e-2, 6e-3, 2e-3]
        primals, errors = [], []
        for step in steps:
            corners = np.array(
                [
                    primal(CENTERS + step * (first * MOTION + second * RIGHT))
                    for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ]
            )
            numerical = np.column_stack(
                [
                    (corners[0] + corners[1] - corners[2] - corners[3]) / (4 * step),
                    (corners[0] - corners[1] + corners[2] - corners[3]) / (4 * step),
                    (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * step**2),
                ]
            )
            primals.append(corners)
            errors.append(abs(actual[:, 1:] - numerical))
        np.savez_compressed(
            tmp_path / "frozen-ecp-mixed.npz",
            centers=CENTERS,
            weights=weights,
            left=MOTION,
            right=RIGHT,
            native=actual,
            primal_baseline=baseline,
            primal_corners=primals,
            steps=steps,
            errors=errors,
            angular=angular,
            spherical=spherical,
        )
        np.testing.assert_allclose(actual[:, 1:], numerical, atol=3e-8, rtol=2e-6)
        assert np.max(errors[-1]) < max(np.max(errors[0]) / 8, 5e-9)
        assert work[0] <= limits[1] and work[8] <= limits[0]
        assert work[1:4] == (0, 0, 0) and work[4] > 0 and work[9] > 3
        record_testsuite_property(
            f"ecp_mixed_l{angular}_spherical{spherical}",
            json.dumps(
                {
                    "errors": np.array(errors).tolist(),
                    "work": work,
                    "limits": limits.tolist(),
                }
            ),
        )


def test_bilinearity_ordered_weights_translation_and_alias_publication(
    mixed_native: Any,
) -> None:
    with _owner(mixed_native) as (handle, sizes, _):
        weights = _weights(sizes[0])
        status, baseline, work = _apply(mixed_native, handle, weights)
        assert status == 0, mixed_native.ecp_direction_error().decode()
        status, swapped, _ = _apply(mixed_native, handle, weights, RIGHT, MOTION)
        assert status == 0
        np.testing.assert_allclose(
            swapped, baseline[:, [0, 2, 1, 3]], atol=2e-13, rtol=2e-12
        )
        status, scaled, _ = _apply(mixed_native, handle, weights, -1.7 * MOTION, RIGHT)
        assert status == 0
        np.testing.assert_allclose(
            scaled, baseline * [1, -1.7, 1, -1.7], atol=2e-13, rtol=2e-12
        )
        status, rigid, _ = _apply(
            mixed_native, handle, weights, np.tile([0.2, -0.1, 0.3], (3, 1))
        )
        assert status == 0
        np.testing.assert_array_equal(rigid[:, [1, 3]], 0)
        for weight in (np.zeros_like(weights), weights - weights.T):
            status, empty, observed = _apply(mixed_native, handle, weight)
            assert status == 0
            np.testing.assert_array_equal(empty, 0)
            assert observed[:8] == work[:8]
        output = weights.reshape(-1)[:8].reshape(2, 4)
        status, alias, _ = _apply(mixed_native, handle, weights, output=output)
        assert status == 0
        np.testing.assert_array_equal(alias, baseline)


def test_limits_owner_and_bad_frames_never_publish_partial_scalars(
    mixed_native: Any,
) -> None:
    with _owner(mixed_native) as (handle, sizes, _):
        weights = _weights(sizes[0])
        limits = _plan(mixed_native, handle)
        status, baseline, work = _apply(mixed_native, handle, weights)
        assert status == 0
        for axis in range(2):
            bounded = limits.copy()
            bounded[axis] -= 1
            status, unchanged, _ = _apply(mixed_native, handle, weights, limits=bounded)
            assert status != 0
            np.testing.assert_array_equal(unchanged, 817)
        for at, epoch in ((1, 0), (3, 0), (work[9], 0), (0, 1)):
            mixed_native.ecp_direction_failure(handle, at, epoch)
            status, unchanged, _ = _apply(mixed_native, handle, weights)
            assert status != 0 and b"stale" in mixed_native.ecp_direction_error()
            np.testing.assert_array_equal(unchanged, 817)
            mixed_native.ecp_direction_failure(handle, 0, 0)
            status, recovered, _ = _apply(mixed_native, handle, weights)
            assert status == 0
            np.testing.assert_array_equal(recovered, baseline)
        bad_weights, bad_left, bad_right = weights.copy(), MOTION.copy(), RIGHT.copy()
        bad_weights.flat[-1], bad_left.flat[-1], bad_right.flat[-1] = (
            np.nan,
            np.inf,
            -np.inf,
        )
        for options in (
            {"weights": bad_weights},
            {"left": bad_left},
            {"right": bad_right},
            {"weights": np.full_like(weights, 1e308)},
            {"counts": (weights.size - 1, MOTION.size, RIGHT.size)},
            {"counts": (weights.size, MOTION.size - 1, RIGHT.size)},
            {"counts": (weights.size, MOTION.size, RIGHT.size - 1)},
            {"missing_validator": True},
        ):
            status, unchanged, _ = _apply(
                mixed_native, handle, **{"weights": weights, **options}
            )
            assert status != 0
            np.testing.assert_array_equal(unchanged, 817)
        status, recovered, _ = _apply(mixed_native, handle, weights)
        assert status == 0
        np.testing.assert_array_equal(recovered, baseline)


@pytest.mark.parametrize("atoms", [25, 40])
def test_basis_free_atoms_do_not_allocate_a_coordinate_hessian(
    mixed_native: Any, atoms: int
) -> None:
    """Only source-retention scaling is tested, not a complete electronic endpoint."""
    with _owner(mixed_native) as (handle, sizes, _):
        weights = _weights(sizes[0])
        original_plan = _plan(mixed_native, handle)
        status, baseline, _ = _apply(mixed_native, handle, weights)
        assert status == 0
    centers = np.vstack([CENTERS, np.arange((atoms - 3) * 3).reshape(-1, 3) + 30.0])
    left = np.vstack([MOTION, np.zeros((atoms - 3, 3))])
    right = np.vstack([RIGHT, np.zeros((atoms - 3, 3))])
    with _owner(mixed_native, centers=centers) as (handle, _, _):
        np.testing.assert_array_equal(_plan(mixed_native, handle), original_plan)
        status, actual, _ = _apply(mixed_native, handle, weights, left, right)
        assert status == 0
        np.testing.assert_array_equal(actual, baseline)


@pytest.mark.parametrize("empty", [False, True])
def test_unsupported_operator_never_implies_a_checked_grid_second_source(
    mixed_native: Any, empty: bool
) -> None:
    options = {"terms": np.empty((0, 5))} if empty else {"general": True}
    with _owner(mixed_native, **options) as (handle, sizes, _):
        plan = np.full(2, 817, dtype=np.uintp)
        assert (
            mixed_native.ecp_mixed_plan(handle, plan.ctypes.data_as(SIZE_POINTER)) != 0
        )
        np.testing.assert_array_equal(plan, 817)
        status, unchanged, _ = _apply(
            mixed_native,
            handle,
            _weights(sizes[0]),
            limits=np.full(2, 1 << 30, dtype=np.uintp),
        )
        assert status != 0 and b"outside Gaussian" in mixed_native.ecp_direction_error()
        np.testing.assert_array_equal(unchanged, 817)
