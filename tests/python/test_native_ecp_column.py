"""Independent primal controls for complete native residual-ECP curvature columns."""

from __future__ import annotations

import ctypes as ct
import time
from typing import TYPE_CHECKING, Any, cast

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
from test_native_ecp_mixed import RIGHT, _weights

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def column_native(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Any:
    """Bind the compiled core, not a test copy of the production ECP source."""
    builder = cast("Any", native_fixture)
    loaded = builder.__wrapped__(tmp_path_factory, required_native_cxx, policy_library)
    loaded.ecp_column_plan.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.ecp_column_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        SIZE_POINTER,
        POINTER,
        ct.c_size_t,
        ct.c_int,
    ]
    loaded.ecp_column_mutation.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.ecp_column_response.argtypes = (
        [ct.c_void_p]
        + [POINTER, ct.c_size_t] * 3
        + [SIZE_POINTER, POINTER, ct.c_size_t, ct.c_int]
    )
    return loaded


def _plan(native: Any, handle: Any) -> np.ndarray:
    """Retain the actual structural backend and five mandatory source caps."""
    result = np.zeros(6, dtype=np.uintp)
    assert native.ecp_column_plan(handle, result.ctypes.data_as(SIZE_POINTER)) == 0, (
        native.ecp_direction_error().decode()
    )
    return result


def _apply(
    native: Any, handle: Any, weights: np.ndarray, right: np.ndarray = RIGHT
) -> tuple[np.ndarray, np.ndarray]:
    """Consume one right leg without a source replay for any left coordinate."""
    plan = _plan(native, handle)
    output = np.full((2, 2 + 2 * right.size), -781.25)
    assert (
        native.ecp_column_apply(
            handle,
            _pointer(weights),
            weights.size,
            _pointer(right),
            right.size,
            plan[1:].ctypes.data_as(SIZE_POINTER),
            _pointer(output),
            output.size,
            0,
        )
        == 0
    ), native.ecp_direction_error().decode()
    work = np.zeros(10, dtype=np.uintp)
    native.ecp_direction_work(handle, work.ctypes.data_as(SIZE_POINTER))
    assert work[8] <= plan[1]
    assert np.all(work[:4] <= plan[2:])
    return output, work


def _response(
    native: Any,
    handle: Any,
    weights: np.ndarray,
    moving: np.ndarray,
    right: np.ndarray = RIGHT,
    output: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Supply an explicit right weight JVP, not a certified SCF response."""
    plan = _plan(native, handle)
    if output is None:
        output = np.full((2, 2 + 2 * right.size), -781.25)
    status = native.ecp_column_response(
        handle,
        _pointer(weights),
        weights.size,
        _pointer(moving),
        moving.size,
        _pointer(right),
        right.size,
        plan[1:].ctypes.data_as(SIZE_POINTER),
        _pointer(output),
        output.size,
        0,
    )
    work = np.zeros(10, dtype=np.uintp)
    if not status:
        native.ecp_direction_work(handle, work.ctypes.data_as(SIZE_POINTER))
        assert work[8] <= plan[1]
        assert np.all(work[:4] <= plan[2:])
    return status, output, work


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("spherical", [False, True])
def test_supplied_weight_column_against_independent_primal(
    column_native: Any, spherical: bool, general: bool, tmp_path: Path
) -> None:
    """Right-only weight motion contributes once, with no extra integral sweep."""
    native = column_native
    with _owner(native, angular=3, spherical=spherical, general=general) as (
        handle,
        sizes,
        terms,
    ):
        weights = _weights(sizes[0])
        moving = np.ascontiguousarray(
            np.random.default_rng(2822).normal(scale=0.014, size=weights.shape)
        )
        status, actual, work = _response(native, handle, weights, moving)
        assert status == 0, native.ecp_direction_error().decode()
        frozen, frozen_work = _apply(native, handle, weights)
        np.testing.assert_array_equal(work[:8], frozen_work[:8])
        np.testing.assert_array_equal(actual[:, 2:11], frozen[:, 2:11])
        status, density_only, _ = _response(
            native, handle, weights, moving, np.zeros_like(RIGHT)
        )
        assert status == 0
        np.testing.assert_allclose(
            actual[:, 1], frozen[:, 1] + density_only[:, 1], atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            actual[:, 11:],
            frozen[:, 11:] + density_only[:, 11:],
            atol=3e-12,
            rtol=3e-11,
        )
        steps = np.array([1e-2, 3e-3, 1e-3])
        legs = np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
        baseline = np.einsum(
            "pij,ij->p", _oracle(CENTERS, terms, 3, spherical), weights
        )
        corners, left_corners, right_corners = [], [], []
        for step in steps:
            rows, left_rows = [], []
            for leg in legs:
                rows.append(
                    [
                        np.einsum(
                            "pij,ij->p",
                            _oracle(
                                CENTERS + step * (first * leg + second * RIGHT),
                                terms,
                                3,
                                spherical,
                            ),
                            weights + second * step * moving,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                )
                left_rows.append(
                    [
                        np.einsum(
                            "pij,ij->p",
                            _oracle(CENTERS + sign * step * leg, terms, 3, spherical),
                            weights,
                        )
                        for sign in (1, -1)
                    ]
                )
            corners.append(rows)
            left_corners.append(left_rows)
            right_corners.append(
                [
                    np.einsum(
                        "pij,ij->p",
                        _oracle(CENTERS + sign * step * RIGHT, terms, 3, spherical),
                        weights + sign * step * moving,
                    )
                    for sign in (1, -1)
                ]
            )
        corners = np.asarray(corners, dtype=np.float64)
        left_corners = np.asarray(left_corners, dtype=np.float64)
        right_corners = np.asarray(right_corners, dtype=np.float64)
        gradient = (left_corners[:, :, 0] - left_corners[:, :, 1]) / (
            2 * steps[:, None, None]
        )
        mixed = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None, None] ** 2)
        right_control = (right_corners[:, 0] - right_corners[:, 1]) / (
            2 * steps[:, None]
        )
        expected = np.concatenate(
            (right_control[:, None, :], gradient, mixed), axis=1
        ).transpose(0, 2, 1)
        errors = np.abs(expected - actual[None, :, 1:])
        np.savez_compressed(
            tmp_path / "ecp-supplied-weight-column-primal.npz",
            native=actual,
            weights=weights,
            weights_right=moving,
            centers=CENTERS,
            right=RIGHT,
            terms=terms,
            spherical=spherical,
            general=general,
            steps=steps,
            primal_baseline=baseline,
            primal_corners=corners,
            left_corners=left_corners,
            right_corners=right_corners,
            errors=errors,
            plan=_plan(native, handle),
            work=work,
            frozen_work=frozen_work,
        )
        np.testing.assert_allclose(actual[:, 0], baseline, atol=2e-11, rtol=2e-10)
        np.testing.assert_allclose(actual[:, 1:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors


@pytest.mark.parametrize("general", [False, True])
def test_supplied_weight_column_refusal_alias_mutation_and_recovery(
    column_native: Any, general: bool
) -> None:
    """The native snapshots, not a test-side input validator, reject actual mutations."""
    native = column_native
    with _owner(native, angular=1, general=general) as (handle, sizes, _):
        weights = _weights(sizes[0])
        moving = -0.07 * weights
        status, baseline, work = _response(native, handle, weights, moving)
        assert status == 0, native.ecp_direction_error().decode()
        for malformed in (
            moving.ravel()[:-1],
            np.full_like(moving, np.nan),
            np.full_like(moving, np.inf),
        ):
            status, output, _ = _response(native, handle, weights, malformed)
            assert status != 0
            np.testing.assert_array_equal(output, -781.25)
        output = np.full_like(baseline, -781.25)
        plan = _plan(native, handle)
        assert (
            native.ecp_column_response(
                handle,
                _pointer(weights),
                weights.size,
                POINTER(),
                moving.size,
                _pointer(RIGHT),
                RIGHT.size,
                plan[1:].ctypes.data_as(SIZE_POINTER),
                _pointer(output),
                output.size,
                0,
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)
        frozen, _ = _apply(native, handle, weights)
        status, actual, _ = _response(native, handle, weights, np.empty(0))
        assert status == 0
        np.testing.assert_array_equal(actual, frozen)
        for index in range(3):
            inputs = [weights.copy(), moving.copy(), RIGHT.copy()]
            shared = np.full(
                max(baseline.size, *(value.size for value in inputs)), -781.25
            )
            shape = inputs[index].shape
            shared[: inputs[index].size] = inputs[index].ravel()
            inputs[index] = shared[: inputs[index].size].reshape(shape)
            status, output, _ = _response(
                native,
                handle,
                *inputs,
                output=shared[: baseline.size].reshape(baseline.shape),
            )
            assert status == 0, native.ecp_direction_error().decode()
            np.testing.assert_array_equal(output, baseline)
        for mode, index in ((1, 0), (2, 2), (3, 1)):
            inputs = [weights.copy(), moving.copy(), RIGHT.copy()]
            native.ecp_column_mutation(handle, mode, int(work[9]))
            status, output, _ = _response(native, handle, *inputs)
            assert status != 0
            np.testing.assert_array_equal(output, -781.25)
            native.ecp_column_mutation(handle, 0, 0)
            status, recovered, _ = _response(native, handle, weights, moving)
            assert status == 0
            np.testing.assert_array_equal(recovered, baseline)
            assert inputs[index].flat[0] != (weights, moving, RIGHT)[index].flat[0]


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("angular", range(4))
def test_complete_ecp_column_against_independent_primal_operators(
    column_native: Any, angular: int, spherical: bool, general: bool, tmp_path: Path
) -> None:
    """All physical gradients and mixed components retain both residual parts."""
    native = column_native
    with _owner(native, angular=angular, spherical=spherical, general=general) as (
        handle,
        sizes,
        terms,
    ):
        weights = _weights(sizes[0])
        plan = _plan(native, handle)
        assert plan[0] == (2 if general else 1)
        actual, work = _apply(native, handle, weights)
        steps = np.array([1e-2, 3e-3, 1e-3])
        legs = np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
        primal = np.einsum(
            "pij,ij->p", _oracle(CENTERS, terms, angular, spherical), weights
        )
        np.testing.assert_allclose(actual[:, 0], primal, atol=2e-11, rtol=2e-10)
        corners = np.array(
            [
                [
                    [
                        np.einsum(
                            "pij,ij->p",
                            _oracle(
                                CENTERS + step * (first * leg + second * RIGHT),
                                terms,
                                angular,
                                spherical,
                            ),
                            weights,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    for leg in legs
                ]
                for step in steps
            ]
        )
        gradients = (
            corners[:, :, 0] + corners[:, :, 1] - corners[:, :, 2] - corners[:, :, 3]
        ) / (4 * steps[:, None, None])
        mixed = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None, None] ** 2)
        expected = np.concatenate((gradients, mixed), axis=1).transpose(0, 2, 1)
        errors = np.abs(expected - actual[None, :, 2:])
        np.testing.assert_allclose(actual[:, 2:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
        np.testing.assert_allclose(
            actual[:, 2:11] @ RIGHT.ravel(), actual[:, 1], atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            actual[:, 2:11].reshape(2, 3, 3).sum(axis=1), 0, atol=3e-12
        )
        np.testing.assert_allclose(
            actual[:, 11:].reshape(2, 3, 3).sum(axis=1), 0, atol=3e-12
        )
        np.savez_compressed(
            tmp_path / "ecp-column-primal-control.npz",
            native=actual,
            weights=weights,
            centers=CENTERS,
            right=RIGHT,
            terms=terms,
            angular=angular,
            spherical=spherical,
            general=general,
            steps=steps,
            primal_baseline=primal,
            primal_corners=corners,
            errors=errors,
            plan=plan,
            work=work,
        )


@pytest.mark.parametrize("general", [False, True])
def test_ecp_column_raw_symmetry_linearity_empty_and_rigid_motion(
    column_native: Any, general: bool, tmp_path: Path
) -> None:
    """Matrix columns retain raw bilinear symmetry without a symmetry repair."""
    native = column_native
    with _owner(native, angular=0, general=general) as (handle, sizes, _):
        weights = _weights(sizes[0])
        actual, _ = _apply(native, handle, weights)
        first, _ = _apply(native, handle, weights, MOTION)
        combined, _ = _apply(
            native, handle, weights, np.ascontiguousarray(2 * MOTION - RIGHT)
        )
        np.testing.assert_allclose(
            combined[:, 11:], 2 * first[:, 11:] - actual[:, 11:], atol=3e-12, rtol=3e-11
        )
        rigid, _ = _apply(native, handle, weights, np.tile([0.13, -0.07, 0.11], (3, 1)))
        np.testing.assert_array_equal(rigid[:, 1], 0)
        np.testing.assert_array_equal(rigid[:, 11:], 0)
        np.testing.assert_array_equal(rigid[:, 0], actual[:, 0])
        np.testing.assert_array_equal(rigid[:, 2:11], actual[:, 2:11])
        empty, _ = _apply(native, handle, np.zeros_like(weights))
        np.testing.assert_array_equal(empty, 0)
        legs = np.eye(9).reshape(-1, 3, 3)
        matrix = np.array(
            [_apply(native, handle, weights, leg)[0][:, 11:] for leg in legs]
        ).transpose(1, 2, 0)
        np.testing.assert_allclose(
            matrix, matrix.transpose(0, 2, 1), atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            matrix @ RIGHT.ravel(), actual[:, 11:], atol=3e-12, rtol=3e-11
        )
        np.savez_compressed(
            tmp_path / "ecp-column-raw-matrix.npz",
            matrix=matrix,
            native=actual,
            weights=weights,
            centers=CENTERS,
            right=RIGHT,
        )


@pytest.mark.parametrize("general", [False, True])
def test_ecp_column_caps_failure_publication_and_authenticated_recovery(
    column_native: Any, general: bool
) -> None:
    """Every source cap, final owner check and borrowed input mutation is enforced."""
    native = column_native
    with _owner(native, angular=0, general=general) as (handle, sizes, _):
        weights, right = _weights(sizes[0]), RIGHT.copy()
        plan = _plan(native, handle)
        baseline, work = _apply(native, handle, weights, right)

        def rejected(
            limits: np.ndarray = plan[1:],
            missing: int = 0,
            weight_count: int = weights.size,
            right_count: int = right.size,
            output_count: int = baseline.size,
        ) -> None:
            output = np.full_like(baseline, -781.25)
            assert (
                native.ecp_column_apply(
                    handle,
                    _pointer(weights),
                    weight_count,
                    _pointer(right),
                    right_count,
                    limits.ctypes.data_as(SIZE_POINTER),
                    _pointer(output),
                    output_count,
                    missing,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)

        for index in np.flatnonzero(plan[1:]):
            limits = plan[1:].copy()
            limits[index] -= 1
            rejected(limits=limits)
        rejected(missing=1)
        rejected(weight_count=weights.size - 1)
        rejected(right_count=right.size - 1)
        rejected(output_count=baseline.size - 1)
        for argument in (1, 3, 5, 6):
            output = np.full_like(baseline, -781.25)
            arguments = [
                handle,
                _pointer(weights),
                weights.size,
                _pointer(right),
                right.size,
                plan[1:].ctypes.data_as(SIZE_POINTER),
                _pointer(output),
                output.size,
                0,
            ]
            arguments[argument] = None
            assert native.ecp_column_apply(*arguments) != 0
            np.testing.assert_array_equal(output, -781.25)
        for values in (weights, right):
            saved = values.copy()
            values.flat[0] = np.nan
            rejected()
            values[:] = saved
        for at in (1, 4, int(work[9]) - 1, int(work[9])):
            native.ecp_direction_failure(handle, at, 0)
            rejected()
            native.ecp_direction_failure(handle, 0, 0)
            np.testing.assert_array_equal(
                _apply(native, handle, weights, right)[0], baseline
            )
        native.ecp_direction_failure(handle, 0, 1)
        rejected()
        native.ecp_direction_failure(handle, 0, 0)
        for mode, values in ((1, weights), (2, right)):
            saved = values.copy()
            native.ecp_column_mutation(handle, mode, int(work[9]))
            rejected()
            values[:] = saved
            native.ecp_column_mutation(handle, 0, 0)
            np.testing.assert_array_equal(
                _apply(native, handle, weights, right)[0], baseline
            )


@pytest.mark.parametrize("general", [False, True])
def test_ecp_column_source_work_does_not_repeat_for_spectator_coordinates(
    column_native: Any, general: bool, tmp_path: Path
) -> None:
    """Expand to 32 physical atoms without repeating immutable AO/projector work."""
    native = column_native
    records = []
    for atoms in (3, 32):
        centers = np.vstack(
            (
                CENTERS,
                np.column_stack(
                    (
                        10 + 2 * np.arange(atoms - 3),
                        np.ones(atoms - 3),
                        np.zeros(atoms - 3),
                    )
                ),
            )
        )
        with _owner(native, angular=0, general=general, centers=centers) as (
            handle,
            sizes,
            _,
        ):
            right = np.zeros_like(centers)
            right[:3] = RIGHT
            start = time.perf_counter()
            actual, work = _apply(native, handle, _weights(sizes[0]), right)
            elapsed = time.perf_counter() - start
            records.append((actual, work, _plan(native, handle), elapsed))
            np.testing.assert_array_equal(
                actual[:, 2 : 2 + 3 * atoms].reshape(2, atoms, 3)[:, 3:], 0
            )
            np.testing.assert_array_equal(
                actual[:, 2 + 3 * atoms :].reshape(2, atoms, 3)[:, 3:], 0
            )
    np.testing.assert_array_equal(records[0][1][:8], records[1][1][:8])
    np.testing.assert_array_equal(records[0][0][:, :11], records[1][0][:, :11])
    np.testing.assert_array_equal(records[0][0][:, 11:], records[1][0][:, 98:107])
    np.savez_compressed(
        tmp_path / "ecp-column-source-reuse.npz",
        small=records[0][0],
        large=records[1][0],
        work=np.array([record[1] for record in records]),
        plans=np.array([record[2] for record in records]),
        seconds=np.array([record[3] for record in records]),
    )


def test_ecp_column_empty_operator_has_no_hidden_source_work(
    column_native: Any,
) -> None:
    """The empty backend remains explicit and returns complete zero columns."""
    native = column_native
    with _owner(native, terms=np.empty((0, 5))) as (handle, sizes, _):
        actual, work = _apply(native, handle, _weights(sizes[0]))
        assert _plan(native, handle)[0] == 0
        np.testing.assert_array_equal(actual, 0)
        np.testing.assert_array_equal(work[:8], 0)


def test_raw_ecp_second_gate_cannot_be_hidden_by_zero_weights_or_motion(
    column_native: Any, tmp_path: Path
) -> None:
    """Real first-order-qualified but second-order-unqualified grids still refuse."""
    native = column_native
    for exponent_scale in (10.0, 40.0, 160.0, 640.0):
        with _owner(native, angular=0, general=True, exponent_scale=exponent_scale) as (
            handle,
            _,
            terms,
        ):
            gaps = np.zeros(4)
            assert native.ecp_direction_gate_gaps(handle, _pointer(gaps)) == 0
        primal, gradient = np.max(gaps[:2]), np.max(gaps[2:])
        if not primal > 0 or not gradient > 0:
            continue
        terms = terms.copy()
        scale = min(2e-9 / primal, 2e-8 / gradient) * 0.75
        terms[:, 4] *= scale
        with _owner(native, angular=0, terms=terms, exponent_scale=exponent_scale) as (
            handle,
            sizes,
            _,
        ):
            gaps = np.zeros(4)
            assert native.ecp_direction_gate_gaps(handle, _pointer(gaps)) == 0
            assert np.max(gaps[:2]) < 2e-9 and np.max(gaps[2:]) < 2e-8
            weights = np.zeros((sizes[0], sizes[0]))
            right = np.zeros_like(RIGHT)
            output = np.full((2, 20), -781.25)
            plan = _plan(native, handle)
            status = native.ecp_column_apply(
                handle,
                _pointer(weights),
                weights.size,
                _pointer(right),
                right.size,
                plan[1:].ctypes.data_as(SIZE_POINTER),
                _pointer(output),
                output.size,
                0,
            )
            if status == 0:
                continue
            assert b"order-2" in native.ecp_direction_error(), (
                native.ecp_direction_error()
            )
            np.testing.assert_array_equal(output, -781.25)
            np.savez_compressed(
                tmp_path / "ecp-column-unprojected-second-gate.npz",
                terms=terms,
                exponent_scale=exponent_scale,
                coefficient_scale=scale,
                first_order_gaps=gaps,
                weights=weights,
                right=right,
            )
            return
    pytest.fail("no real first-order-qualified second-order grid refusal was exercised")
