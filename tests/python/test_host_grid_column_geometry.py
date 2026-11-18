"""Independent primal controls for one-pass moving-grid curvature columns."""

from __future__ import annotations

import ctypes as ct
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from test_host_grid_mixed_geometry import (
    CENTERS,
    LEFT,
    POINTER,
    RIGHT,
    SIZE_POINTER,
    _decimal_weights,
    _grid,
    _owner,
    _pointer,
)
from test_host_grid_mixed_geometry import native as native_fixture

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def column_native(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Any:
    """Bind the actual compiled scalar-Graph/native source, with mandatory tools."""
    builder = cast("Any", native_fixture)
    loaded = builder.__wrapped__(tmp_path_factory, required_native_cxx, policy_library)
    loaded.host_grid_column_plan.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_grid_column_bind.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_grid_column_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_grid_column_read.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t] + [
        kind for _ in range(3) for kind in (POINTER, ct.c_size_t)
    ]
    return loaded


def _plan(native: Any, handle: Any) -> np.ndarray:
    result = np.zeros(5, dtype=np.uintp)
    native.host_grid_column_plan(handle, result.ctypes.data_as(SIZE_POINTER))
    return result


def _work(native: Any, handle: Any) -> np.ndarray:
    result = np.zeros(7, dtype=np.uintp)
    native.host_grid_column_work(handle, result.ctypes.data_as(SIZE_POINTER))
    return result


def _column(
    native: Any,
    handle: Any,
    shape: tuple,
    begin: int,
    count: int,
    right: np.ndarray = RIGHT,
) -> tuple[int, np.ndarray, np.ndarray]:
    jets = np.full((20, count, shape[1]), -9713.5)
    weights = np.full((2 + 6 * shape[0], count), -9713.5)
    status = native.host_grid_column_read(
        handle,
        begin,
        count,
        _pointer(right),
        right.size,
        _pointer(jets),
        jets.size,
        _pointer(weights),
        weights.size,
    )
    return status, jets, weights


def _spatial_oracle(
    centers: np.ndarray, points: np.ndarray, cartesian: bool
) -> np.ndarray:
    """Normalized order-three libcint spatial jets, not production motion jets."""
    from pyscf import gto
    from pyscf.dft import numint

    basis = [
        [degree, [0.7 + 0.13 * degree, 0.7], [0.22 + 0.04 * degree, 0.3]]
        for degree in range(4)
    ]
    molecule = gto.M(
        atom=[("H", center) for center in centers],
        basis={"H": basis},
        spin=len(centers) % 2,
        cart=cartesian,
        unit="Bohr",
        verbose=0,
    )
    normalization = 1 / np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    return numint.eval_ao(molecule, points, deriv=3) * normalization


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("cartesian", [False, True])
def test_complete_partition_columns_against_independent_primal_controls(
    column_native: Any,
    version: int,
    cartesian: bool,
    tmp_path: Path,
) -> None:
    """All physical coordinates use 70-digit displaced primal partitions."""
    native = column_native
    with _owner(native, version=version, cartesian=cartesian) as (handle, shape):
        points, retained, atomic, owners, radii = _grid(native, handle, shape)
        candidates = np.flatnonzero(
            (retained / atomic > 0.1) & (retained / atomic < 0.9)
        )
        assert candidates.size
        count = 3
        begin = min(int(candidates[len(candidates) // 2]), shape[2] - count)
        section = slice(begin, begin + count)
        status, jets, actual = _column(native, handle, shape, begin, count)
        assert status == 0, native.host_mixed_error().decode()
        np.testing.assert_array_equal(actual[0], retained[section])
        np.testing.assert_allclose(
            jets,
            _spatial_oracle(CENTERS, points[section], cartesian),
            atol=3e-13,
            rtol=3e-12,
        )
        steps = np.array([3e-3, 1e-3, 3e-4])
        legs = np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
        controls = np.array(
            [
                [
                    _decimal_weights(
                        CENTERS,
                        points[section],
                        owners[section],
                        atomic[section],
                        radii if version == 4 else None,
                        3,
                        leg,
                        RIGHT,
                        step,
                    )
                    for leg in legs
                ]
                for step in steps
            ]
        )
        expected = np.concatenate((controls[:, :, 0], controls[:, :, 2]), axis=1)
        errors = np.abs(expected - actual[None, 2:])
        np.testing.assert_allclose(actual[1], controls[-1, 0, 1], atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(actual[2:], expected[-1], atol=3e-7, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 12, 2e-9)
        np.testing.assert_allclose(
            RIGHT.ravel() @ actual[2:11], actual[1], atol=3e-13, rtol=3e-12
        )
        work, plan = _work(native, handle), _plan(native, handle)
        assert work[2] == count and work[3] == count * shape[1]
        assert work[4] <= plan[3] and work[5] + work[6] <= plan[4]
        np.savez_compressed(
            tmp_path / "grid-column-primal-control.npz",
            centers=CENTERS,
            points=points[section],
            owners=owners[section],
            atomic=atomic[section],
            radii=radii,
            version=version,
            cartesian=cartesian,
            right=RIGHT,
            steps=steps,
            expected=expected,
            errors=errors,
            native=actual,
            jets=jets,
            work=work,
            plan=plan,
        )


@pytest.mark.parametrize("iterations", [1, 2, 4, 5])
def test_partition_iteration_columns_preserve_prescription(
    column_native: Any,
    iterations: int,
) -> None:
    native = column_native
    with _owner(native, version=4, maximum=0, iterations=iterations) as (handle, shape):
        points, _, atomic, owners, radii = _grid(native, handle, shape)
        status, _, actual = _column(native, handle, shape, 100, 2)
        assert status == 0, native.host_mixed_error().decode()
        reference = _decimal_weights(
            CENTERS,
            points[100:102],
            owners[100:102],
            atomic[100:102],
            radii,
            iterations,
            LEFT,
            RIGHT,
            1e-4,
        )
        np.testing.assert_allclose(
            LEFT.ravel() @ actual[2:11], reference[0], atol=3e-8, rtol=2e-6
        )
        np.testing.assert_allclose(
            LEFT.ravel() @ actual[11:], reference[2], atol=3e-8, rtol=2e-6
        )


def test_raw_matrix_symmetry_linearity_rigid_motion_and_partial_tiles(
    column_native: Any,
    tmp_path: Path,
) -> None:
    native = column_native
    with _owner(native, maximum=0, tile=7) as (handle, shape):
        status, _, baseline = _column(native, handle, shape, 30, 3)
        assert status == 0
        legs = np.eye(9).reshape(-1, 3, 3)
        matrix = np.array(
            [_column(native, handle, shape, 30, 3, leg)[2][11:] for leg in legs]
        )
        np.testing.assert_allclose(
            matrix, matrix.transpose(1, 0, 2), atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            np.einsum("ijp,i->jp", matrix, RIGHT.ravel()),
            baseline[11:],
            atol=3e-12,
            rtol=3e-11,
        )
        rigid = np.tile([0.13, -0.07, 0.11], (3, 1))
        status, _, translated = _column(native, handle, shape, 30, 3, rigid)
        assert status == 0
        np.testing.assert_array_equal(translated[1], 0)
        np.testing.assert_array_equal(translated[11:], 0)
        np.testing.assert_array_equal(translated[2:11], baseline[2:11])
        status, _, first = _column(native, handle, shape, 30, 3, LEFT)
        assert status == 0
        status, _, combined = _column(native, handle, shape, 30, 3, 2 * LEFT - RIGHT)
        assert status == 0
        np.testing.assert_allclose(
            combined[11:], 2 * first[11:] - baseline[11:], atol=3e-12, rtol=3e-11
        )
        _, _, single = _column(native, handle, shape, 31, 1)
        np.testing.assert_array_equal(single[:, 0], baseline[:, 1])
        status, _, zero = _column(native, handle, shape, shape[2], 0)
        assert status == 0 and zero.size == 0
        np.savez_compressed(
            tmp_path / "grid-column-raw-matrix.npz",
            matrix=matrix,
            native=baseline,
            right=RIGHT,
        )


def test_refusals_owner_lifetime_caps_aliases_and_recovery(column_native: Any) -> None:
    native = column_native
    with _owner(native, maximum=0) as (handle, shape):
        status, _, baseline = _column(native, handle, shape, 30, 3)
        assert status == 0
        plan = _plan(native, handle)
        for index in range(len(plan)):
            limits = plan.copy()
            limits[index] -= 1
            assert (
                native.host_grid_column_bind(
                    handle, limits.ctypes.data_as(SIZE_POINTER)
                )
                != 0
            )
        for mode in (4, 5, 6):
            start = _work(native, handle)[2]
            native.host_mixed_fault(handle, mode, int(start + 2))
            status, jets, weights = _column(native, handle, shape, 30, 3)
            assert status != 0
            np.testing.assert_array_equal(jets, -9713.5)
            np.testing.assert_array_equal(weights, -9713.5)
            native.host_mixed_fault(handle, 0, 0)
            np.testing.assert_array_equal(
                _column(native, handle, shape, 30, 3)[2], baseline
            )
        direction = RIGHT.copy()
        start = _work(native, handle)[2]
        native.host_mixed_fault(handle, 7, int(start + 3))
        status, jets, weights = _column(native, handle, shape, 30, 3, direction)
        assert status != 0 and b"direction changed" in native.host_mixed_error()
        np.testing.assert_array_equal(jets, -9713.5)
        np.testing.assert_array_equal(weights, -9713.5)
        np.testing.assert_array_equal(
            _column(native, handle, shape, 30, 3)[2], baseline
        )
        for kind in range(4):
            borrowed = native.host_mixed_borrowed(handle, kind)
            jets = np.full((20, 3, shape[1]), -9713.5)
            assert (
                native.host_grid_column_read(
                    handle,
                    30,
                    3,
                    _pointer(RIGHT),
                    RIGHT.size,
                    _pointer(jets),
                    jets.size,
                    borrowed,
                    60,
                )
                != 0
            )
            np.testing.assert_array_equal(jets, -9713.5)
        jets = np.full((20, 3, shape[1]), -9713.5)
        alias = np.full((20, 3), -9713.5)
        alias.ravel()[:9] = RIGHT.ravel()
        assert (
            native.host_grid_column_read(
                handle,
                30,
                3,
                _pointer(alias),
                9,
                _pointer(jets),
                jets.size,
                _pointer(alias),
                alias.size,
            )
            == 0
        )
        np.testing.assert_array_equal(alias, baseline)
        for begin, count, size in ((shape[2], 1, 9), (0, shape[3] + 1, 9), (30, 3, 8)):
            status, jets, weights = _column(
                native, handle, shape, begin, count, RIGHT.ravel()[:size]
            )
            assert status != 0
            np.testing.assert_array_equal(jets, -9713.5)
            np.testing.assert_array_equal(weights, -9713.5)
        invalid = RIGHT.copy()
        invalid[0, 0] = np.nan
        assert _column(native, handle, shape, 30, 3, invalid)[0] != 0
        jets = np.full((20, 3, shape[1]), -9713.5)
        weights = np.full((20, 3), -9713.5)
        for direction, ao, column in (
            (None, _pointer(jets), _pointer(weights)),
            (_pointer(RIGHT), None, _pointer(weights)),
            (_pointer(RIGHT), _pointer(jets), None),
            (_pointer(RIGHT), _pointer(jets), _pointer(jets)),
        ):
            assert (
                native.host_grid_column_read(
                    handle, 30, 3, direction, 9, ao, jets.size, column, weights.size
                )
                != 0
            )
            np.testing.assert_array_equal(jets, -9713.5)
            np.testing.assert_array_equal(weights, -9713.5)


def test_signed_measures_identity_and_single_atom(column_native: Any) -> None:
    native = column_native
    with _owner(native, version=4, maximum=0) as (handle, shape):
        _, _, atomic, _, _ = _grid(native, handle, shape)
        negative = np.flatnonzero(atomic < 0)
        assert negative.size
        begin = int(negative[0])
        assert _column(native, handle, shape, begin, 1)[0] == 0
        measures = native.host_mixed_borrowed(handle, 3)
        saved = measures[begin]
        measures[begin] *= 1.1
        assert _column(native, handle, shape, begin, 1)[0] != 0
        measures[begin] = saved
        assert _column(native, handle, shape, begin, 1)[0] == 0
    with _owner(native, maximum=0, centers=CENTERS[:1], numbers=(1,)) as (
        handle,
        shape,
    ):
        status, _, weights = _column(native, handle, shape, 3, 5, RIGHT[:1])
        assert status == 0
        np.testing.assert_array_equal(weights[1:], 0)
    coincident = CENTERS.copy()
    coincident[1] = coincident[0]
    with _owner(native, maximum=0, centers=coincident) as (handle, shape):
        status, jets, weights = _column(native, handle, shape, 3, 5)
        assert status != 0
        np.testing.assert_array_equal(jets, -9713.5)
        np.testing.assert_array_equal(weights, -9713.5)


@pytest.mark.parametrize("atoms", [3, 25, 40])
def test_source_work_is_not_replayed_for_each_left_coordinate(
    column_native: Any,
    atoms: int,
    tmp_path: Path,
) -> None:
    """Semantic AO/pair visits stay one source; this is not endpoint timing."""
    native = column_native
    centers = np.array(
        [
            [1.7 * (atom % 5), 1.9 * ((atom // 5) % 4), 2.1 * (atom // 20)]
            for atom in range(atoms)
        ]
    )
    right = np.arange(3 * atoms, dtype=float).reshape(atoms, 3) * 0.0001
    with _owner(native, maximum=0, centers=centers, numbers=(1,) * atoms, tile=3) as (
        handle,
        shape,
    ):
        start = time.perf_counter()
        status, _, actual = _column(native, handle, shape, 45, 3, right)
        elapsed = time.perf_counter() - start
        assert status == 0, native.host_mixed_error().decode()
        work, plan = _work(native, handle), _plan(native, handle)
        assert work[2] == 3 and work[3] == 3 * shape[1]
        assert work[4] == 4 * (atoms * (atoms - 1) // 2)
        assert work[5] + work[6] <= plan[4]
        assert plan[0] < 16 * 1024**2
        points, _, atomic, owners, _ = _grid(native, handle, shape)
        reference = _decimal_weights(
            centers,
            points[45:48],
            owners[45:48],
            atomic[45:48],
            None,
            3,
            -right,
            right,
            1e-4,
        )
        np.testing.assert_allclose(
            -right.ravel() @ actual[2 : 2 + 3 * atoms],
            reference[0],
            atol=3e-8,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            -right.ravel() @ actual[2 + 3 * atoms :], reference[2], atol=3e-8, rtol=2e-6
        )
        np.savez_compressed(
            tmp_path / "grid-column-source-work.npz",
            centers=centers,
            right=right,
            native=actual,
            plan=plan,
            work=work,
            seconds=elapsed,
        )
