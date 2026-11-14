"""Native analytic molecular weight motion with independent product oracles."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header
from test_grid_response import decimal_partition

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
CENTERS = np.array([[0.1, -0.2, 0.3], [1.7, 0.2, -0.4], [-0.5, 1.3, 0.7]])
MOTION = np.array([[0.2, -0.1, 0.3], [-0.1, 0.2, -0.15], [0.08, -0.04, 0.12]])


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    directory = tmp_path_factory.mktemp("host-grid-geometry")
    (directory / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    library = required_native_cxx.build_shared(
        [ROOT / "src/dft/grid.cpp", ROOT / "tests/native/host_grid_geometry_probe.cpp"],
        directory / "grid.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(directory),
        ],
    )
    loaded = ct.CDLL(str(library))
    loaded.host_grid_error.restype = ct.c_char_p
    loaded.host_grid_create.argtypes = [
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
        ct.POINTER(ct.c_int),
        POINTER,
    ]
    loaded.host_grid_create.restype = ct.c_void_p
    loaded.host_grid_destroy.argtypes = [ct.c_void_p]
    loaded.host_grid_size.argtypes = [ct.c_void_p]
    loaded.host_grid_size.restype = ct.c_size_t
    loaded.host_grid_workspace.argtypes = [ct.c_void_p, ct.c_size_t]
    loaded.host_grid_workspace.restype = ct.c_size_t
    loaded.host_grid_planned_pairs.argtypes = [ct.c_void_p, ct.c_size_t]
    loaded.host_grid_planned_pairs.restype = ct.c_size_t
    loaded.host_grid_export.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        POINTER,
        ct.POINTER(ct.c_uint32),
    ]
    loaded.host_grid_retained_weights.argtypes = [ct.c_void_p]
    loaded.host_grid_retained_weights.restype = POINTER
    loaded.host_grid_direction.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    loaded.host_grid_adjoint.argtypes = [ct.c_void_p, POINTER, POINTER]
    return loaded


@contextmanager
def _owner(
    native: Any,
    version: int,
    numbers: tuple = (1, 8, 6),
    centers: np.ndarray = CENTERS,
    iterations: int = 3,
) -> Any:
    numbers_array = np.array(numbers, dtype=np.int32)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    handle = native.host_grid_create(
        version,
        iterations,
        len(numbers),
        numbers_array.ctypes.data_as(ct.POINTER(ct.c_int)),
        _pointer(centers),
    )
    if not handle:
        raise RuntimeError(native.host_grid_error().decode())
    try:
        size = native.host_grid_size(handle)
        points, weights, raw = np.empty((size, 3)), np.empty(size), np.empty(size)
        owners = np.empty(size, dtype=np.uint32)
        native.host_grid_export(
            handle,
            _pointer(points),
            _pointer(weights),
            _pointer(raw),
            owners.ctypes.data_as(ct.POINTER(ct.c_uint32)),
        )
        yield handle, (points, weights, raw, owners)
    finally:
        native.host_grid_destroy(handle)


def _direction(
    native: Any,
    handle: int,
    direction: np.ndarray,
    begin: int,
    count: int,
    output: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
) -> tuple:
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    if output is None:
        output = np.full(count, -9182.5)
    if workspace is None:
        workspace = np.empty(native.host_grid_workspace(handle, count) // 8)
    work = (ct.c_size_t * 2)()
    status = native.host_grid_direction(
        handle,
        begin,
        count,
        _pointer(direction),
        direction.size,
        _pointer(output),
        output.size,
        _pointer(workspace),
        workspace.size,
        work,
    )
    if status:
        raise RuntimeError(native.host_grid_error().decode())
    return output, tuple(work)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_weight_direction_matches_independent_displaced_products_and_adjoint(
    native: Any,
    version: int,
    iterations: int,
    record_testsuite_property: Any,
) -> None:
    with _owner(native, version, iterations=iterations) as (handle, data):
        points, weights, raw, owners = data
        actual, work = _direction(native, handle, MOTION, 0, len(weights))
        assert work == (len(weights), len(weights) * 3)
        assert native.host_grid_planned_pairs(handle, len(weights)) == work[1]
        tiled = []
        for begin in range(0, len(weights), 97):
            count = min(97, len(weights) - begin)
            tiled.append(_direction(native, handle, MOTION, begin, count)[0])
        np.testing.assert_array_equal(np.concatenate(tiled), actual)
        sensitivity = np.exp(-0.2 * np.sum(points**2, axis=1))
        gradient = np.empty_like(CENTERS)
        assert (
            native.host_grid_adjoint(handle, _pointer(sensitivity), _pointer(gradient))
            == 0
        )
        assert sensitivity @ actual == pytest.approx(
            np.sum(gradient * MOTION), abs=3e-11, rel=3e-12
        )
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-12)
        translated = _direction(
            native, handle, np.tile([0.1, -0.2, 0.3], (3, 1)), 0, len(weights)
        )[0]
        np.testing.assert_allclose(translated, 0, atol=2e-12)
        selected = np.unique(np.linspace(0, len(weights) - 1, 17, dtype=int))
        radii = None
        if version == 4:
            table = dict(GridSpec.treutler().element_radii)
            radii = [Decimal(str(table[number])) for number in (1, 8, 6)]
        with localcontext() as context:
            context.prec = 60
            centers = [[Decimal(str(value)) for value in row] for row in CENTERS]
            motion = [[Decimal(str(value)) for value in row] for row in MOTION]
            for index in selected:
                point = [Decimal(str(value)) for value in points[index]]
                owner = int(owners[index])
                products = []
                step = Decimal("1e-16")
                for sign in (1, -1):
                    shifted = [
                        [
                            value + sign * step * tangent
                            for value, tangent in zip(row, move, strict=True)
                        ]
                        for row, move in zip(centers, motion, strict=True)
                    ]
                    moved_point = [
                        [
                            value + sign * step * tangent
                            for value, tangent in zip(point, motion[owner], strict=True)
                        ]
                    ]
                    products.append(
                        decimal_partition(moved_point, shifted, iterations, radii)[0][
                            owner
                        ]
                    )
                expected = float((products[0] - products[1]) / (2 * step)) * raw[index]
                assert actual[index] == pytest.approx(expected, abs=2e-10, rel=3e-9)
        record_testsuite_property(
            f"grid_weight_v{version}_k{iterations}_points_pairs", work
        )


@pytest.mark.parametrize("version", [1, 4])
def test_weight_direction_failures_aliases_and_late_nonfinite_recover(
    native: Any, version: int
) -> None:
    with _owner(native, version) as (handle, data):
        count = len(data[1])
        expected, _ = _direction(native, handle, MOTION, 0, count)
        workspace = np.empty(native.host_grid_workspace(handle, count) // 8)
        output = np.full(count, -9182.5)
        for settings in (
            {"direction": np.full_like(MOTION, np.nan)},
            {"direction": MOTION[:2]},
            {"begin": 1},
            {"workspace": workspace[:-1]},
            {"output": output[:-1]},
            {"workspace": output},
        ):
            options = {
                "native": native,
                "handle": handle,
                "direction": MOTION,
                "begin": 0,
                "count": count,
                "output": output,
                "workspace": workspace,
            }
            options.update(settings)
            with pytest.raises(RuntimeError):
                _direction(**options)
            np.testing.assert_array_equal(output, -9182.5)
        retained = np.ctypeslib.as_array(
            native.host_grid_retained_weights(handle), shape=(count,)
        )
        saved = retained.copy()
        with pytest.raises(RuntimeError, match="alias"):
            _direction(native, handle, MOTION, 0, count, output=retained)
        np.testing.assert_array_equal(retained, saved)
        alias_output = np.full_like(workspace, -9182.5)
        with pytest.raises(RuntimeError, match="alias"):
            _direction(
                native,
                handle,
                MOTION,
                0,
                count,
                output=alias_output[:count],
                workspace=alias_output,
            )
        np.testing.assert_array_equal(alias_output, -9182.5)
        with pytest.raises(RuntimeError, match="alias"):
            _direction(
                native,
                handle,
                workspace[: MOTION.size],
                0,
                count,
                output=output,
                workspace=workspace,
            )
        np.testing.assert_array_equal(output, -9182.5)
        retained[-1] = np.inf
        try:
            with pytest.raises(RuntimeError, match="nonfinite"):
                _direction(native, handle, MOTION, 0, count, output=output)
            np.testing.assert_array_equal(output, -9182.5)
        finally:
            retained[:] = saved
        actual, _ = _direction(native, handle, MOTION, 0, count)
        np.testing.assert_array_equal(actual, expected)
        aliased = MOTION.ravel().copy()
        actual, _ = _direction(native, handle, aliased, 0, aliased.size, output=aliased)
        np.testing.assert_array_equal(actual, expected[: aliased.size])
        empty, work = _direction(native, handle, MOTION, count, 0)
        assert empty.size == 0 and work == (0, 0)
        assert native.host_grid_workspace(handle, count + 1) == ct.c_size_t(-1).value


def test_single_atom_and_coincident_partition_conventions(native: Any) -> None:
    for centers, numbers in ((CENTERS[:1], (1,)), (np.zeros((2, 3)), (1, 1))):
        with _owner(native, 2, numbers, centers) as (handle, data):
            result, _ = _direction(
                native, handle, MOTION[: len(numbers)], 0, len(data[1])
            )
            np.testing.assert_array_equal(result, 0)


def test_mixed_ni_iodine_caesium_grid_weight_motion(native: Any) -> None:
    """An actual heavy-element grid source, not a molecular endpoint claim."""
    with _owner(native, 4, (28, 53, 55)) as (handle, data):
        actual, _ = _direction(native, handle, MOTION, 0, len(data[1]))
        for step in (1e-4, 3e-5):
            with _owner(native, 4, (28, 53, 55), CENTERS + step * MOTION) as (_, plus):
                plus_weights = plus[1]
            with _owner(native, 4, (28, 53, 55), CENTERS - step * MOTION) as (_, minus):
                expected = (plus_weights - minus[1]) / (2 * step)
            np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("atoms", [25, 40])
def test_fixed_tile_storage_and_actual_pair_work_at_larger_shape(
    native: Any,
    atoms: int,
    record_testsuite_property: Any,
) -> None:
    """Only weight-source capacity, not a large-molecule Hessian qualification."""
    indices = np.arange(atoms)
    centers = np.column_stack((indices % 4, (indices // 4) % 4, indices // 16)) * 3.0
    motion = np.random.default_rng(42711).normal(size=(atoms, 3)) * 0.02
    with _owner(native, 2, (1,) * atoms, centers) as (handle, data):
        count = 31
        bytes_needed = native.host_grid_workspace(handle, count)
        assert 0 < bytes_needed <= (3 * atoms * atoms + 5 * atoms + count) * 8
        capacity = np.empty(bytes_needed // 8)
        output = np.empty(count)
        for begin in (0, len(data[1]) // 2, len(data[1]) - count):
            value, work = _direction(
                native, handle, motion, begin, count, output=output, workspace=capacity
            )
            assert np.all(np.isfinite(value))
            assert work == (count, count * atoms * (atoms - 1) // 2)
            assert native.host_grid_planned_pairs(handle, count) == work[1]
        record_testsuite_property(f"grid_weight_{atoms}_atoms_tile_bytes", bytes_needed)
        record_testsuite_property(f"grid_weight_{atoms}_atoms_point_pair_visits", work)


def test_immutable_grid_accepts_concurrent_disjoint_tile_workspaces(
    native: Any,
) -> None:
    with _owner(native, 4) as (handle, data):
        expected, _ = _direction(native, handle, MOTION, 0, len(data[1]))
        tiles = [
            (begin, min(137, len(expected) - begin))
            for begin in range(0, len(expected), 137)
        ]

        def read(tile: tuple[int, int]) -> np.ndarray:
            begin, count = tile
            return _direction(native, handle, MOTION, begin, count)[0]

        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = np.concatenate(list(workers.map(read, tiles)))
        np.testing.assert_array_equal(actual, expected)
