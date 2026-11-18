"""Independent frozen/supplied-density XC columns, not stationary Hessians."""

from __future__ import annotations

import ctypes as ct
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from test_host_grid_mixed_geometry import CENTERS, LEFT, RIGHT, _grid, _owner, _pointer
from test_host_xc_curvature import CASES, _apply, _density, _primal, _source
from test_host_xc_curvature import native as native_fixture

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)


@pytest.fixture(scope="module")
def column_native(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Any:
    """Compile actual Graph/IR/native sources and a Python-free entry point."""
    import pyscf.dft.libxc

    assert pyscf.dft.libxc.__file__
    builder = cast("Any", native_fixture)
    loaded = builder.__wrapped__(tmp_path_factory, required_native_cxx, policy_library)
    loaded.host_grid_column_reader.argtypes = [ct.c_void_p]
    loaded.host_grid_column_reader.restype = ct.c_void_p
    loaded.host_xc_column_error.restype = ct.c_char_p
    loaded.host_xc_column_create.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        ct.c_char_p,
        SIZE_POINTER,
    ]
    loaded.host_xc_column_create.restype = ct.c_void_p
    loaded.host_xc_column_destroy.argtypes = [ct.c_void_p]
    loaded.host_xc_column_plan.argtypes = [ct.c_void_p, ct.c_size_t, SIZE_POINTER]
    loaded.host_xc_column_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_xc_column_response.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_xc_column_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_xc_column_geometry_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_xc_column_fault.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    library = Path(loaded._name).resolve()
    core = Path(policy_library._name).resolve()
    loaded.column_standalone = required_native_cxx.build_executable(
        [ROOT / "tests/native/host_xc_column_probe.cpp"],
        library.parent / "column-standalone",
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
            "-DGENERATIVEQC_XC_COLUMN_STANDALONE=1",
        ],
        link_args=[
            str(library),
            str(core),
            "-Wl,-rpath," + str(library.parent),
            "-Wl,-rpath," + str(core.parent),
        ],
    )
    return loaded


def _plan(native: Any, reader: Any, spins: int) -> np.ndarray:
    values = np.zeros(7, dtype=np.uintp)
    assert (
        native.host_xc_column_plan(reader, spins, values.ctypes.data_as(SIZE_POINTER))
        == 0
    )
    return values


@contextmanager
def _column_source(native: Any, reader: Any, spins: int, density: np.ndarray) -> Any:
    source = native.host_xc_column_create(
        reader, spins, _pointer(density), density.size, None, None
    )
    if not source:
        raise RuntimeError(native.host_xc_column_error().decode())
    try:
        yield source
    finally:
        native.host_xc_column_destroy(source)


def _column(
    native: Any,
    source: Any,
    right: np.ndarray = RIGHT,
    output: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    if output is None:
        output = np.full(2 + 2 * right.size, -9841.5)
    status = native.host_xc_column_apply(
        source, _pointer(right), right.size, _pointer(output), output.size
    )
    return status, output


def _work(native: Any, source: Any, geometry: bool = False) -> np.ndarray:
    result = np.zeros(7, dtype=np.uintp)
    function = (
        native.host_xc_column_geometry_work if geometry else native.host_xc_column_work
    )
    function(source, result.ctypes.data_as(SIZE_POINTER))
    return result


def _response(
    native: Any,
    source: Any,
    right: np.ndarray,
    density_right: np.ndarray,
    output: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    if output is None:
        output = np.full(2 + 2 * right.size, -9841.5)
    status = native.host_xc_column_response(
        source,
        _pointer(right),
        right.size,
        _pointer(density_right),
        density_right.size,
        _pointer(output),
        output.size,
    )
    return status, output


def _energy(
    native: Any, density: np.ndarray, centers: np.ndarray, maximum: int, cartesian: bool
) -> float:
    """Independent primal at a displaced frame and an explicitly supplied D."""
    with _owner(
        native, 3, maximum=maximum, cartesian=cartesian, centers=centers, tile=257
    ) as (handle, shape):
        points, weights, _, _, _ = _grid(native, handle, shape)
    return _primal(density, centers, points, weights, maximum, cartesian)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("cartesian", [False, True])
def test_supplied_density_response_column_against_independent_primal(
    column_native: Any,
    spins: int,
    cartesian: bool,
    tmp_path: Path,
) -> None:
    """Right-only D motion is independent of the fixed-D left nuclear partial."""
    native = column_native
    with _owner(native, maximum=3, cartesian=cartesian, tile=257) as (handle, shape):
        density = _density(spins, shape[1])
        random = np.random.default_rng(671 + spins + int(cartesian))
        moving = random.normal(scale=0.004, size=density.shape)
        density_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
        reader = native.host_grid_column_reader(handle)
        with _column_source(native, reader, spins, density) as source:
            status, actual = _response(native, source, RIGHT, density_right)
            assert status == 0, native.host_xc_column_error().decode()
            np.testing.assert_allclose(
                actual[0],
                _energy(native, density, CENTERS, 3, cartesian),
                atol=3e-11,
                rtol=2e-12,
            )
            steps, scale = np.array([1e-2, 3e-3, 1e-3]), 0.2
            legs = scale * np.eye(9).reshape(-1, 3, 3)
            corners, gradient_corners, right_corners = [], [], []
            for step in steps:
                rows, gradient_rows = [], []
                for leg in legs:
                    values = [
                        _energy(
                            native,
                            density + second * step * density_right,
                            CENTERS + step * (first * leg + second * RIGHT),
                            3,
                            cartesian,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    rows.append(values)
                    gradient_rows.append(
                        [
                            _energy(
                                native,
                                density,
                                CENTERS + sign * step * leg,
                                3,
                                cartesian,
                            )
                            for sign in (1, -1)
                        ]
                    )
                right_corners.append(
                    [
                        _energy(
                            native,
                            density + sign * step * density_right,
                            CENTERS + sign * step * RIGHT,
                            3,
                            cartesian,
                        )
                        for sign in (1, -1)
                    ]
                )
                corners.append(rows)
                gradient_corners.append(gradient_rows)
            corners = np.asarray(corners, dtype=np.float64)
            gradient_corners = np.asarray(gradient_corners, dtype=np.float64)
            right_corners = np.asarray(right_corners, dtype=np.float64)
            right_control = (right_corners[:, 0] - right_corners[:, 1]) / (2 * steps)
            gradient = (gradient_corners[:, :, 0] - gradient_corners[:, :, 1]) / (
                2 * scale * steps[:, None]
            )
            mixed = (
                corners[:, :, 0]
                - corners[:, :, 1]
                - corners[:, :, 2]
                + corners[:, :, 3]
            ) / (4 * scale * steps[:, None] ** 2)
            expected = np.column_stack((right_control, gradient, mixed))
            errors = np.abs(expected - actual[None, 1:])
            np.savez_compressed(
                tmp_path / "xc-column-supplied-density-primal.npz",
                centers=CENTERS,
                density=density,
                density_right=density_right,
                right=RIGHT,
                native=actual,
                steps=steps,
                left_scale=scale,
                cartesian=cartesian,
                primal_corners=corners,
                gradient_corners=gradient_corners,
                right_corners=right_corners,
                errors=errors,
            )
            np.testing.assert_allclose(actual[1:], expected[-1], atol=3e-7, rtol=2e-6)
            assert errors[-1].max() < max(errors[0].max() / 12, 3e-8)
            status, frozen = _column(native, source)
            assert status == 0
            status, response = _response(
                native, source, np.zeros_like(RIGHT), density_right
            )
            assert status == 0
            np.testing.assert_array_equal(actual[2:11], frozen[2:11])
            np.testing.assert_allclose(
                actual[1], frozen[1] + response[1], atol=3e-11, rtol=3e-10
            )
            np.testing.assert_allclose(
                actual[11:], frozen[11:] + response[11:], atol=3e-11, rtol=3e-10
            )


@pytest.mark.parametrize("spins", [1, 2])
def test_density_response_inputs_alias_authentication_and_recovery(
    column_native: Any, spins: int
) -> None:
    native = column_native
    with _owner(native, maximum=0, tile=73) as (handle, shape):
        density = _density(spins, shape[1])
        delta = np.ascontiguousarray(np.tile(np.eye(shape[1]), (spins, 1, 1)) * 0.03)
        with _column_source(
            native, native.host_grid_column_reader(handle), spins, density
        ) as source:
            status, baseline = _response(native, source, RIGHT, delta)
            assert status == 0
            for invalid in (
                delta.ravel()[:-1],
                np.full_like(delta, np.nan),
                delta.copy(),
            ):
                if invalid.shape == delta.shape and np.isfinite(invalid).all():
                    invalid[0, 0, 1] += 0.1
                status, output = _response(native, source, RIGHT, invalid)
                assert status != 0
                np.testing.assert_array_equal(output, -9841.5)
            output = np.full(20, -9841.5)
            assert (
                native.host_xc_column_response(
                    source,
                    _pointer(RIGHT),
                    9,
                    None,
                    delta.size,
                    _pointer(output),
                    output.size,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -9841.5)
            before = int(_work(native, source)[3])
            saved = delta.copy()
            native.host_xc_column_fault(source, 6, before + shape[2])
            status, output = _response(native, source, RIGHT, delta)
            assert (
                status != 0
                and b"density direction changed" in native.host_xc_column_error()
            )
            np.testing.assert_array_equal(output, -9841.5)
            delta[:] = saved
            native.host_xc_column_fault(source, 0, 0)
            np.testing.assert_array_equal(
                _response(native, source, RIGHT, delta)[1], baseline
            )
            storage = np.full(max(delta.size, 20), -9841.5)
            storage[: delta.size] = delta.ravel()
            status, output = _response(
                native, source, RIGHT, storage[: delta.size], storage[:20]
            )
            assert status == 0
            np.testing.assert_array_equal(output, baseline)


def test_column_matches_accepted_independent_mixed_source(column_native: Any) -> None:
    """An integration check supplements, never replaces, the primal oracles."""
    native = column_native
    with _owner(native, maximum=0, tile=73) as (handle, shape):
        for spins in (1, 2):
            density = _density(spins, shape[1])
            with _column_source(
                native, native.host_grid_column_reader(handle), spins, density
            ) as source:
                status, actual = _column(native, source)
                assert status == 0, native.host_xc_column_error().decode()
            with _source(
                native, native.host_mixed_reader(handle), spins, density
            ) as source:
                status, pair = _apply(native, source)
                assert status == 0
            projected = [
                actual[0],
                LEFT.ravel() @ actual[2:11],
                actual[1],
                LEFT.ravel() @ actual[11:],
            ]
            np.testing.assert_allclose(projected, pair, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize(("spins", "version", "cartesian"), CASES)
def test_complete_frozen_xc_columns_against_independent_primal(
    column_native: Any,
    spins: int,
    version: int,
    cartesian: bool,
    tmp_path: Path,
) -> None:
    """All physical gradient/HVP components use frozen-D libcint/Libxc primals."""
    native = column_native
    maximum = 3 if version == 3 else 1
    with _owner(native, version, maximum=maximum, cartesian=cartesian, tile=257) as (
        handle,
        shape,
    ):
        density = _density(spins, shape[1])
        points, weights, _, _, _ = _grid(native, handle, shape)
        baseline = _primal(density, CENTERS, points, weights, maximum, cartesian)
        reader = native.host_grid_column_reader(handle)
        with _column_source(native, reader, spins, density) as source:
            status, actual = _column(native, source)
            assert status == 0, native.host_xc_column_error().decode()
            np.testing.assert_allclose(actual[0], baseline, atol=3e-11, rtol=2e-12)
            steps = np.array([1e-2, 3e-3, 1e-3])
            scale = 0.2
            legs = scale * np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
            corners, gradient_corners = [], []
            for step in steps:
                rows, first_rows = [], []
                for leg in legs:
                    values = []
                    for first, second in (
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                        (1, 0),
                        (-1, 0),
                    ):
                        centers = CENTERS + step * (first * leg + second * RIGHT)
                        with _owner(
                            native,
                            version,
                            maximum=maximum,
                            cartesian=cartesian,
                            centers=centers,
                            tile=257,
                        ) as (shifted, shifted_shape):
                            shifted_points, shifted_weights, _, _, _ = _grid(
                                native, shifted, shifted_shape
                            )
                        values.append(
                            _primal(
                                density,
                                centers,
                                shifted_points,
                                shifted_weights,
                                maximum,
                                cartesian,
                            )
                        )
                    rows.append(values[:4])
                    first_rows.append(values[4:])
                corners.append(rows)
                gradient_corners.append(first_rows)
            corners = np.array(corners)
            gradient_corners = np.array(gradient_corners)
            gradient = (gradient_corners[:, :, 0] - gradient_corners[:, :, 1]) / (
                2 * scale * steps[:, None]
            )
            mixed = (
                corners[:, :, 0]
                - corners[:, :, 1]
                - corners[:, :, 2]
                + corners[:, :, 3]
            ) / (4 * scale * steps[:, None] ** 2)
            expected = np.concatenate((gradient, mixed), axis=1)
            errors = np.abs(expected - actual[None, 2:])
            np.savez_compressed(
                tmp_path / "frozen-xc-column-primal.npz",
                centers=CENTERS,
                density=density,
                right=RIGHT,
                left_scale=scale,
                native=actual,
                primal_baseline=baseline,
                primal_corners=corners,
                gradient_corners=gradient_corners,
                steps=steps,
                errors=errors,
                grid_version=version,
                cartesian=cartesian,
                maximum=maximum,
                work=_work(native, source),
                geometry_work=_work(native, source, True),
                resources=_plan(native, reader, spins),
            )
            np.testing.assert_allclose(actual[2:], expected[-1], atol=3e-7, rtol=2e-6)
            assert errors[-1].max() < max(errors[0].max() / 12, 3e-8), errors
            np.testing.assert_allclose(
                RIGHT.ravel() @ actual[2:11], actual[1], atol=3e-11, rtol=3e-11
            )


@pytest.mark.parametrize("spins", [1, 2])
def test_raw_symmetry_linearity_rigid_translation_and_alias(
    column_native: Any, spins: int, tmp_path: Path
) -> None:
    native = column_native
    with _owner(native, maximum=0, tile=73) as (handle, shape):
        density = _density(spins, shape[1])
        with _column_source(
            native, native.host_grid_column_reader(handle), spins, density
        ) as source:
            status, baseline = _column(native, source)
            assert status == 0
            legs = np.eye(9).reshape(-1, 3, 3)
            matrix = np.array([_column(native, source, leg)[1][11:] for leg in legs]).T
            np.testing.assert_allclose(matrix, matrix.T, atol=3e-11, rtol=3e-10)
            np.testing.assert_allclose(
                matrix @ RIGHT.ravel(), baseline[11:], atol=3e-11, rtol=3e-10
            )
            _, first = _column(native, source, LEFT)
            _, combined = _column(native, source, 2 * LEFT - RIGHT)
            np.testing.assert_allclose(
                combined[11:], 2 * first[11:] - baseline[11:], atol=3e-11, rtol=3e-10
            )
            _, rigid = _column(native, source, np.tile([0.13, -0.07, 0.11], (3, 1)))
            np.testing.assert_array_equal(rigid[1], 0)
            np.testing.assert_array_equal(rigid[11:], 0)
            np.testing.assert_array_equal(rigid[2:11], baseline[2:11])
            alias = np.full(20, -9841.5)
            alias[:9] = RIGHT.ravel()
            assert (
                native.host_xc_column_apply(
                    source, _pointer(alias), 9, _pointer(alias), 20
                )
                == 0
            )
            np.testing.assert_array_equal(alias, baseline)
            np.savez_compressed(
                tmp_path / "frozen-xc-column-raw-matrix.npz",
                matrix=matrix,
                right=RIGHT,
                native=baseline,
                density=density,
            )


def test_caps_identity_immutable_owner_failures_and_recovery(
    column_native: Any,
) -> None:
    native = column_native
    with _owner(native, maximum=0, tile=73) as (handle, shape):
        spins = 2
        reader = native.host_grid_column_reader(handle)
        density = _density(spins, shape[1])
        resources = _plan(native, reader, spins)
        for index in range(len(resources)):
            limits = resources.copy()
            limits[index] -= 1
            assert not native.host_xc_column_create(
                reader,
                spins,
                _pointer(density),
                density.size,
                None,
                limits.ctypes.data_as(SIZE_POINTER),
            )
        assert not native.host_xc_column_create(
            reader, spins, _pointer(density), density.size, b"wrong", None
        )
        assert not native.host_xc_column_create(
            reader, 0, _pointer(density), density.size, None, None
        )
        assert not native.host_xc_column_create(
            reader, spins, None, density.size, None, None
        )
        with _column_source(native, reader, spins, density) as source:
            status, baseline = _column(native, source)
            assert status == 0
            for mode in (1, 2, 3, 4, 5):
                before = int(_work(native, source)[3])
                saved_density, right = density.copy(), RIGHT.copy()
                native.host_xc_column_fault(source, mode, before + shape[2])
                status, output = _column(native, source, right)
                assert status != 0, (mode, native.host_xc_column_error())
                np.testing.assert_array_equal(output, -9841.5)
                density[:] = saved_density
                native.host_xc_column_fault(source, 0, 0)
                np.testing.assert_array_equal(_column(native, source)[1], baseline)
            assert (
                native.host_xc_column_apply(
                    source, _pointer(RIGHT), 9, _pointer(density), 20
                )
                != 0
            )
            for right, right_size, output, output_size in (
                (None, 9, _pointer(np.zeros(20)), 20),
                (_pointer(RIGHT), 8, _pointer(np.zeros(20)), 20),
                (_pointer(RIGHT), 9, None, 20),
                (_pointer(RIGHT), 9, _pointer(np.zeros(19)), 19),
            ):
                assert (
                    native.host_xc_column_apply(
                        source, right, right_size, output, output_size
                    )
                    != 0
                )


@pytest.mark.parametrize("spins", [1, 2])
def test_vacuum_and_nonvacuum_empty_spin_domain_are_unchanged(
    column_native: Any, spins: int
) -> None:
    native = column_native
    with _owner(native, maximum=0, tile=73) as (handle, shape):
        density = np.zeros((spins, shape[1], shape[1]))
        reader = native.host_grid_column_reader(handle)
        with _column_source(native, reader, spins, density) as source:
            status, output = _column(native, source)
            assert status == 0, native.host_xc_column_error().decode()
            np.testing.assert_array_equal(output, 0)
        if spins == 2:
            density[0] = np.eye(shape[1]) * 0.2
            with _column_source(native, reader, spins, density) as source:
                status, output = _column(native, source)
                assert status != 0
                np.testing.assert_array_equal(output, -9841.5)


@pytest.mark.parametrize("atoms", [3, 25, 40])
def test_complete_frozen_xc_source_work_has_no_left_coordinate_replay(
    column_native: Any,
    atoms: int,
    tmp_path: Path,
) -> None:
    """Complete frozen XC is still not a complete electronic endpoint."""
    native = column_native
    centers = np.array(
        [
            [1.7 * (atom % 5), 1.9 * ((atom // 5) % 4), 2.1 * (atom // 20)]
            for atom in range(atoms)
        ]
    )
    right = np.arange(3 * atoms, dtype=float).reshape(atoms, 3) * 0.0001
    with _owner(native, maximum=0, centers=centers, numbers=(1,) * atoms, tile=73) as (
        handle,
        shape,
    ):
        density = _density(2, shape[1])
        reader = native.host_grid_column_reader(handle)
        resources = _plan(native, reader, 2)
        with _column_source(native, reader, 2, density) as source:
            before = _work(native, source, True)
            start = time.perf_counter()
            status, output = _column(native, source, right)
            seconds = time.perf_counter() - start
            assert status == 0, native.host_xc_column_error().decode()
            work, geometry = _work(native, source), _work(native, source, True) - before
            assert work[0] == work[1] == 1 and work[3] == shape[2]
            assert work[4] == resources[1] and work[5] == resources[2]
            assert geometry[3] == resources[3] == shape[1] * shape[2]
            assert work[6] <= resources[4] and geometry[4] == resources[5]
            assert geometry[5] + geometry[6] <= resources[6]
            assert resources[0] < 64 * 1024**2
            np.savez_compressed(
                tmp_path / "frozen-xc-column-source-work.npz",
                centers=centers,
                density=density,
                right=right,
                native=output,
                resources=resources,
                work=work,
                geometry_work=geometry,
                seconds=seconds,
            )


def test_native_xc_column_runs_without_python_or_compiler(
    column_native: Any, tmp_path: Path
) -> None:
    empty = tmp_path / "no-python"
    empty.mkdir()
    result = subprocess.run(
        [str(column_native.column_standalone)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PATH": str(empty),
            "PYTHONPATH": str(empty),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert (
        result.stdout.strip()
        == "XC columns: frozen/supplied-density, Cartesian/spherical RKS/UKS"
    )
