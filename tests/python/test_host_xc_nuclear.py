"""Independent AO/Libxc qualification of actual native moving-grid sources."""

from __future__ import annotations

import ctypes as ct
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header
from test_host_xc_response import _independent_fock

from tools.generate_host_grid_maps import generate

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
CENTERS = np.array([[0.1, -0.2, 0.3], [1.5, 0.3, -0.2]])
MOTION = np.array([[0.17, -0.08, 0.12], [-0.09, 0.11, -0.15]])


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Compile current geometry/grid/XC and borrow only the unchanged AO core."""
    directory = tmp_path_factory.mktemp("host-xc-nuclear")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = [
        "xc",
        "coefficient_response",
        "gga_field_response",
        "ao_first_motion",
    ]
    inventory = parse_inventory(payload)
    entries = programs(inventory)
    sources = []
    for entry in entries:
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    (directory / "host_grid_maps.hpp").write_text(generate())
    (directory / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    core = Path(policy_library._name).resolve()
    library = required_native_cxx.build_shared(
        [
            *sources,
            registry,
            ROOT / "src/dft/grid.cpp",
            ROOT / "src/dft/host_grid_geometry.cpp",
            ROOT / "src/dft/host_xc_nuclear.cpp",
            ROOT / "tests/native/host_xc_nuclear_probe.cpp",
        ],
        directory / "nuclear.so",
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
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )
    loaded = ct.CDLL(str(library))
    loaded.host_nuclear_error.restype = ct.c_char_p
    loaded.host_nuclear_create.argtypes = [
        POINTER,
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_char_p,
        ct.c_char_p,
    ]
    loaded.host_nuclear_create.restype = ct.c_void_p
    loaded.host_nuclear_destroy.argtypes = [ct.c_void_p]
    loaded.host_nuclear_bind.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        ct.c_char_p,
        SIZE_POINTER,
        SIZE_POINTER,
    ]
    loaded.host_nuclear_sizes.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_nuclear_grid.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        ct.POINTER(ct.c_uint32),
    ]
    loaded.host_nuclear_borrowed.argtypes = [ct.c_void_p, ct.c_uint]
    loaded.host_nuclear_borrowed.restype = POINTER
    loaded.host_nuclear_failure.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.host_nuclear_read.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ] + [POINTER] * 4
    loaded.host_nuclear_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_nuclear_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.motion_identity = next(
        entry.identity.encode() for entry in entries if entry.name == "ao_first_motion"
    )
    return loaded


def _method(spins: int) -> bytes:
    return resolve_method(
        "B3LYP", spin="unpolarized" if spins == 1 else "polarized"
    ).identity.encode()


@contextmanager
def _owner(
    native: Any,
    spins: int = 1,
    maximum: int = 1,
    tile: int = 17,
    version: int = 2,
    centers: np.ndarray = CENTERS,
    density: np.ndarray | None = None,
) -> Any:
    basis = 2 * (maximum + 1) ** 2
    if density is None:
        rng = np.random.default_rng(728)
        factors = rng.normal(size=(spins, basis, basis)) / np.sqrt(basis)
        density = factors @ factors.transpose(0, 2, 1) + 0.05 * np.eye(basis)
    density = np.ascontiguousarray(density, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    handle = native.host_nuclear_create(
        _pointer(centers),
        maximum,
        version,
        spins,
        tile,
        _pointer(density),
        native.motion_identity,
        _method(spins),
    )
    if not handle:
        raise RuntimeError(native.host_nuclear_error().decode())
    try:
        sizes = (ct.c_size_t * 10)()
        native.host_nuclear_sizes(handle, sizes)
        yield handle, density, tuple(sizes)
    finally:
        native.host_nuclear_destroy(handle)


def _grid(native: Any, handle: int, count: int) -> tuple:
    points, weights, owners = (
        np.empty((count, 3)),
        np.empty(count),
        np.empty(count, dtype=np.uint32),
    )
    native.host_nuclear_grid(
        handle,
        _pointer(points),
        _pointer(weights),
        owners.ctypes.data_as(ct.POINTER(ct.c_uint32)),
    )
    return points, weights, owners


def _ao(centers: np.ndarray, points: np.ndarray, maximum: int) -> np.ndarray:
    """Independent libcint normalized spherical spatial AO jets."""
    gto = pytest.importorskip("pyscf.gto")
    numint = pytest.importorskip("pyscf.dft.numint")
    basis = [
        [angular, [0.7 + 0.13 * angular, 0.7], [0.22 + 0.04 * angular, 0.3]]
        for angular in range(maximum + 1)
    ]
    molecule = gto.M(
        atom=[("H", center) for center in centers],
        basis={"H": basis},
        unit="Bohr",
        verbose=0,
    )
    return numint.eval_ao(molecule, points, deriv=1)


def _read(
    native: Any,
    handle: int,
    begin: int,
    count: int,
    basis: int,
    direction: np.ndarray = MOTION,
) -> tuple:
    outputs = (
        np.full((4, count, basis), -871.5),
        np.full((4, count, basis), -871.5),
        np.full(count, -871.5),
        np.full(count, -871.5),
    )
    status = native.host_nuclear_read(
        handle,
        begin,
        count,
        _pointer(direction),
        direction.size,
        *map(_pointer, outputs),
    )
    if status:
        raise RuntimeError(native.host_nuclear_error().decode())
    return outputs


def _apply(
    native: Any, handle: int, density: np.ndarray, direction: np.ndarray = MOTION
) -> np.ndarray:
    output = np.full_like(density, -871.5)
    status = native.host_nuclear_apply(
        handle, _pointer(direction), direction.size, _pointer(output), output.size
    )
    if status:
        raise RuntimeError(native.host_nuclear_error().decode())
    return output


@pytest.mark.parametrize("maximum", [0, 1, 2, 3])
def test_actual_normalized_ao_motion_matches_independent_displacements(
    native: Any, maximum: int
) -> None:
    with _owner(native, maximum=maximum, tile=23) as (handle, _, sizes):
        basis, points_count = sizes[:2]
        points, weights, owners = _grid(native, handle, points_count)
        for begin in (0, points_count // 2 - 7, points_count - 23):
            ao, moving, weight, _ = _read(native, handle, begin, 23, basis)
            selected = points[begin : begin + 23]
            np.testing.assert_allclose(
                ao, _ao(CENTERS, selected, maximum), atol=2e-13, rtol=2e-12
            )
            np.testing.assert_array_equal(weight, weights[begin : begin + 23])
            errors = []
            for step in (1e-3, 2e-4, 4e-5):
                delta_points = MOTION[owners[begin : begin + 23]] * step
                numerical = (
                    _ao(CENTERS + step * MOTION, selected + delta_points, maximum)
                    - _ao(CENTERS - step * MOTION, selected - delta_points, maximum)
                ) / (2 * step)
                errors.append(np.max(np.abs(numerical - moving)))
            assert errors[-1] < 2e-8
            assert errors[-1] < errors[0] / 50
        translated = _read(
            native, handle, 11, 23, basis, np.tile([0.2, -0.3, 0.4], (2, 1))
        )
        np.testing.assert_array_equal(translated[1], 0)
        np.testing.assert_allclose(translated[3], 0, atol=1e-13)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_full_frozen_xc_source_matches_independent_moving_fock(
    native: Any, spins: int, version: int, record_testsuite_property: Any
) -> None:
    tile = 97 if version == 4 else 17
    with _owner(native, spins=spins, version=version, tile=tile) as (
        handle,
        density,
        sizes,
    ):
        points, _, owners = _grid(native, handle, sizes[1])
        actual = _apply(native, handle, density)
        errors = []
        for step in (1e-3, 2e-4, 4e-5):
            focks = []
            for sign in (1, -1):
                centers = CENTERS + sign * step * MOTION
                with _owner(
                    native,
                    spins=spins,
                    version=version,
                    tile=tile,
                    centers=centers,
                    density=density,
                ) as (shifted, _, moved_sizes):
                    shifted_points, shifted_weights, shifted_owners = _grid(
                        native, shifted, moved_sizes[1]
                    )
                    np.testing.assert_array_equal(shifted_owners, owners)
                    np.testing.assert_allclose(
                        shifted_points,
                        points + sign * step * MOTION[owners],
                        atol=2e-14,
                        rtol=2e-15,
                    )
                    focks.append(
                        _independent_fock(
                            density, _ao(centers, shifted_points, 1), shifted_weights
                        )
                    )
            numerical = (focks[0] - focks[1]) / (2 * step)
            errors.append(float(np.max(np.abs(numerical - actual))))
        np.testing.assert_allclose(actual, numerical, atol=2e-8, rtol=2e-6)
        assert errors[-1] < errors[0] / 20
        np.testing.assert_allclose(
            actual, actual.transpose(0, 2, 1), atol=2e-13, rtol=2e-13
        )
        census = (ct.c_size_t * 11)()
        native.host_nuclear_work(handle, census)
        assert tuple(census)[2:5] == (sizes[1], sizes[8], sizes[9])
        assert census[5] == census[6] == 1
        assert census[8] == sizes[1]
        assert census[9] == 3 * spins * ((sizes[1] + tile - 1) // tile)
        assert census[10] == sizes[7]
        record_testsuite_property(
            f"frozen_xc_s{spins}_v{version}",
            json.dumps({"sizes": sizes, "errors": errors, "work": list(census)}),
        )
        translated = _apply(native, handle, density, np.tile([0.2, -0.3, 0.4], (2, 1)))
        np.testing.assert_allclose(translated, 0, atol=3e-12)
        with _owner(
            native, spins=spins, version=version, tile=tile + 12, density=density
        ) as (other, _, _):
            np.testing.assert_allclose(
                _apply(native, other, density), actual, atol=2e-13, rtol=2e-13
            )


def test_caps_exact_closure_alias_failures_and_transactional_recovery(
    native: Any,
) -> None:
    with _owner(native, spins=2) as (handle, density, sizes):
        expected = _apply(native, handle, density)
        first, second = list(sizes[2:6]), list(sizes[6:10])
        for group in (0, 1):
            for index in range(4):
                caps = [first.copy(), second.copy()]
                caps[group][index] -= 1
                assert (
                    native.host_nuclear_bind(
                        handle,
                        native.motion_identity,
                        _method(2),
                        (ct.c_size_t * 4)(*caps[0]),
                        (ct.c_size_t * 4)(*caps[1]),
                    )
                    != 0
                )
        for motion, method in (
            (b"forged", _method(2)),
            (native.motion_identity, b"forged"),
            (native.motion_identity, _method(1)),
        ):
            assert native.host_nuclear_bind(handle, motion, method, None, None) != 0
        for mode, at in ((1, 1), (1, 15), (2, 5), (3, 8)):
            output = np.full_like(density, -871.5)
            native.host_nuclear_failure(handle, mode, at)
            assert (
                native.host_nuclear_apply(
                    handle, _pointer(MOTION), 6, _pointer(output), output.size
                )
                != 0
            )
            np.testing.assert_array_equal(output, -871.5)
            native.host_nuclear_failure(handle, 0, 0)
            np.testing.assert_array_equal(_apply(native, handle, density), expected)
        assert (
            native.host_nuclear_apply(
                handle, _pointer(MOTION), 6, _pointer(density), density.size
            )
            != 0
        )
        for kind in (0, 1, 2):
            assert (
                native.host_nuclear_apply(
                    handle,
                    _pointer(MOTION),
                    6,
                    native.host_nuclear_borrowed(handle, kind),
                    density.size,
                )
                != 0
            )
        output = np.empty_like(density)
        output.ravel()[:6] = MOTION.ravel()
        assert (
            native.host_nuclear_apply(
                handle, _pointer(output), 6, _pointer(output), output.size
            )
            == 0
        )
        np.testing.assert_array_equal(output, expected)
        bad = MOTION.copy()
        bad[0, 0] = np.nan
        with pytest.raises(RuntimeError, match="nonfinite"):
            _apply(native, handle, density, bad)
        np.testing.assert_array_equal(_apply(native, handle, density), expected)


def test_geometry_output_alias_and_late_nonfinite_retained_weight(native: Any) -> None:
    with _owner(native) as (handle, density, sizes):
        basis, points = sizes[:2]
        expected = _read(native, handle, 5, 9, basis)
        ao, moving, weight, delta_weight = [
            np.full_like(values, -871.5) for values in expected
        ]
        ao.ravel()[:6] = MOTION.ravel()
        assert (
            native.host_nuclear_read(
                handle,
                5,
                9,
                _pointer(ao),
                6,
                _pointer(ao),
                _pointer(moving),
                _pointer(weight),
                _pointer(delta_weight),
            )
            == 0
        )
        for actual, reference in zip(
            (ao, moving, weight, delta_weight), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, reference)
        assert (
            native.host_nuclear_read(
                handle,
                5,
                9,
                _pointer(MOTION),
                6,
                _pointer(ao),
                _pointer(ao),
                _pointer(weight),
                _pointer(delta_weight),
            )
            != 0
        )
        retained = np.ctypeslib.as_array(
            native.host_nuclear_borrowed(handle, 2), shape=(points,)
        )
        original = retained[-1]
        retained[-1] = np.nan
        output = np.full_like(density, -871.5)
        try:
            assert (
                native.host_nuclear_apply(
                    handle, _pointer(MOTION), 6, _pointer(output), output.size
                )
                != 0
            )
            np.testing.assert_array_equal(output, -871.5)
        finally:
            retained[-1] = original
        assert np.isfinite(_apply(native, handle, density)).all()


@pytest.mark.parametrize("spins", [1, 2])
def test_exact_production_vacuum_branch(native: Any, spins: int) -> None:
    with _owner(native, spins=spins, density=np.zeros((spins, 8, 8))) as (
        handle,
        density,
        _,
    ):
        np.testing.assert_array_equal(_apply(native, handle, density), 0)


@pytest.mark.parametrize(
    "offset,value", [(0, 0.9), (6, -1), (22, -0.5), (23, 8.5), (24, 100), (25, 4)]
)
def test_malformed_packed_basis_admission_and_recovery(
    native: Any, offset: int, value: float
) -> None:
    with _owner(native) as (handle, density, _):
        expected = _apply(native, handle, density)
        packed = native.host_nuclear_borrowed(handle, 0)
        original = packed[offset]
        packed[offset] = value
        try:
            assert (
                native.host_nuclear_bind(
                    handle, native.motion_identity, _method(1), None, None
                )
                != 0
            )
            with pytest.raises(RuntimeError):
                _apply(native, handle, density)
        finally:
            packed[offset] = original
        np.testing.assert_array_equal(_apply(native, handle, density), expected)


def test_direction_linearity_and_invalid_tile_shapes(native: Any) -> None:
    with _owner(native) as (handle, density, sizes):
        second = np.array([[-0.07, 0.13, -0.02], [0.14, 0.03, -0.12]])
        first_result = _apply(native, handle, density)
        second_result = _apply(native, handle, density, second)
        np.testing.assert_allclose(
            _apply(native, handle, density, 2 * MOTION - 0.4 * second),
            2 * first_result - 0.4 * second_result,
            atol=1e-13,
            rtol=1e-12,
        )
        for begin, count in ((sizes[1], 1), (0, 18)):
            with pytest.raises(RuntimeError, match="tile binding"):
                _read(native, handle, begin, count, sizes[0])
        empty = _read(native, handle, sizes[1], 0, sizes[0])
        assert all(values.size == 0 for values in empty)
        output = np.full_like(density, -871.5)
        for direction_count, output_count in ((5, output.size), (6, output.size - 1)):
            assert (
                native.host_nuclear_apply(
                    handle,
                    _pointer(MOTION),
                    direction_count,
                    _pointer(output),
                    output_count,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -871.5)
