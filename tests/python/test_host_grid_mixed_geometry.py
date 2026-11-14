"""Independent primal controls for bounded native mixed AO/partition geometry."""

from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header
from test_grid_response import decimal_partition

from tools.generate_grid_geometry_response import generate

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
CENTERS = np.array([[0.1, -0.2, 0.3], [1.7, 0.2, -0.4], [-0.5, 1.3, 0.7]])
LEFT = np.array([[0.2, -0.1, 0.3], [-0.1, 0.2, -0.15], [0.08, -0.04, 0.12]])
RIGHT = np.array([[-0.07, 0.15, 0.08], [0.06, -0.13, 0.04], [0.12, 0.03, -0.06]])


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Compile current consumers and the shared Graph math with strict host flags."""
    directory = tmp_path_factory.mktemp("host-mixed-grid")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["ao_first_motion"]
    inventory = parse_inventory(payload)
    sources = []
    for entry in programs(inventory):
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    (directory / "host_grid_geometry_ad.hpp").write_text(generate("cpu"))
    (directory / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    core = Path(policy_library._name).resolve()
    sources.extend(
        [
            registry,
            ROOT / "src/dft/grid.cpp",
            ROOT / "src/dft/host_grid_geometry.cpp",
            ROOT / "src/dft/host_grid_mixed_geometry.cpp",
            ROOT / "src/dft/host_grid_column_geometry.cpp",
            ROOT / "tests/native/host_grid_mixed_geometry_probe.cpp",
        ]
    )
    flags = [
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
    ]
    link = [str(core), "-Wl,-rpath," + str(core.parent)]
    library = required_native_cxx.build_shared(
        sources, directory / "mixed.so", compile_args=flags, link_args=link
    )
    executable = required_native_cxx.build_executable(
        sources,
        directory / "standalone",
        compile_args=[*flags, "-DGENERATIVEQC_MIXED_GRID_STANDALONE=1"],
        link_args=link,
    )
    loaded = ct.CDLL(str(library))
    loaded.standalone = executable
    loaded.host_mixed_error.restype = ct.c_char_p
    loaded.host_mixed_create.argtypes = [
        POINTER,
        ct.POINTER(ct.c_int),
        ct.c_size_t,
        ct.c_uint,
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
        ct.c_bool,
    ]
    loaded.host_mixed_create.restype = ct.c_void_p
    loaded.host_mixed_destroy.argtypes = [ct.c_void_p]
    loaded.host_mixed_shape.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_mixed_plan.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_mixed_bind.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_mixed_export.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        POINTER,
        ct.POINTER(ct.c_uint32),
        POINTER,
    ]
    loaded.host_mixed_read.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t] + [
        kind for _ in range(6) for kind in (POINTER, ct.c_size_t)
    ]
    loaded.host_mixed_fault.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.host_mixed_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_mixed_borrowed.argtypes = [ct.c_void_p, ct.c_uint]
    loaded.host_mixed_borrowed.restype = POINTER
    return loaded


@contextmanager
def _owner(
    native: Any,
    version: int = 3,
    iterations: int = 3,
    maximum: int = 3,
    cartesian: bool = False,
    tile: int = 11,
    centers: np.ndarray = CENTERS,
    numbers: tuple = (1, 8, 6),
) -> Any:
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    atomic_numbers = np.array(numbers, dtype=np.int32)
    handle = native.host_mixed_create(
        _pointer(centers),
        atomic_numbers.ctypes.data_as(ct.POINTER(ct.c_int)),
        len(numbers),
        maximum,
        version,
        iterations,
        tile,
        cartesian,
    )
    if not handle:
        raise RuntimeError(native.host_mixed_error().decode())
    try:
        shape = (ct.c_size_t * 4)()
        native.host_mixed_shape(handle, shape)
        yield handle, tuple(shape)
    finally:
        native.host_mixed_destroy(handle)


def _grid(native: Any, handle: int, shape: tuple) -> tuple:
    atoms, _, points, _ = shape
    xyz, weight, atomic = np.empty((points, 3)), np.empty(points), np.empty(points)
    owners, radii = np.empty(points, dtype=np.uint32), np.empty(atoms)
    native.host_mixed_export(
        handle,
        _pointer(xyz),
        _pointer(weight),
        _pointer(atomic),
        owners.ctypes.data_as(ct.POINTER(ct.c_uint32)),
        _pointer(radii),
    )
    return xyz, weight, atomic, owners, radii


def _read(
    native: Any,
    handle: int,
    shape: tuple,
    begin: int,
    count: int,
    left: np.ndarray = LEFT,
    right: np.ndarray = RIGHT,
    outputs: tuple | None = None,
) -> tuple:
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    if outputs is None:
        outputs = (
            np.full((4, count, shape[1]), -9713.5),
            np.full((3, 4, count, shape[1]), -9713.5),
            np.full(count, -9713.5),
            np.full((3, count), -9713.5),
        )
    bindings = [
        value
        for array in (left, right, *outputs)
        for value in (_pointer(array), array.size)
    ]
    return native.host_mixed_read(handle, begin, count, *bindings), outputs


def _ao(
    centers: np.ndarray, points: np.ndarray, maximum: int, cartesian: bool
) -> np.ndarray:
    """Independent libcint spatial values/gradients with public component norms."""
    gto = pytest.importorskip("pyscf.gto")
    numint = pytest.importorskip("pyscf.dft.numint")
    basis = [
        [degree, [0.7 + 0.13 * degree, 0.7], [0.22 + 0.04 * degree, 0.3]]
        for degree in range(maximum + 1)
    ]
    molecule = gto.M(
        atom=[("H", center) for center in centers],
        basis={"H": basis},
        cart=cartesian,
        spin=len(centers) % 2,
        unit="Bohr",
        verbose=0,
    )
    normalization = 1 / np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    return numint.eval_ao(molecule, points, deriv=1) * normalization


def _decimal_weights(
    centers: np.ndarray,
    points: np.ndarray,
    owners: np.ndarray,
    atomic: np.ndarray,
    radii: np.ndarray | None,
    iterations: int,
    left: np.ndarray,
    right: np.ndarray,
    step: float,
) -> np.ndarray:
    """Four-corner primal differences in 70 digits, not production AD/log products."""
    with localcontext() as context:
        context.prec = 70
        convert = lambda array: [
            [Decimal(str(value)) for value in row] for row in array
        ]
        base_centers, base_points = convert(centers), convert(points)
        first, second = convert(left), convert(right)
        decimal_radii = (
            None if radii is None else [Decimal(str(value)) for value in radii]
        )
        delta = Decimal(str(step))

        def values(sign_left: int, sign_right: int) -> list:
            displacement = [
                [
                    delta * (sign_left * left_value + sign_right * right_value)
                    for left_value, right_value in zip(row_left, row_right, strict=True)
                ]
                for row_left, row_right in zip(first, second, strict=True)
            ]
            shifted = [
                [
                    coordinate + change
                    for coordinate, change in zip(center, motion, strict=True)
                ]
                for center, motion in zip(base_centers, displacement, strict=True)
            ]
            shifted_points = [
                [
                    coordinate + change
                    for coordinate, change in zip(
                        point, displacement[owner], strict=True
                    )
                ]
                for point, owner in zip(base_points, owners, strict=True)
            ]
            partition = decimal_partition(
                shifted_points, shifted, iterations, decimal_radii
            )
            return [row[owner] for row, owner in zip(partition, owners, strict=True)]

        positive_left, negative_left = values(1, 0), values(-1, 0)
        positive_right, negative_right = values(0, 1), values(0, -1)
        pp, pm, mp, mm = values(1, 1), values(1, -1), values(-1, 1), values(-1, -1)
        derivatives = [
            [
                (positive - negative) / (2 * delta)
                for positive, negative in zip(positive_left, negative_left, strict=True)
            ],
            [
                (positive - negative) / (2 * delta)
                for positive, negative in zip(
                    positive_right, negative_right, strict=True
                )
            ],
            [
                (plus_plus - plus_minus - minus_plus + minus_minus)
                / (4 * delta * delta)
                for plus_plus, plus_minus, minus_plus, minus_minus in zip(
                    pp, pm, mp, mm, strict=True
                )
            ],
        ]
        return (
            np.array([[float(value) for value in row] for row in derivatives]) * atomic
        )


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("cartesian", [False, True])
def test_mixed_geometry_against_independent_primal_controls(
    native: Any, version: int, cartesian: bool, tmp_path: Path
) -> None:
    with _owner(native, version, cartesian=cartesian) as (handle, shape):
        points, weights, atomic, owners, radii = _grid(native, handle, shape)
        partition = weights / atomic
        candidates = np.flatnonzero((partition > 0.1) & (partition < 0.9))
        assert candidates.size
        count = 7
        begin = min(int(candidates[len(candidates) // 2]), shape[2] - count)
        section = slice(begin, begin + count)
        status, actual = _read(native, handle, shape, begin, count)
        assert status == 0, native.host_mixed_error().decode()
        np.testing.assert_allclose(
            actual[0],
            _ao(CENTERS, points[section], 3, cartesian),
            atol=3e-14,
            rtol=2e-13,
        )
        np.testing.assert_array_equal(actual[2], weights[section])
        errors, primal_ao, weight_controls = [], [], []
        steps = [3e-3, 1e-3, 3e-4]
        for step in steps:
            corners = []
            for sign_left, sign_right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                displacement = step * (sign_left * LEFT + sign_right * RIGHT)
                corners.append(
                    _ao(
                        CENTERS + displacement,
                        points[section] + displacement[owners[section]],
                        3,
                        cartesian,
                    )
                )
            mixed = (corners[0] - corners[1] - corners[2] + corners[3]) / (
                4 * step * step
            )
            ao_control = np.stack(
                [
                    (corners[0] + corners[1] - corners[2] - corners[3]) / (4 * step),
                    (corners[0] - corners[1] + corners[2] - corners[3]) / (4 * step),
                    mixed,
                ]
            )
            reference_weights = _decimal_weights(
                CENTERS,
                points[section],
                owners[section],
                atomic[section],
                radii if version == 4 else None,
                3,
                LEFT,
                RIGHT,
                step,
            )
            errors.append(
                [
                    float(np.max(np.abs(actual[1] - ao_control))),
                    float(np.max(np.abs(actual[3] - reference_weights))),
                ]
            )
            primal_ao.append(corners)
            weight_controls.append(reference_weights)
        np.savez_compressed(
            tmp_path / "mixed-grid-controls.npz",
            centers=CENTERS,
            left=LEFT,
            right=RIGHT,
            points=points[section],
            owners=owners[section],
            atomic=atomic[section],
            radii=radii,
            steps=steps,
            errors=errors,
            ao=actual[0],
            moving_ao=actual[1],
            weights=actual[2],
            moving_weights=actual[3],
            primal_ao=primal_ao,
            weight_controls=weight_controls,
        )
        np.testing.assert_allclose(actual[1], ao_control, atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(actual[3], reference_weights, atol=3e-7, rtol=2e-6)
        assert errors[-1][0] < max(errors[0][0] / 12, 2e-8)
        assert errors[-1][1] < max(errors[0][1] / 12, 2e-9)


@pytest.mark.parametrize("iterations", [1, 2, 4, 5])
def test_partition_iteration_prescription_and_mixed_symmetry(
    native: Any, iterations: int
) -> None:
    with _owner(native, 4, iterations=iterations, maximum=1) as (handle, shape):
        points, _, atomic, owners, radii = _grid(native, handle, shape)
        begin, count = shape[2] // 3 + 47, 5
        status, actual = _read(native, handle, shape, begin, count)
        assert status == 0, native.host_mixed_error().decode()
        section = slice(begin, begin + count)
        reference = _decimal_weights(
            CENTERS,
            points[section],
            owners[section],
            atomic[section],
            radii,
            iterations,
            LEFT,
            RIGHT,
            1e-4,
        )
        np.testing.assert_allclose(actual[3], reference, atol=3e-8, rtol=2e-6)
        status, swapped = _read(native, handle, shape, begin, count, RIGHT, LEFT)
        assert status == 0, native.host_mixed_error().decode()
        np.testing.assert_allclose(actual[1][2], swapped[1][2], atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(actual[3][2], swapped[3][2], atol=2e-14, rtol=2e-13)


def test_linearity_translation_partial_tiles_and_direction_alias(native: Any) -> None:
    with _owner(native, maximum=2) as (handle, shape):
        status, baseline = _read(native, handle, shape, 55, 7)
        assert status == 0, native.host_mixed_error().decode()
        for start, size in ((55, 3), (58, 4)):
            status, part = _read(native, handle, shape, start, size)
            assert status == 0
            offset = start - 55
            np.testing.assert_allclose(
                part[1], baseline[1][:, :, offset : offset + size], atol=2e-15
            )
            np.testing.assert_allclose(
                part[3], baseline[3][:, offset : offset + size], atol=2e-15
            )
        status, scaled = _read(native, handle, shape, 55, 7, 2 * LEFT, -3 * RIGHT)
        assert status == 0
        np.testing.assert_allclose(scaled[1][2], -6 * baseline[1][2], atol=2e-14)
        np.testing.assert_allclose(scaled[3][2], -6 * baseline[3][2], atol=2e-14)
        translation = np.tile([0.2, -0.1, 0.05], (3, 1))
        status, shifted = _read(native, handle, shape, 55, 7, translation)
        assert status == 0
        np.testing.assert_allclose(shifted[1][[0, 2]], 0, atol=1e-14)
        np.testing.assert_allclose(shifted[3][[0, 2]], 0, atol=1e-14)
        alias = np.empty_like(baseline[0])
        alias.ravel()[:9] = LEFT.ravel()
        outputs = (
            alias,
            np.empty_like(baseline[1]),
            np.empty_like(baseline[2]),
            np.empty_like(baseline[3]),
        )
        status, aliased = _read(
            native, handle, shape, 55, 7, alias.ravel()[:9], outputs=outputs
        )
        assert status == 0
        for actual, expected in zip(aliased, baseline, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_caps_refusals_transactional_owner_alias_and_recovery(native: Any) -> None:
    with _owner(native, maximum=1) as (handle, shape):
        limits = (ct.c_size_t * 4)()
        native.host_mixed_plan(handle, limits)
        for index in range(4):
            denied = (ct.c_size_t * 4)(*limits)
            denied[index] -= 1
            assert native.host_mixed_bind(handle, denied) != 0
        assert native.host_mixed_bind(handle, limits) == 0
        for mode, after in ((1, 0), (2, 7), (3, 0)):
            assert native.host_mixed_bind(handle, limits) == 0
            native.host_mixed_fault(handle, mode, after)
            status, outputs = _read(native, handle, shape, 55, 7)
            assert status != 0
            for output in outputs:
                np.testing.assert_array_equal(output, -9713.5)
            native.host_mixed_fault(handle, 0, 0)
            assert _read(native, handle, shape, 55, 7)[0] == 0
        for kind in range(4):
            borrowed = native.host_mixed_borrowed(handle, kind)
            unsafe = np.ctypeslib.as_array(borrowed, shape=(7,))
            before = unsafe.copy()
            status, outputs = _read(native, handle, shape, 55, 7)
            assert status == 0
            status, _ = _read(
                native,
                handle,
                shape,
                55,
                7,
                outputs=(outputs[0], outputs[1], unsafe, outputs[3]),
            )
            assert status != 0
            np.testing.assert_array_equal(unsafe, before)
        bad = LEFT.copy()
        bad[1, 0] = np.nan
        status, outputs = _read(native, handle, shape, 55, 7, bad)
        assert status != 0
        for output in outputs:
            np.testing.assert_array_equal(output, -9713.5)
        assert _read(native, handle, shape, shape[2], 0)[0] == 0
        assert _read(native, handle, shape, shape[2], 1)[0] != 0
        status, outputs = _read(native, handle, shape, 55, 7)
        assert status == 0
        overlapping = outputs[3].ravel()[:7]
        saved = tuple(output.copy() for output in outputs)
        assert (
            _read(
                native,
                handle,
                shape,
                55,
                7,
                outputs=(outputs[0], outputs[1], overlapping, outputs[3]),
            )[0]
            != 0
        )
        for actual, expected in zip(outputs, saved, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_atomic_measure_identity_gate_and_recovery(native: Any) -> None:
    with _owner(native, maximum=1) as (handle, shape):
        _, weights, atomic, _, _ = _grid(native, handle, shape)
        candidate = int(
            np.flatnonzero((weights / atomic > 0.2) & (weights / atomic < 0.8))[0]
        )
        pointer = native.host_mixed_borrowed(handle, 3)
        saved = pointer[candidate]
        pointer[candidate] = saved * 1.001
        try:
            status, outputs = _read(native, handle, shape, candidate, 1)
            assert status != 0
            assert b"does not match" in native.host_mixed_error()
            for output in outputs:
                np.testing.assert_array_equal(output, -9713.5)
        finally:
            pointer[candidate] = saved
        assert _read(native, handle, shape, candidate, 1)[0] == 0


def test_signed_default_quadrature_measures_are_not_clipped_or_refused(
    native: Any, tmp_path: Path
) -> None:
    with _owner(native, version=4, maximum=1) as (handle, shape):
        points, weights, atomic, owners, radii = _grid(native, handle, shape)
        negative = np.flatnonzero(atomic < 0)
        assert negative.size
        begin = int(negative[len(negative) // 2])
        status, actual = _read(native, handle, shape, begin, 1)
        assert status == 0, native.host_mixed_error().decode()
        section = slice(begin, begin + 1)
        reference = _decimal_weights(
            CENTERS,
            points[section],
            owners[section],
            atomic[section],
            radii,
            3,
            LEFT,
            RIGHT,
            1e-4,
        )
        np.testing.assert_array_equal(actual[2], weights[section])
        np.testing.assert_allclose(actual[3], reference, atol=3e-8, rtol=2e-6)
        np.savez_compressed(
            tmp_path / "signed-grid-control.npz",
            points=points[section],
            atomic=atomic[section],
            owners=owners[section],
            native=actual[3],
            reference=reference,
        )


@pytest.mark.parametrize("atoms", [25, 40])
def test_explicit_fixed_tile_work_and_storage(native: Any, atoms: int) -> None:
    centers = np.array(
        [
            [1.7 * (atom % 5), 1.9 * ((atom // 5) % 4), 2.1 * (atom // 20)]
            for atom in range(atoms)
        ]
    )
    with _owner(native, maximum=0, centers=centers, numbers=(1,) * atoms, tile=3) as (
        handle,
        shape,
    ):
        direction = np.arange(3 * atoms, dtype=float).reshape(atoms, 3) * 0.0001
        status, outputs = _read(native, handle, shape, 45, 3, direction, -direction)
        assert status == 0, native.host_mixed_error().decode()
        points, _, atomic, owners, _ = _grid(native, handle, shape)
        reference = _decimal_weights(
            centers,
            points[45:48],
            owners[45:48],
            atomic[45:48],
            None,
            3,
            direction,
            -direction,
            1e-4,
        )
        np.testing.assert_allclose(outputs[3], reference, atol=3e-8, rtol=2e-6)
        work = (ct.c_size_t * 5)()
        native.host_mixed_work(handle, work)
        assert work[2] == 3 and work[3] == 3 * shape[1]
        assert work[4] == 4 * (atoms * (atoms - 1) // 2)
        resources = (ct.c_size_t * 4)()
        native.host_mixed_plan(handle, resources)
        assert resources[1] == 3 and resources[2] == 3 * shape[1]
        assert resources[3] == work[4]
        assert resources[0] < 512 * 1024


def test_single_atom_zero_geometry_and_separation_refusal(native: Any) -> None:
    with _owner(native, maximum=1, centers=CENTERS[:1], numbers=(1,)) as (
        handle,
        shape,
    ):
        status, outputs = _read(native, handle, shape, 3, 5, LEFT[:1], RIGHT[:1])
        assert status == 0
        np.testing.assert_array_equal(outputs[1], 0)
        np.testing.assert_array_equal(outputs[3], 0)
    coincident = CENTERS.copy()
    coincident[1] = coincident[0]
    with _owner(native, maximum=0, centers=coincident) as (handle, shape):
        status, outputs = _read(native, handle, shape, 3, 5)
        assert status != 0
        for output in outputs:
            np.testing.assert_array_equal(output, -9713.5)


def test_native_mixed_geometry_runs_without_python_or_compiler(
    native: Any, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(
        [str(native.standalone)],
        env={**os.environ, "PATH": str(empty), "PYTHONPATH": ""},
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )


def test_host_codegen_rejects_unknown_backend_and_has_no_runtime_dependency(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="backend"):
        generate("other")
    script = (
        "import importlib.abc, runpy, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:\n"
        "   raise RuntimeError('forbidden build-time dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = ['generator', '--backend', 'cpu', '--output', {str(tmp_path / 'host.hpp')!r}]\n"
        f"runpy.run_path({str(ROOT / 'tools/generate_grid_geometry_response.py')!r}, run_name='__main__')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
