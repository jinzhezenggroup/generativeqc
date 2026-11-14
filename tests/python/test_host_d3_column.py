"""Independent complete D3 columns on the Python-free linked AOT consumer."""

from __future__ import annotations

import ctypes as ct
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from generativeqc_compiler.geometry.d3_column import D3_COLUMN_ARRAYS
from generativeqc_compiler.method._generated_parameters import d3_parameters

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
INTEGER_POINTER = ct.POINTER(ct.c_int32)
FIT = d3_parameters("B3LYP-D3(BJ)")
PARAMETERS = np.array([FIT[key] for key in ("s6", "s8", "a1", "a2", "s9")] + [0.0] * 3)
HEAVY = np.array([28, 53, 55, 1], dtype=np.int32)
POSITIONS = np.array([[0, 0, 0], [3.2, 0.3, 0.1], [-0.4, 4.5, 0.2], [0.1, -0.2, 2.1]])


def _pointer(values: np.ndarray) -> Any:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    """Build the production static target with only its four explicit D3 leaves."""
    directory = tmp_path_factory.mktemp("host-d3-column")
    inventory = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    inventory.update(spins=["unpolarized"], arrays=list(D3_COLUMN_ARRAYS))
    (directory / "inventory.json").write_text(json.dumps(inventory))
    (directory / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(native_d3_column LANGUAGES CXX)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT}/cmake/GenerativeQCGenerated.cmake")\n'
        f'include("{ROOT}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        f'generativeqc_add_host_hessian_aot(leaves "{directory}/inventory.json")\n'
        f'add_library(probe SHARED "{ROOT}/tests/native/host_d3_column_probe.cpp")\n'
        "target_link_libraries(probe PRIVATE leaves)\n"
        f'add_executable(standalone "{ROOT}/tests/native/host_d3_column_probe.cpp")\n'
        "target_compile_definitions(standalone PRIVATE GENERATIVEQC_D3_COLUMN_STANDALONE=1)\n"
        "target_link_libraries(standalone PRIVATE leaves)\n"
    )
    cmake = shutil.which("cmake")
    assert cmake is not None, "D3 column qualification requires CMake"
    build = directory / "build"
    for command, timeout in (
        (
            [
                cmake,
                "-S",
                str(directory),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DCMAKE_CXX_FLAGS=-Wall -Wextra -Werror",
                "-DCMAKE_CXX_COMPILER=" + required_native_cxx.compiler,
                "-DCMAKE_CXX_COMPILER_LAUNCHER=" + required_native_cxx.cache,
            ],
            180,
        ),
        ([cmake, "--build", str(build), "--parallel", "2", "--verbose"], 300),
    ):
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout
        )
        assert result.returncode == 0, result.stdout + result.stderr
        (directory / ("configure.log" if "-S" in command else "build.log")).write_text(
            result.stdout + result.stderr
        )
    loaded = cast("Any", ct.CDLL(str(build / "libprobe.so")))
    loaded.host_d3_error.restype = ct.c_char_p
    loaded.host_d3_plan.argtypes = [ct.c_size_t, SIZE_POINTER]
    loaded.host_d3_create.argtypes = [
        INTEGER_POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        SIZE_POINTER,
    ]
    loaded.host_d3_create.restype = ct.c_void_p
    loaded.host_d3_destroy.argtypes = [ct.c_void_p]
    loaded.host_d3_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_int,
    ]
    loaded.host_d3_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_d3_mutate_snapshot_inputs.argtypes = [ct.c_void_p]
    loaded._test_executable = build / "standalone"
    return loaded


def _plan(native: Any, atoms: int) -> np.ndarray:
    values = np.zeros(4, dtype=np.uintp)
    assert native.host_d3_plan(atoms, values.ctypes.data_as(SIZE_POINTER)) == 0
    return values


def _create(
    native: Any,
    numbers: np.ndarray = HEAVY,
    positions: np.ndarray = POSITIONS,
    parameters: np.ndarray = PARAMETERS,
    caps: np.ndarray | None = None,
) -> Any:
    if caps is None:
        caps = _plan(native, len(numbers))[[0, 2, 3]].copy()
    return native.host_d3_create(
        numbers.ctypes.data_as(INTEGER_POINTER),
        numbers.size,
        _pointer(positions),
        positions.size,
        _pointer(parameters),
        caps.ctypes.data_as(SIZE_POINTER),
    )


def _apply(native: Any, owner: Any, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.full(2, -781.25)
    gradient, action = np.full_like(right, -781.25), np.full_like(right, -781.25)
    assert (
        native.host_d3_apply(
            owner,
            _pointer(right),
            right.size,
            _pointer(gradient),
            gradient.size,
            _pointer(action),
            action.size,
            _pointer(value),
            0,
        )
        == 0
    ), native.host_d3_error().decode()
    work = np.zeros(5, dtype=np.uintp)
    native.host_d3_work(owner, work.ctypes.data_as(SIZE_POINTER))
    return np.concatenate((value, gradient.ravel(), action.ravel())), work


def _oracle(numbers: np.ndarray, positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Independent simple-dftd3; finite differences exist only in test controls."""
    interface = pytest.importorskip("dftd3.interface")
    damping = interface.RationalDampingParam(
        **{key: FIT[key] for key in ("s6", "s8", "a1", "a2", "s9")}
    )
    model = interface.DispersionModel(numbers, positions)
    model.set_realspace_cutoff(disp2=1e6, disp3=1e6, cn=1e6)
    result = model.get_dispersion(damping, grad=True)
    return float(result["energy"]), result["gradient"]


@pytest.mark.parametrize("atoms", [4, 25, 40])
def test_complete_columns_against_independent_multistep_gradients(
    native: Any, atoms: int, tmp_path: Path
) -> None:
    """D3-only endpoints: fixed 2e-8 curvature and 2e-11 primal/invariant gates."""
    numbers = np.resize(HEAVY, atoms).astype(np.int32)
    positions = (
        POSITIONS.copy()
        if atoms == 4
        else np.random.default_rng(atoms).normal(size=(atoms, 3))
        * (5 if atoms == 25 else 8)
    )
    right = np.random.default_rng(2845).normal(size=positions.shape) * 0.2
    owner = _create(native, numbers, positions)
    assert owner, native.host_d3_error().decode()
    try:
        started = time.perf_counter()
        actual, work = _apply(native, owner, right)
        seconds = time.perf_counter() - started
        energy, gradient = _oracle(numbers, positions)
        np.testing.assert_allclose(actual[0], energy, atol=2e-11, rtol=0)
        np.testing.assert_allclose(
            actual[2 : 2 + positions.size].reshape(positions.shape),
            gradient,
            atol=2e-11,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual[1], np.sum(gradient * right), atol=2e-11, rtol=0
        )
        steps = np.array([3e-3, 1e-3, 3e-4])
        displaced = np.array(
            [
                [
                    _oracle(numbers, positions + sign * step * right)[1]
                    for sign in (1, -1)
                ]
                for step in steps
            ]
        )
        controls = (displaced[:, 0] - displaced[:, 1]) / (2 * steps[:, None, None])
        errors = np.max(
            np.abs(controls - actual[2 + positions.size :].reshape(positions.shape)),
            axis=(1, 2),
        )
        assert errors[-1] < 2e-8
        assert errors[-1] < max(errors[0] / 8, 5e-11)
        bound = _plan(native, atoms)
        assert sum(int(work[index]) for index in (0, 2, 3)) <= bound[2]
        assert work[4] <= bound[3]
        np.savez(
            tmp_path / "d3-column-independent.npz",
            numbers=numbers,
            positions=positions,
            right=right,
            native=actual,
            work=work,
            plan=bound,
            steps=steps,
            oracle_energy=energy,
            oracle_gradient=gradient,
            displaced=displaced,
            errors=errors,
            seconds=seconds,
        )
    finally:
        native.host_d3_destroy(owner)


def test_raw_hessian_symmetry_rigid_motion_linearity_and_rotation(
    native: Any, tmp_path: Path
) -> None:
    owner = _create(native)
    assert owner
    try:
        dimension = POSITIONS.size
        matrix = np.column_stack(
            [
                _apply(native, owner, row.reshape(POSITIONS.shape))[0][2 + dimension :]
                for row in np.eye(dimension)
            ]
        )
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-11, rtol=0)
        steps = np.array([3e-4, 1e-4, 3e-5])
        controls = []
        for step in steps:
            controls.append(
                np.column_stack(
                    [
                        (
                            (
                                _oracle(
                                    HEAVY,
                                    POSITIONS + step * unit.reshape(POSITIONS.shape),
                                )[1]
                                - _oracle(
                                    HEAVY,
                                    POSITIONS - step * unit.reshape(POSITIONS.shape),
                                )[1]
                            )
                            / (2 * step)
                        ).ravel()
                        for unit in np.eye(dimension)
                    ]
                )
            )
        errors = np.max(np.abs(np.array(controls) - matrix), axis=(1, 2))
        assert np.max(errors) < 2e-8
        np.savez(
            tmp_path / "d3-full-raw-independent.npz",
            native=matrix,
            oracle=np.array(controls),
            steps=steps,
            errors=errors,
        )
        np.testing.assert_allclose(
            matrix @ np.tile(np.eye(3), (len(HEAVY), 1)), 0, atol=2e-11, rtol=0
        )
        right = np.random.default_rng(2945).normal(size=POSITIONS.shape)
        actual, _ = _apply(native, owner, right)
        np.testing.assert_allclose(
            matrix @ right.ravel(), actual[2 + dimension :], atol=2e-11, rtol=0
        )
        np.testing.assert_array_equal(
            _apply(native, owner, np.ones_like(right))[0][2 + dimension :], 0
        )
        rotation = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))[0]
        rotated = _create(
            native, positions=np.ascontiguousarray(POSITIONS @ rotation.T)
        )
        assert rotated
        try:
            expected = actual[2 + dimension :].reshape(right.shape) @ rotation.T
            np.testing.assert_allclose(
                _apply(native, rotated, np.ascontiguousarray(right @ rotation.T))[0][
                    2 + dimension :
                ].reshape(right.shape),
                expected,
                atol=2e-11,
                rtol=0,
            )
        finally:
            native.host_d3_destroy(rotated)
        native.host_d3_mutate_snapshot_inputs(owner)
        np.testing.assert_array_equal(_apply(native, owner, right)[0], actual)
    finally:
        native.host_d3_destroy(owner)


@pytest.mark.parametrize("failure", [1, 2, 3, 4])
def test_validation_actual_input_mutation_reentry_and_recovery(
    native: Any, failure: int
) -> None:
    owner = _create(native)
    assert owner
    try:
        right = np.ones_like(POSITIONS)
        gradient, action, value = (
            np.full_like(right, -781.25),
            np.full_like(right, -781.25),
            np.full(2, -781.25),
        )
        assert (
            native.host_d3_apply(
                owner,
                _pointer(right),
                right.size,
                _pointer(gradient),
                gradient.size,
                _pointer(action),
                action.size,
                _pointer(value),
                failure,
            )
            != 0
        )
        np.testing.assert_array_equal(gradient, -781.25)
        np.testing.assert_array_equal(action, -781.25)
        np.testing.assert_array_equal(value, -781.25)
        assert np.isfinite(_apply(native, owner, np.ones_like(right))[0]).all()
    finally:
        native.host_d3_destroy(owner)


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_required_numeric_and_semantic_caps(native: Any, cap: int) -> None:
    caps = _plan(native, len(HEAVY))[[0, 2, 3]].copy()
    caps[cap] -= 1
    assert not _create(native, caps=caps)
    assert b"ceiling" in native.host_d3_error()


@pytest.mark.parametrize("parameter", [4, 5, 6, 7])
def test_atm_and_finite_cutoffs_are_not_silently_dropped(
    native: Any, parameter: int
) -> None:
    parameters = PARAMETERS.copy()
    parameters[parameter] = 1
    assert not _create(native, parameters=parameters)
    assert b"uncut two-body BJ" in native.host_d3_error()


def test_isolated_atom_and_python_free_executable(native: Any) -> None:
    owner = _create(native, HEAVY[:1], POSITIONS[:1])
    assert owner
    try:
        np.testing.assert_array_equal(_apply(native, owner, np.ones((1, 3)))[0], 0)
    finally:
        native.host_d3_destroy(owner)
    result = subprocess.run(
        [str(native._test_executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={"PATH": "/no-python", "PYTHONPATH": "/no-python"},
    )
    assert "native AOT D3(BJ) two-body columns" in result.stdout


def test_invalid_shapes_nonfinite_alias_and_failure_recovery(native: Any) -> None:
    owner = _create(native)
    assert owner
    try:
        right = np.arange(POSITIONS.size, dtype=float).reshape(POSITIONS.shape) * 0.1
        expected, _ = _apply(native, owner, right)
        for invalid in (
            right.ravel()[:-1],
            np.full_like(right, np.nan),
            np.full_like(right, np.inf),
        ):
            gradient, action, value = (
                np.full_like(right, -781.25),
                np.full_like(right, -781.25),
                np.full(2, -781.25),
            )
            assert (
                native.host_d3_apply(
                    owner,
                    _pointer(invalid),
                    invalid.size,
                    _pointer(gradient),
                    gradient.size,
                    _pointer(action),
                    action.size,
                    _pointer(value),
                    0,
                )
                != 0
            )
            np.testing.assert_array_equal(gradient, -781.25)
            np.testing.assert_array_equal(action, -781.25)
        output, value = np.full_like(right, -781.25), np.full(2, -781.25)
        assert (
            native.host_d3_apply(
                owner,
                _pointer(right),
                right.size,
                _pointer(output),
                output.size,
                _pointer(output),
                output.size,
                _pointer(value),
                0,
            )
            != 0
        )
        alias, gradient = right.copy(), np.full_like(right, -781.25)
        assert (
            native.host_d3_apply(
                owner,
                _pointer(alias),
                alias.size,
                _pointer(gradient),
                gradient.size,
                _pointer(alias),
                alias.size,
                _pointer(value),
                0,
            )
            == 0
        )
        np.testing.assert_array_equal(alias.ravel(), expected[2 + right.size :])
        np.testing.assert_array_equal(_apply(native, owner, right)[0], expected)
    finally:
        native.host_d3_destroy(owner)


def test_dense_normalization_underflow_is_not_promoted(native: Any) -> None:
    """Keep the incumbent TensorIR normalization-domain refusal, not its fallback."""
    numbers = np.resize(HEAVY, 40).astype(np.int32)
    positions = np.random.default_rng(40).normal(size=(40, 3)) * 5
    owner = _create(native, numbers, positions)
    assert owner
    try:
        right = np.ones_like(positions)
        for _ in range(2):
            gradient, action, value = (
                np.full_like(right, -781.25),
                np.full_like(right, -781.25),
                np.full(2, -781.25),
            )
            assert (
                native.host_d3_apply(
                    owner,
                    _pointer(right),
                    right.size,
                    _pointer(gradient),
                    gradient.size,
                    _pointer(action),
                    action.size,
                    _pointer(value),
                    0,
                )
                != 0
            )
            assert b"underflow fallback" in native.host_d3_error()
            np.testing.assert_array_equal(gradient, -781.25)
            np.testing.assert_array_equal(action, -781.25)
            np.testing.assert_array_equal(value, -781.25)
    finally:
        native.host_d3_destroy(owner)


@pytest.mark.parametrize("atoms", [0, 4097, np.iinfo(np.uintp).max])
def test_plan_refuses_unrealizable_atom_extents(native: Any, atoms: int) -> None:
    output = np.full(4, 875, dtype=np.uintp)
    assert native.host_d3_plan(atoms, output.ctypes.data_as(SIZE_POINTER)) != 0
    np.testing.assert_array_equal(output, 875)


def test_invalid_bound_data_and_null_or_partial_output_aliases(native: Any) -> None:
    numbers = HEAVY.copy()
    numbers[0] = 999
    assert not _create(native, numbers)
    assert not _create(native, positions=POSITIONS.ravel()[:-1])
    parameters = PARAMETERS.copy()
    parameters[0] = np.nan
    assert not _create(native, parameters=parameters)
    positions = POSITIONS.copy()
    positions[0, 0] = np.inf
    assert not _create(native, positions=positions)
    owner = _create(native)
    assert owner
    try:
        right = np.ones_like(POSITIONS)
        gradient, action, value = (
            np.full_like(right, -781.25),
            np.full_like(right, -781.25),
            np.full(2, -781.25),
        )
        assert (
            native.host_d3_apply(
                owner,
                None,
                right.size,
                _pointer(gradient),
                gradient.size,
                _pointer(action),
                action.size,
                _pointer(value),
                0,
            )
            != 0
        )
        assert (
            native.host_d3_apply(
                owner,
                _pointer(right),
                right.size,
                None,
                gradient.size,
                _pointer(action),
                action.size,
                _pointer(value),
                0,
            )
            != 0
        )
        overlapping = np.full(right.size + 1, -781.25)
        assert (
            native.host_d3_apply(
                owner,
                _pointer(right),
                right.size,
                _pointer(overlapping[:-1]),
                right.size,
                _pointer(overlapping[1:]),
                right.size,
                _pointer(value),
                0,
            )
            != 0
        )
        np.testing.assert_array_equal(overlapping, -781.25)
        np.testing.assert_array_equal(gradient, -781.25)
        np.testing.assert_array_equal(action, -781.25)
        np.testing.assert_array_equal(value, -781.25)
        assert np.isfinite(_apply(native, owner, right)[0]).all()
    finally:
        native.host_d3_destroy(owner)
