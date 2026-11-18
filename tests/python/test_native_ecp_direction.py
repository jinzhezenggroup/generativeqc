"""Independent primal oracles for bounded native residual-ECP nuclear sources."""

from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
INT_POINTER = ct.POINTER(ct.c_int)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
CENTERS = np.array([[0.13, -0.21, 0.17], [0.43, 0.19, 3.2], [-0.63, 0.24, -1.6]])
MOTION = np.array([[0.16, -0.09, 0.13], [-0.12, 0.07, -0.11], [0.03, -0.08, 0.14]])


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


def _terms(general: bool = False) -> np.ndarray:
    """Explicit synthetic test-only residuals, not an atomic-potential benchmark."""
    return np.array(
        [
            [center, channel, power, exponent, coefficient * scale]
            for center, scale in ((0, 1.0), (2, -0.8))
            for channel in range(-1, 4 if general else 3)
            for power in (range(5) if general else (2,))
            for exponent, coefficient in ((0.67, -0.012), (1.23, 0.0081))
        ],
        dtype=float,
    )


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Compile strict ABI glue only; every scientific path is in the actual core."""
    directory = tmp_path_factory.mktemp("ecp-direction")
    core = Path(policy_library._name).resolve()
    binary = required_native_cxx.build_shared(
        [ROOT / "tests/native/ecp_direction_probe.cpp"],
        directory / "probe.so",
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
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent), "-Wl,--no-undefined"],
    )
    loaded = ct.CDLL(str(binary))
    loaded.ecp_direction_error.restype = ct.c_char_p
    loaded.ecp_direction_create.argtypes = [
        POINTER,
        ct.c_size_t,
        INT_POINTER,
        INT_POINTER,
        ct.c_uint,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        POINTER,
        ct.c_size_t,
    ]
    loaded.ecp_direction_create.restype = ct.c_void_p
    loaded.ecp_direction_destroy.argtypes = [ct.c_void_p]
    loaded.ecp_direction_sizes.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.ecp_direction_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        SIZE_POINTER,
        ct.c_int,
    ]
    loaded.ecp_direction_failure.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int]
    loaded.ecp_direction_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.ecp_direction_raw_contract.argtypes = [ct.c_void_p, POINTER, POINTER]
    loaded.ecp_direction_gate_gaps.argtypes = [ct.c_void_p, POINTER]
    return loaded


@contextmanager
def _owner(
    native: Any,
    angular: int = 1,
    spherical: bool = True,
    general: bool = False,
    terms: np.ndarray | None = None,
    centers: np.ndarray = CENTERS,
    exponent_scale: float = 1.0,
) -> Any:
    if terms is None:
        terms = _terms(general)
    terms = np.ascontiguousarray(terms, dtype=float).reshape(-1, 5)
    centers = np.ascontiguousarray(centers, dtype=float)
    numbers = np.full(len(centers), 2, dtype=np.int32)
    cores = np.zeros(len(centers), dtype=np.int32)
    numbers[:3] = [11, 1, 11]
    cores[:3] = [10, 0, 10]
    handle = native.ecp_direction_create(
        _pointer(centers),
        len(centers),
        numbers.ctypes.data_as(INT_POINTER),
        cores.ctypes.data_as(INT_POINTER),
        angular,
        int(spherical),
        int(angular <= 1),
        exponent_scale,
        _pointer(terms),
        len(terms),
    )
    assert handle, native.ecp_direction_error().decode()
    try:
        sizes = (ct.c_size_t * 8)()
        native.ecp_direction_sizes(handle, sizes)
        yield handle, tuple(sizes), terms
    finally:
        native.ecp_direction_destroy(handle)


def _apply(
    native: Any, handle: Any, sizes: tuple, direction: np.ndarray = MOTION
) -> tuple:
    direction = np.ascontiguousarray(direction, dtype=float)
    output = np.full((2, sizes[0], sizes[0]), np.nan)
    assert (
        native.ecp_direction_apply(
            handle,
            _pointer(direction),
            direction.size,
            _pointer(output),
            output.size,
            None,
            0,
        )
        == 0
    ), native.ecp_direction_error().decode()
    work = (ct.c_size_t * 10)()
    native.ecp_direction_work(handle, work)
    assert work[8] <= sizes[3]
    assert work[0] <= sizes[4] and work[1] <= sizes[5]
    assert work[2] <= sizes[6] and work[3] <= sizes[7]
    np.testing.assert_array_equal(output, output.transpose(0, 2, 1))
    return output, tuple(work)


def _oracle(
    centers: np.ndarray, terms: np.ndarray, angular: int, spherical: bool
) -> np.ndarray:
    """Independent libcint primal residual operators, no native derivative seed."""
    gto = pytest.importorskip("pyscf.gto")
    first = [[0.8 + 0.1 * angular, 1]]
    second = [[0.6, 1]]
    if angular <= 1:
        first = [[0.8 + 0.1 * angular, 0.7], [0.23 + 0.02 * angular, 0.3]]
        second = [[0.6, 0.7], [0.18, 0.3]]
    result = []
    for part in range(2):
        ecp = {}
        for center, label in ((0, "Na0"), (2, "Na2")):
            channels = []
            for channel in sorted(
                {int(term[1]) for term in terms if int(term[0]) == center}
            ):
                if (channel == -1) != (part == 0):
                    continue
                powers = [[] for _ in range(5)]
                for atom, current, power, exponent, coefficient in terms:
                    if int(atom) == center and int(current) == channel:
                        powers[int(power)].append([exponent, coefficient])
                channels.append([channel, powers])
            ecp[label] = [10, channels]
        molecule = gto.M(
            atom=[("Na0", centers[0]), ("H1", centers[1]), ("Na2", centers[2])],
            # Omit Na2: its residual operator moves, but it carries no AO shell.
            basis={"Na0": [[angular, *first]], "H1": [[0, *second]]},
            ecp=ecp,
            cart=not spherical,
            unit="Bohr",
            spin=1,
            verbose=0,
        )
        normalization = 1 / np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
        result.append(
            molecule.intor("ECPscalar")
            * normalization[:, None]
            * normalization[None, :]
        )
    return np.array(result)


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("angular", range(4))
def test_native_direction_against_independent_primal_values(
    native: Any,
    angular: int,
    spherical: bool,
    general: bool,
    tmp_path: Path,
    record_testsuite_property: Any,
) -> None:
    with _owner(native, angular, spherical, general) as (handle, sizes, terms):
        assert sizes[2] == (2 if general else 1)
        actual, work = _apply(native, handle, sizes)
        errors, primals = [], []
        # Resolve the quadratic difference regime before subtraction noise dominates.
        steps = np.array([1e-2, 2e-3, 4e-4])
        for step in steps:
            positive = _oracle(CENTERS + step * MOTION, terms, angular, spherical)
            negative = _oracle(CENTERS - step * MOTION, terms, angular, spherical)
            numerical = (positive - negative) / (2 * step)
            errors.append(float(np.max(np.abs(actual - numerical))))
            primals.append([positive, negative])
        np.savez_compressed(
            tmp_path / "independent-ecp-direction.npz",
            centers=CENTERS,
            motion=MOTION,
            terms=terms,
            native=actual,
            primals=primals,
            steps=steps,
            errors=errors,
        )
        np.testing.assert_allclose(actual, numerical, atol=2e-8, rtol=4e-7)
        assert errors[-1] < max(errors[0] / 20, 3e-12)
        record_testsuite_property(
            f"ecp_l{angular}_{'sph' if spherical else 'cart'}_{'grid' if general else 'analytic'}",
            json.dumps({"sizes": sizes, "work": work, "errors": errors}),
        )


@pytest.mark.parametrize("general", [False, True])
def test_physical_coordinate_gate_and_raw_matrix_contract(
    native: Any, general: bool
) -> None:
    with _owner(native, general=general) as (handle, sizes, _):
        actual, _ = _apply(native, handle, sizes)
        raw = np.empty_like(actual)
        assert (
            native.ecp_direction_raw_contract(handle, _pointer(MOTION), _pointer(raw))
            == 0
        ), native.ecp_direction_error().decode()
        np.testing.assert_allclose(actual, raw, atol=3e-13, rtol=2e-12)
        zero, _ = _apply(native, handle, sizes, np.zeros_like(MOTION))
        np.testing.assert_array_equal(zero, 0)
        translation = np.broadcast_to([0.13, -0.21, 0.17], MOTION.shape).copy()
        translated, _ = _apply(native, handle, sizes, translation)
        np.testing.assert_array_equal(translated, 0)
        other = np.array(
            [[0.01, -0.04, 0.09], [0.02, 0.03, -0.06], [-0.05, 0.08, 0.02]]
        )
        second, _ = _apply(native, handle, sizes, other)
        combined, _ = _apply(native, handle, sizes, 1.3 * MOTION - 0.4 * other)
        np.testing.assert_allclose(
            combined, 1.3 * actual - 0.4 * second, atol=2e-13, rtol=2e-12
        )


@pytest.mark.parametrize("general", [False, True])
def test_limits_owner_publication_alias_and_recovery(
    native: Any, general: bool
) -> None:
    with _owner(native, general=general) as (handle, sizes, _):
        expected, work = _apply(native, handle, sizes)
        output = np.full_like(expected, -781.25)

        def reject(
            limits: Any = None,
            direction: Any = MOTION,
            count: int = 9,
            missing: int = 0,
        ) -> None:
            assert (
                native.ecp_direction_apply(
                    handle,
                    None if direction is None else _pointer(direction),
                    count,
                    _pointer(output),
                    output.size,
                    limits,
                    missing,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)

        for index, capacity in enumerate(sizes[3:]):
            if capacity:
                limits = list(sizes[3:])
                limits[index] -= 1
                reject((ct.c_size_t * 5)(*limits))
        reject(missing=1)
        reject(direction=None)
        reject(count=8)
        nonfinite = MOTION.copy()
        nonfinite[0, 0] = np.nan
        reject(direction=nonfinite)
        for at, epoch in ((1, 0), (work[-1], 0), (0, 1)):
            native.ecp_direction_failure(handle, at, epoch)
            reject()
            native.ecp_direction_failure(handle, 0, 0)
            restored, _ = _apply(native, handle, sizes)
            np.testing.assert_array_equal(restored, expected)
        output.ravel()[:9] = MOTION.ravel()
        assert (
            native.ecp_direction_apply(
                handle, _pointer(output), 9, _pointer(output), output.size, None, 0
            )
            == 0
        )
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("general", [False, True])
def test_basis_free_atoms_do_not_expand_derivative_storage(
    native: Any, general: bool
) -> None:
    with _owner(native, general=general) as (handle, sizes, _):
        expected, expected_work = _apply(native, handle, sizes)
    centers = np.vstack([CENTERS, np.full((126, 3), 50.0)])
    direction = np.vstack([MOTION, np.full((126, 3), 0.2)])
    with _owner(native, general=general, centers=centers) as (handle, extended, _):
        assert extended[1] == 129
        assert extended[3:] == sizes[3:]
        actual, work = _apply(native, handle, extended, direction)
        np.testing.assert_array_equal(actual, expected)
        assert work[:8] == expected_work[:8]


def test_empty_residual_is_an_explicit_zero_source(native: Any) -> None:
    with _owner(native, terms=np.empty((0, 5))) as (handle, sizes, _):
        assert sizes[2] == 0
        result, work = _apply(native, handle, sizes)
        np.testing.assert_array_equal(result, 0)
        assert not any(work[:8])


def test_unqualified_quadrature_rejects_even_a_zero_direction(native: Any) -> None:
    terms = _terms(True)
    terms[:, 4] *= 1e10
    with _owner(native, terms=terms) as (handle, sizes, _):
        output = np.full((2, sizes[0], sizes[0]), -781.25)
        original = output.copy()
        assert (
            native.ecp_direction_raw_contract(
                handle, _pointer(MOTION), _pointer(output)
            )
            != 0
        )
        assert b"convergence gate failed" in native.ecp_direction_error()
        output[:] = original
        zero = np.zeros_like(MOTION)
        assert (
            native.ecp_direction_apply(
                handle,
                _pointer(zero),
                zero.size,
                _pointer(output),
                output.size,
                None,
                0,
            )
            != 0
        )
        assert b"convergence gate failed" in native.ecp_direction_error()
        np.testing.assert_array_equal(output, original)


def test_native_source_executes_without_python_on_path(
    tmp_path: Path, required_native_cxx: Any, policy_library: Any
) -> None:
    core = Path(policy_library._name).resolve()
    executable = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_ecp_direction.cpp"],
        tmp_path / "native-source",
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
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )
    empty = tmp_path / "no-python"
    empty.mkdir()
    result = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PATH": str(empty),
            "PYTHONPATH": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert "analytic/checked-grid, Cartesian/spherical" in result.stdout


def test_gradient_gate_cannot_be_hidden_by_a_zero_projection(
    native: Any, record_testsuite_property: Any
) -> None:
    """Construct a real derivative-only grid failure, not a mocked gate predicate."""
    for exponent_scale in (40.0, 160.0, 640.0, 2560.0):
        with _owner(native, general=True, exponent_scale=exponent_scale) as (
            handle,
            _,
            terms,
        ):
            gaps = np.zeros(4)
            assert native.ecp_direction_gate_gaps(handle, _pointer(gaps)) == 0
        primal, derivative = max(gaps[:2]), max(gaps[2:])
        if derivative <= 15 * primal or not primal:
            continue
        scale = 0.5 * (2e-9 / primal + 2e-8 / derivative)
        terms = terms.copy()
        terms[:, 4] *= scale
        with _owner(native, terms=terms, exponent_scale=exponent_scale) as (
            handle,
            sizes,
            _,
        ):
            actual_gaps = np.zeros(4)
            assert native.ecp_direction_gate_gaps(handle, _pointer(actual_gaps)) == 0
            assert max(actual_gaps[:2]) < 2e-9
            assert max(actual_gaps[2:]) > 2e-8
            output = np.full((2, sizes[0], sizes[0]), -781.25)
            assert (
                native.ecp_direction_raw_contract(
                    handle, _pointer(MOTION), _pointer(output)
                )
                != 0
            )
            assert b"convergence gate failed" in native.ecp_direction_error()
            zero = np.zeros_like(MOTION)
            output.fill(-781.25)
            assert (
                native.ecp_direction_apply(
                    handle,
                    _pointer(zero),
                    zero.size,
                    _pointer(output),
                    output.size,
                    None,
                    0,
                )
                != 0
            )
            assert b"convergence gate failed" in native.ecp_direction_error()
            np.testing.assert_array_equal(output, -781.25)
            record_testsuite_property(
                "unprojected_derivative_gate",
                json.dumps(
                    {
                        "exponent_scale": exponent_scale,
                        "coefficient_scale": scale,
                        "gaps": actual_gaps.tolist(),
                    }
                ),
            )
            return
    pytest.fail(
        "sharp-AO controls did not exercise a derivative-only quadrature refusal"
    )
