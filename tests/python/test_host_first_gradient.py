"""Independent all-component gates for the native gradient-only first source."""

from __future__ import annotations

import ctypes as ct
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_factorized_second import EXPONENTS, POSITIONS, reference

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
ORDERS = ct.POINTER(ct.c_uint)
SCENARIOS = (
    (POSITIONS, EXPONENTS),
    (np.zeros((4, 3)), EXPONENTS),
    (POSITIONS * 0.3, np.array((0.003, 5.0, 0.017, 9.0))),
    (POSITIONS * 1e-5 + (0.2, -0.3, 0.1), np.array((0.14, 0.31, 0.67, 1.2))),
    (POSITIONS + (4.0, -2.0, 1.0), EXPONENTS),
)


@pytest.fixture(scope="module")
def gradient_native(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Any:
    """Call the actual core owner through a small C test ABI, not a fake kernel."""
    directory = tmp_path_factory.mktemp("host-first-gradient")
    core = Path(policy_library._name).resolve()
    path = required_native_cxx.build_shared(
        [ROOT / "tests/native/host_first_gradient_probe.cpp"],
        directory / "gradient.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )
    library = ct.CDLL(str(path))
    library.host_packaged_gradient.argtypes = [ORDERS, ct.c_uint, POINTER, POINTER]
    library.host_packaged_gradient.restype = ct.c_int
    library.host_packaged_gradient_available.restype = ct.c_bool
    return library


@pytest.mark.parametrize("angular", tuple(product(range(3), repeat=4)))
@pytest.mark.parametrize("scenario", range(len(SCENARIOS)))
def test_complete_first_gradient_components_against_libcint(
    gradient_native: Any, angular: tuple[int, ...], scenario: int
) -> None:
    """Every s/p/d Cartesian component retains all independent center partials."""
    if not gradient_native.host_packaged_gradient_available():
        pytest.skip("packaged CPU first-derivative artifact is unavailable")
    positions, exponents = SCENARIOS[scenario]
    # Reorigin only the independent oracle too: raw libcint product-center
    # subtraction can lose tiny separations near a nonzero common origin.
    _, expected = reference(0, angular, positions - positions[3], exponents=exponents)
    orders = np.asarray(angular, dtype=np.uint32)
    record = np.ascontiguousarray(np.r_[exponents, positions.ravel()])
    original = record.copy()
    for component in range(expected.shape[1]):
        output = np.full(14, -718.125)
        assert (
            gradient_native.host_packaged_gradient(
                orders.ctypes.data_as(ORDERS),
                component,
                record.ctypes.data_as(POINTER),
                output[1:].ctypes.data_as(POINTER),
            )
            == 0
        )
        assert output[0] == output[-1] == -718.125
        for weight in (1.0, -0.37):
            actual, target = weight * output[1:-1], weight * expected[:, component]
            norm = float(np.max(abs(target)))
            np.testing.assert_allclose(actual, target, atol=3e-11, rtol=3e-11)
            np.testing.assert_allclose(
                actual,
                target,
                atol=max(3e-11 * norm, 64 * np.nextafter(0.0, 1.0)),
                rtol=3e-11,
            )
            np.testing.assert_allclose(
                actual.reshape(4, 3).sum(axis=0),
                0,
                atol=3e-11 * (1 + norm),
                rtol=3e-11,
            )
    np.testing.assert_array_equal(record, original)


@pytest.mark.parametrize("angular", ((3, 0, 0, 0), (2, 3, 2, 3), (4, 0, 0, 0)))
def test_unsupported_gradient_class_publishes_nothing(
    gradient_native: Any, angular: tuple[int, ...]
) -> None:
    """Capability misses cannot select the stationary three-pass f recurrence."""
    orders = np.asarray(angular, dtype=np.uint32)
    record = np.ascontiguousarray(np.r_[EXPONENTS, POSITIONS.ravel()])
    output = np.full(14, -718.125)
    assert (
        gradient_native.host_packaged_gradient(
            orders.ctypes.data_as(ORDERS),
            0,
            record.ctypes.data_as(POINTER),
            output[1:].ctypes.data_as(POINTER),
        )
        == 1
    )
    np.testing.assert_array_equal(output, -718.125)


@pytest.mark.parametrize("angular", ((0, 0, 0, 0), (2, 1, 0, 2)))
@pytest.mark.parametrize(
    "failure", ("zero", "negative", "nan", "infinite", "overflow", "component")
)
def test_invalid_first_gradient_is_not_a_capability_miss(
    gradient_native: Any, angular: tuple[int, ...], failure: str
) -> None:
    """Domain/finite failures retain the output and forbid silent fallback."""
    if not gradient_native.host_packaged_gradient_available():
        pytest.skip("packaged CPU first-derivative artifact is unavailable")
    orders = np.asarray(angular, dtype=np.uint32)
    record = np.ascontiguousarray(np.r_[EXPONENTS, POSITIONS.ravel()])
    component = 0
    if failure == "zero":
        record[0] = 0
    elif failure == "negative":
        record[0] = -1
    elif failure == "nan":
        record[6] = np.nan
    elif failure == "infinite":
        record[0] = np.inf
    elif failure == "overflow":
        record[:4] = 1e308
    else:
        component = 10001
    output = np.full(14, -718.125)
    assert (
        gradient_native.host_packaged_gradient(
            orders.ctypes.data_as(ORDERS),
            component,
            record.ctypes.data_as(POINTER),
            output[1:].ctypes.data_as(POINTER),
        )
        == -1
    )
    np.testing.assert_array_equal(output, -718.125)


@pytest.mark.parametrize("argument", ("angular", "record", "output"))
def test_null_first_gradient_arguments_are_rejected(
    gradient_native: Any, argument: str
) -> None:
    """Invalid pointers are errors even when the packaged artifact is absent."""
    orders = np.zeros(4, dtype=np.uint32)
    record = np.ascontiguousarray(np.r_[EXPONENTS, POSITIONS.ravel()])
    output = np.full(14, -718.125)
    pointers = [
        orders.ctypes.data_as(ORDERS),
        record.ctypes.data_as(POINTER),
        output[1:].ctypes.data_as(POINTER),
    ]
    pointers[("angular", "record", "output").index(argument)] = None
    assert (
        gradient_native.host_packaged_gradient(pointers[0], 0, pointers[1], pointers[2])
        == -1
    )
    np.testing.assert_array_equal(output, -718.125)
