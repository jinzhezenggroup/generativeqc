"""Independent physical f/g gates for the reusable compiler factor schedule."""

from __future__ import annotations

import ctypes
import shutil
from contextlib import nullcontext
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.factorized_matrix import emit_factorized_matrix_cpu
from generativeqc_compiler.integral.factorized_second import emit_factorized_second
from generativeqc_compiler.integral.shell_spec import cartesian_components

from tools.generativeqc_validation.second_derivatives import (
    _gaussian_squared_norm,
    libcint_eri_hessian,
)

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.array(
    (
        (0.13, -0.31, 0.24),
        (-0.43, 0.27, 0.51),
        (0.68, -0.14, -0.22),
        (-0.21, 0.48, -0.63),
    )
)
EXPONENTS = np.array((0.6, 0.8, 1.1, 0.9))
DIRECTION = np.array(
    ((0.03, -0.05, 0.02), (-0.07, 0.01, 0.04), (0.02, 0.03, -0.06), (0.01, -0.02, 0.05))
)


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Build once through the verified compiler-cache adapter, without a GPU."""
    return _library(tmp_path_factory.mktemp("factorized-second"), "full")


@pytest.fixture(scope="module")
def value_library(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Exercise the actual scalar-only emitted ABI, not a truncated full jet."""
    return _library(tmp_path_factory.mktemp("factorized-value"), "value")


@pytest.fixture(scope="module")
def matrix_library(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the exact build-only matrix source through the verified cache."""
    compiler = shutil.which("c++")
    assert compiler is not None, "C++ compiler is required for the matrix ABI gate"
    directory = tmp_path_factory.mktemp("factorized-matrix")
    source = directory / "matrix.cpp"
    source.write_text(
        emit_factorized_matrix_cpu()
        + r"""
extern "C" bool matrix(const unsigned* angular, unsigned component,
    const double* record, double* output) {
  const unsigned incidence[4]{0,1,2,3};
  return generativeqc::integrals::full_range_primitive_matrix_cpu(angular,component,record,incidence,3,output);
}
extern "C" bool matrix_owned(const unsigned* angular, unsigned component,
    const double* record, const unsigned* incidence, unsigned owners, double* output) {
  return generativeqc::integrals::full_range_primitive_matrix_cpu(angular,component,record,incidence,owners,output);
}
"""
    )
    output = directory / "matrix.so"
    result = CppCompilerAdapter(Path(compiler), compile_timeout=600).compile_shared(
        source, output, includes=(ROOT / "src",), options=("-ffp-contract=off",)
    )
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(output))
    loaded.matrix.restype = ctypes.c_bool
    loaded.matrix.argtypes = (
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )
    loaded.matrix_owned.restype = ctypes.c_bool
    loaded.matrix_owned.argtypes = (
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
    )
    return loaded


OWNER_PATTERNS = tuple(
    (head + (owners,), owners)
    for owners in range(1, 4)
    for head in product(range(owners + 1), repeat=3)
    if tuple(dict.fromkeys(owner for owner in head if owner < owners))
    == tuple(range(owners))
)


def _owner_projection(incidence: tuple, owners: int) -> np.ndarray:
    """Independent constant incidence from mathematical to physical coordinates."""
    transform = np.zeros((3 * owners, 9))
    for center in range(3):
        if incidence[center] < owners:
            for axis in range(3):
                transform[3 * incidence[center] + axis, 3 * center + axis] = 1
    return transform


def _owned_matrix_evaluate(
    library: ctypes.CDLL,
    angular: tuple,
    component: int,
    positions: np.ndarray,
    incidence: tuple,
    owners: int,
    exponents: np.ndarray = EXPONENTS,
) -> np.ndarray:
    """Exercise the dynamic 13/43/91-field operation and its exact write bound."""
    orders = np.asarray(angular, dtype=np.uint32)
    mapping = np.asarray(incidence, dtype=np.uint32)
    record = np.r_[exponents, positions.ravel()]
    fields = 1 + 3 * owners + 9 * owners**2
    output = np.full(fields + 1, -781.25)
    assert library.matrix_owned(
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        int(component),
        record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        owners,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert output[-1] == -781.25 and np.isfinite(output).all()
    return output[:-1]


@pytest.mark.parametrize(("incidence", "owners"), OWNER_PATTERNS)
@pytest.mark.parametrize(
    "angular", ((0, 0, 0, 0), (1, 1, 1, 1), (2, 1, 0, 2), (3, 2, 1, 0), (3, 3, 3, 3))
)
@pytest.mark.parametrize(
    "geometry",
    ("ordinary", "coincident", "signed-zero", "diffuse-tight", "translated", "near"),
)
def test_physical_owner_matrix_against_independent_analytic_hessian(
    matrix_library: ctypes.CDLL,
    angular: tuple,
    incidence: tuple,
    owners: int,
    geometry: str,
) -> None:
    """Independent ordered second partials cover sparse-axis composition.

    Fixed gates protect ordinary-scale fields; norm-scaled gates also protect
    tiny components. Distinct coincident owners remain independent coordinates.
    No native matrix symmetry, finite differences, or generated product rules
    participate in the oracle.
    """
    coordinates = POSITIONS.copy()
    exponents = EXPONENTS
    if geometry == "coincident":
        coordinates.fill(0.0)
    elif geometry == "signed-zero":
        coordinates = np.copysign(np.zeros((4, 3)), (-1.0) ** np.indices((4, 3)).sum(0))
    elif geometry == "diffuse-tight":
        exponents = np.array((0.004, 17.0, 0.023, 31.0))
    elif geometry == "translated":
        coordinates += (8.0, -11.0, 5.0)
    elif geometry == "near":
        coordinates *= 1e-6
    positions = np.asarray(
        [coordinates[owner if owner < owners else 3] for owner in incidence]
    )
    values, gradients = reference(0, angular, positions, exponents=exponents)
    hessian = libcint_eri_hessian(angular, exponents, positions).reshape(12, 12, -1)
    transform = _owner_projection(incidence, owners)
    gradient = transform @ gradients[:9]
    matrix = np.einsum("ai,ijk,bj->abk", transform, hessian[:9, :9], transform)
    dimension = 3 * owners
    for component in np.unique(np.linspace(0, values.size - 1, 5, dtype=int)):
        actual = _owned_matrix_evaluate(
            matrix_library, angular, component, positions, incidence, owners, exponents
        )
        targets = (
            (actual[:1], values[component : component + 1], 2e-11, 2e-11),
            (actual[1 : 1 + dimension], gradient[:, component], 2e-10, 2e-10),
            (actual[1 + dimension :], matrix[:, :, component].ravel(), 2e-9, 2e-8),
        )
        for returned, expected, absolute, relative in targets:
            np.testing.assert_allclose(returned, expected, atol=absolute, rtol=relative)
            np.testing.assert_allclose(
                returned,
                expected,
                atol=max(absolute * np.max(abs(expected)), 64 * np.nextafter(0.0, 1.0)),
                rtol=relative,
            )
        raw = actual[1 + dimension :].reshape(dimension, dimension)
        np.testing.assert_allclose(raw, raw.T, atol=2e-12, rtol=0)


@pytest.mark.parametrize(("incidence", "owners"), OWNER_PATTERNS)
def test_physical_owner_matrix_all_angular_support_bounds(
    matrix_library: ctypes.CDLL, incidence: tuple, owners: int
) -> None:
    """All s--f tuples publish finite jets without reading or writing dead bins.

    Analytic and independent gradient oracles are covered separately. This
    executable ABI sweep exercises every per-axis polynomial extent and owner
    partition, including mixed-axis components and the full Cartesian f tail.
    """
    positions = np.asarray(
        [POSITIONS[owner if owner < owners else 3] for owner in incidence]
    )
    dimension = 3 * owners
    for angular in product(range(4), repeat=4):
        components = int(
            np.prod([(degree + 1) * (degree + 2) // 2 for degree in angular])
        )
        for component in sorted({0, components // 2, components - 1}):
            actual = _owned_matrix_evaluate(
                matrix_library, angular, component, positions, incidence, owners
            )
            matrix = actual[1 + dimension :].reshape(dimension, dimension)
            np.testing.assert_allclose(matrix, matrix.T, atol=2e-12, rtol=0)


@pytest.mark.parametrize(("incidence", "owners"), OWNER_PATTERNS)
@pytest.mark.parametrize("exponent", (0.0, 1e-30))
def test_physical_owner_matrix_declines_invalid_or_overflowed_jets(
    matrix_library: ctypes.CDLL, incidence: tuple, owners: int, exponent: float
) -> None:
    """Structural-zero pruning must not hide a domain error or nonfinite jet."""
    angular = np.full(4, 3, dtype=np.uint32)
    mapping = np.asarray(incidence, dtype=np.uint32)
    record = np.r_[np.full(4, exponent), np.zeros(12)]
    fields = 1 + 3 * owners + 9 * owners**2
    output = np.full(fields + 1, -781.25)
    assert not matrix_library.matrix_owned(
        angular.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        0,
        record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        owners,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert output[-1] == -781.25


@pytest.mark.parametrize(("incidence", "owners"), OWNER_PATTERNS)
@pytest.mark.parametrize("angular", ((0, 0, 0, 0), (2, 1, 0, 2), (3, 3, 3, 3)))
def test_physical_owner_matrix_against_independent_primitive_gradients(
    matrix_library: ctypes.CDLL, angular: tuple, incidence: tuple, owners: int
) -> None:
    """All 14 owner partitions retain independent normalized-AO raw partials."""
    positions = np.asarray(
        [POSITIONS[owner if owner < owners else 3] for owner in incidence]
    )
    values, gradients = reference(0, angular, positions)
    transform = _owner_projection(incidence, owners)
    expected_gradient = transform @ gradients[:9]
    dimension = 3 * owners
    expected_matrix = np.zeros((dimension, dimension, values.size))
    step = 2e-4
    for column in range(dimension):
        for displacement, weight in ((-2, 1), (-1, -8), (1, 8), (2, -1)):
            moved = positions.copy()
            for center in range(4):
                if incidence[center] == column // 3:
                    moved[center, column % 3] += displacement * step
            shifted = reference(0, angular, moved)[1]
            expected_matrix[:, column] += (
                weight * (transform @ shifted[:9]) / (12 * step)
            )
    for component in np.unique(np.linspace(0, values.size - 1, 3, dtype=int)):
        actual = _owned_matrix_evaluate(
            matrix_library, angular, component, positions, incidence, owners
        )
        matrix = actual[1 + dimension :].reshape(dimension, dimension)
        np.testing.assert_allclose(actual[0], values[component], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(
            actual[1 : 1 + dimension],
            expected_gradient[:, component],
            atol=2e-10,
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            matrix, expected_matrix[:, :, component], atol=2e-9, rtol=2e-8
        )
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-12, rtol=0)
        shifted = _owned_matrix_evaluate(
            matrix_library,
            angular,
            component,
            positions + (0.5, -0.2, 0.8),
            incidence,
            owners,
        )
        np.testing.assert_allclose(actual, shifted, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize(("incidence", "owners"), OWNER_PATTERNS)
def test_physical_owner_matrix_does_not_merge_coincident_distinct_owners(
    matrix_library: ctypes.CDLL, incidence: tuple, owners: int
) -> None:
    """Owner labels, not numeric coordinates, define the independent motions."""
    positions = np.tile(POSITIONS[0], (4, 1))
    actual = _owned_matrix_evaluate(
        matrix_library, (1, 0, 1, 0), 4, positions, incidence, owners
    )
    canonical = _matrix_evaluate(matrix_library, (1, 0, 1, 0), 4, positions)
    transform = _owner_projection(incidence, owners)
    matrix = actual[1 + 3 * owners :].reshape(3 * owners, 3 * owners)
    assert np.max(abs(matrix)) > 1e-3
    np.testing.assert_allclose(actual[0], canonical[0], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        actual[1 : 1 + 3 * owners], transform @ canonical[1:10], atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        matrix,
        transform @ canonical[10:].reshape(9, 9) @ transform.T,
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize(
    "failure",
    (
        "zero_count",
        "large_count",
        "origin",
        "large_owner",
        "noncanonical",
        "unused",
        "geometry",
        "angular",
        "component",
        "exponent",
        "coordinate",
        "null_orders",
        "null_record",
        "null_mapping",
        "null_output",
    ),
)
def test_physical_owner_matrix_rejects_invalid_bindings(
    matrix_library: ctypes.CDLL, failure: str
) -> None:
    """Reject inconsistent physical incidence before writing any result fields."""
    orders = np.zeros(4, dtype=np.uint32)
    incidence = np.array((0, 1, 2, 3), dtype=np.uint32)
    record = np.r_[EXPONENTS, POSITIONS.ravel(), 1.0]
    owners, component = 3, 0
    output = np.full(92, -781.25)
    if failure == "zero_count":
        owners = 0
    elif failure == "large_count":
        owners = 4
    elif failure == "origin":
        incidence[3] = 2
    elif failure == "large_owner":
        incidence[0] = 4
    elif failure == "noncanonical":
        incidence[:] = (1, 0, 2, 2)
        owners = 2
    elif failure == "unused":
        incidence[:] = (0, 0, 0, 2)
        owners = 2
    elif failure == "geometry":
        incidence[:] = (0, 0, 0, 1)
        owners = 1
    elif failure == "angular":
        orders[0] = 4
    elif failure == "component":
        component = 1
    elif failure == "exponent":
        record[0] = 0
    elif failure == "coordinate":
        record[4] = np.nan
    arguments = [
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        component,
        record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        incidence.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        owners,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ]
    if failure.startswith("null_"):
        arguments[
            {"null_orders": 0, "null_record": 2, "null_mapping": 3, "null_output": 5}[
                failure
            ]
        ] = None
    assert not matrix_library.matrix_owned(*arguments)
    np.testing.assert_array_equal(output, -781.25)


def _matrix_evaluate(
    library: ctypes.CDLL,
    angular: tuple,
    component: int,
    positions: np.ndarray,
    exponents: np.ndarray = EXPONENTS,
) -> np.ndarray:
    """Retain the declared 91-field ABI and a trailing write-bound canary."""
    orders = np.asarray(angular, dtype=np.uint32)
    record = np.r_[exponents, positions.ravel(), 1.0]
    output = np.full(92, -718.125)
    assert library.matrix(
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        int(component),
        record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert output[-1] == -718.125 and np.isfinite(output).all()
    return output[:91]


@pytest.mark.parametrize(
    ("angular", "exponents"),
    (
        ((0, 0, 0, 0), EXPONENTS),
        ((1, 0, 0, 0), EXPONENTS),
        ((1, 1, 1, 1), EXPONENTS),
        ((2, 0, 0, 0), EXPONENTS),
        ((2, 1, 0, 2), EXPONENTS),
        ((3, 1, 2, 0), EXPONENTS),
        ((3, 3, 3, 3), EXPONENTS),
        ((2, 1, 2, 1), np.array((0.08, 0.31, 4.7, 12.0))),
    ),
)
def test_matrix_against_independent_normalized_primitive_gradients(
    matrix_library: ctypes.CDLL, angular: tuple, exponents: np.ndarray
) -> None:
    """Every raw partial matches normalized libcint and fourth-order differences."""
    values, gradients = reference(0, angular, POSITIONS, exponents=exponents)
    expected = np.zeros((9, 9, values.size))
    step = 2e-4
    for column in range(9):
        for displacement, weight in ((-2, 1), (-1, -8), (1, 8), (2, -1)):
            moved = POSITIONS.copy()
            moved.flat[column] += displacement * step
            shifted = reference(0, angular, moved, exponents=exponents)[1]
            expected[:, column] += weight * shifted[:9] / (12 * step)
    for component in np.unique(np.linspace(0, values.size - 1, 5, dtype=int)):
        actual = _matrix_evaluate(
            matrix_library, angular, component, POSITIONS, exponents
        )
        matrix = actual[10:].reshape(9, 9)
        np.testing.assert_allclose(actual[0], values[component], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(
            actual[1:10], gradients[:9, component], atol=2e-10, rtol=2e-10
        )
        np.testing.assert_allclose(
            matrix, expected[:, :, component], atol=2e-9, rtol=2e-8
        )
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-12, rtol=0)
        translated = _matrix_evaluate(
            matrix_library, angular, component, POSITIONS + (0.5, -0.2, 0.8), exponents
        )
        np.testing.assert_allclose(actual, translated, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("permutation", ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)))
def test_matrix_keeps_the_fourth_mathematical_center(
    matrix_library: ctypes.CDLL, permutation: tuple
) -> None:
    """Changing the fixed center recovers nonzero D partials, not a numeric zero."""
    actual = _matrix_evaluate(matrix_library, (0, 0, 0, 0), 0, POSITIONS)
    projection = np.vstack((np.eye(9), -np.tile(np.eye(3), (1, 3))))
    gradient = projection @ actual[1:10]
    matrix = projection @ actual[10:].reshape(9, 9) @ projection.T
    ordering = np.array(
        [3 * center + axis for center in permutation for axis in range(3)]
    )
    moved = _matrix_evaluate(
        matrix_library,
        (0, 0, 0, 0),
        0,
        POSITIONS[list(permutation)],
        EXPONENTS[list(permutation)],
    )
    assert np.max(np.abs(gradient[9:])) > 1e-3
    np.testing.assert_allclose(moved[0], actual[0], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        moved[1:10], gradient[ordering[:9]], atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        moved[10:].reshape(9, 9),
        matrix[np.ix_(ordering[:9], ordering[:9])],
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("failure", ("angular", "component", "exponent", "coordinate"))
def test_matrix_rejects_invalid_primitive_bindings(
    matrix_library: ctypes.CDLL, failure: str
) -> None:
    """Reject bad domains before writing the declared output or its canary."""
    angular = np.zeros(4, dtype=np.uint32)
    record = np.r_[EXPONENTS, POSITIONS.ravel(), 1.0]
    component = 0
    if failure == "angular":
        angular[2] = 4
    elif failure == "component":
        component = 1
    elif failure == "exponent":
        record[1] = 0
    else:
        record[7] = np.nan
    output = np.full(92, -718.125)
    assert not matrix_library.matrix(
        angular.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        component,
        record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    np.testing.assert_array_equal(output, -718.125)


def _library(directory: Path, output_mode: str) -> ctypes.CDLL:
    """Compile the requested canonical operation with verified cache ownership."""
    pytest.importorskip("pyscf")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = directory / "factorized.cpp"
    source.write_text(
        emit_factorized_second(output=output_mode)
        + r"""
extern "C" bool factorized(unsigned family, unsigned radial, const unsigned* angular,
    unsigned component, const double* record, const double* direction, double* output) {
  return generativeqc::scf::factorized_second::primitive(family,radial,angular,component,record,direction,output);
}
"""
    )
    output = directory / "factorized.so"
    result = CppCompilerAdapter(Path(compiler), compile_timeout=600).compile_shared(
        source, output, includes=(ROOT / "src",), options=("-ffp-contract=off",)
    )
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(output))
    loaded.factorized.restype = ctypes.c_bool
    loaded.factorized.argtypes = (
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        *(ctypes.POINTER(ctypes.c_double),) * 3,
    )
    return loaded


@pytest.mark.parametrize(
    ("family", "angular", "radial"),
    (
        (0, (0, 0, 0, 0), 0),
        (0, (2, 1, 0, 1), 0),
        (0, (3, 3, 3, 3), 0),
        (0, (1, 0, 1, 0), 1),
        (0, (1, 0, 1, 0), 2),
        (1, (3, 2, 4), 0),
        (2, (4, 4), 0),
    ),
)
def test_scalar_value_operation_matches_independent_integrals_and_exact_output_size(
    value_library: ctypes.CDLL, family: int, angular: tuple, radial: int
) -> None:
    """Nonzero direction does not change the explicitly scalar output ABI."""
    count = len(angular)
    expected = reference(family, angular, POSITIONS[:count], radial)[0].ravel()
    orders = np.array(angular, dtype=np.uint32)
    record = np.r_[EXPONENTS[:count], POSITIONS[:count].ravel(), 1.0]
    motion = np.ascontiguousarray(DIRECTION[:count])
    for component in np.unique(np.linspace(0, expected.size - 1, 7, dtype=int)):
        result = np.full(2, -781.25)
        assert value_library.factorized(
            family,
            radial,
            orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            int(component),
            record.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            motion.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        np.testing.assert_allclose(
            result[0], expected[component], atol=2e-11, rtol=2e-10
        )
        assert result[1] == -781.25


def evaluate(
    library: ctypes.CDLL,
    family: int,
    angular: tuple,
    component: int,
    radial: int = 0,
    *,
    exponents: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> tuple[bool, np.ndarray]:
    """Retain the physical record and independent-center packing of the ABI."""
    count = len(angular)
    orders = np.array(angular, dtype=np.uint32)
    record = np.r_[
        EXPONENTS[:count] if exponents is None else exponents,
        (POSITIONS[:count] if positions is None else positions).ravel(),
        1.0,
    ]
    motion = np.ascontiguousarray(DIRECTION[:count] if direction is None else direction)
    output = np.full(1 + 6 * count, np.nan)
    valid = library.factorized(
        family,
        radial,
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        component,
        *(
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            for array in (record, motion, output)
        ),
    )
    return valid, output


def reference(
    family: int,
    angular: tuple,
    positions: np.ndarray,
    radial: int = 0,
    *,
    exponents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use independent libcint integrals/gradients, not generated factor graphs."""
    from pyscf import gto

    exponents = EXPONENTS if exponents is None else exponents
    count = len(angular)
    names = ("H", "He", "Li", "Be")[:count]
    molecule = gto.M(
        atom=[
            (f"ghost-{name}", position)
            for name, position in zip(names, positions, strict=True)
        ],
        basis={
            name: [[order, [exponent, 1.0]]]
            for name, order, exponent in zip(names, angular, exponents, strict=False)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    labels = tuple(cartesian_components(order) for order in angular)
    shape = tuple(map(len, labels))
    norms = [
        _gaussian_squared_norm(exponent, component)
        for exponent, components in zip(exponents, labels, strict=False)
        for component in components
    ]
    scales = np.split(
        np.sqrt(molecule.intor("int1e_ovlp_cart").diagonal() / norms),
        np.cumsum(shape)[:-1],
    )
    normalization = np.ones(shape)
    for axis, scale in enumerate(scales):
        dimensions = [1] * count
        dimensions[axis] = len(scale)
        normalization *= scale.reshape(dimensions)
    with (
        molecule.with_range_coulomb(0.3 if radial == 1 else -0.3)
        if radial
        else nullcontext()
    ):
        operator = ("int2e", "int3c2e", "int2c2e")[family]
        value = (
            molecule.intor_by_shell(operator + "_cart", tuple(range(count)))
            / normalization
        )
        if family == 0:
            permutations = ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1))
            gradients = [
                -molecule.intor_by_shell("int2e_ip1_cart", order, comp=3).transpose(
                    0, *(1 + order.index(axis) for axis in range(4))
                )
                for order in permutations
            ]
        else:
            first = -molecule.intor_by_shell(
                operator + "_ip1_cart", tuple(range(count)), comp=3
            )
            if count == 2:
                second = -molecule.intor_by_shell(
                    "int2c2e_ip1_cart", (1, 0), comp=3
                ).transpose(0, 2, 1)
                gradients = [first, second]
            else:
                second = -molecule.intor_by_shell(
                    "int3c2e_ip1_cart", (1, 0, 2), comp=3
                ).transpose(0, 2, 1, 3)
                third = -molecule.intor_by_shell("int3c2e_ip2_cart", (0, 1, 2), comp=3)
                gradients = [first, second, third]
    return value.ravel(), (np.asarray(gradients) / normalization).reshape(3 * count, -1)


@pytest.mark.parametrize(
    "family,angular,radial",
    (
        (0, (0, 0, 0, 0), 0),
        (0, (1, 0, 1, 0), 0),
        (0, (2, 1, 0, 1), 0),
        (0, (3, 3, 3, 3), 0),
        (0, (3, 3, 3, 3), 1),
        (0, (3, 3, 3, 3), 2),
        (1, (0, 0, 0), 0),
        (1, (2, 1, 3), 0),
        (1, (3, 3, 4), 0),
        (2, (0, 0), 0),
        (2, (3, 4), 0),
        (2, (4, 4), 0),
    ),
)
def test_factorized_physical_value_gradient_jvp(
    library: ctypes.CDLL, family: int, angular: tuple, radial: int
) -> None:
    count = len(angular)
    value, gradient = reference(family, angular, POSITIONS[:count], radial)
    step = 2e-4
    displaced = [
        reference(
            family, angular, POSITIONS[:count] + sign * step * DIRECTION[:count], radial
        )[1]
        for sign in (-1, 1)
    ]
    coarse = (displaced[1] - displaced[0]) / (2 * step)
    displaced = [
        reference(
            family,
            angular,
            POSITIONS[:count] + sign * step / 2 * DIRECTION[:count],
            radial,
        )[1]
        for sign in (-1, 1)
    ]
    fine = (displaced[1] - displaced[0]) / step
    jvp = (4 * fine - coarse) / 3
    size = value.size
    for component in sorted({0, 1 if size > 1 else 0, size // 3, size // 2, size - 1}):
        valid, actual = evaluate(library, family, angular, component, radial)
        assert valid
        expected = np.r_[value[component], gradient[:, component], jvp[:, component]]
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-9)
        np.testing.assert_allclose(
            actual[1 : 1 + 3 * count].reshape(count, 3).sum(axis=0), 0, atol=3e-12
        )
        np.testing.assert_allclose(
            actual[1 + 3 * count :].reshape(count, 3).sum(axis=0), 0, atol=3e-12
        )


@pytest.mark.parametrize(
    "family,angular", ((0, (3, 2, 1, 0)), (1, (3, 2, 4)), (2, (4, 3)))
)
def test_factorized_translation_and_real_exponents(
    library: ctypes.CDLL, family: int, angular: tuple
) -> None:
    count = len(angular)
    valid, actual = evaluate(library, family, angular, 0, direction=np.ones((count, 3)))
    assert valid
    np.testing.assert_allclose(actual[1 + 3 * count :], 0, atol=2e-13)
    for exponent in (0, -0.2, np.nan, np.inf):
        exponents = EXPONENTS[:count].copy()
        exponents[-1] = exponent
        valid, _ = evaluate(library, family, angular, 0, exponents=exponents)
        assert not valid


def test_factorized_emission_is_finite_and_host_only() -> None:
    source = emit_factorized_second()
    assert source == emit_factorized_second()
    assert len(source) < 16 * 1024**2
    assert "PySCF" not in source and "cupy" not in source
    with pytest.raises(ValueError, match="backend"):
        emit_factorized_second(backend="python")
    with pytest.raises(ValueError, match="output"):
        emit_factorized_second(output="automatic")
