"""Independent algebra and cached native probes for shared history emission.

CUDA schedule source is compiled as host code here; this is not GPU execution
evidence. Real-device consumer qualification remains a separate optional gate.
"""

from __future__ import annotations

import ctypes
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.ordered_history import (
    ordered_history_program,
    retained_history_schedule,
)
from generativeqc_compiler.tensor.ordered_history_emit import (
    CudaFlagFailure,
    emit_cholesky,
    emit_cpu_dot,
    emit_dot_loop,
)
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp_statement

if TYPE_CHECKING:
    from conftest import NativeCxx
    from generativeqc_compiler.tensor.program import Program


def _value(program: Program, **inputs: float) -> float:
    return float(
        next(
            iter(
                execute(
                    program, {name: np.asarray(value) for name, value in inputs.items()}
                ).outputs.values()
            )
        )
    )


def test_ordinary_tensor_stages_against_independent_weighted_history_algebra() -> None:
    rng = np.random.default_rng(23119)
    p = ordered_history_program()
    for n in (1, 2, 4, 8):
        d = n + 3
        df, residual, weights = (
            rng.normal(size=(n, d)),
            rng.normal(size=d),
            rng.uniform(1, 3, n),
        )
        rhs = np.zeros(n)
        beta = np.zeros((n, n))
        for i in range(n):
            total = 0.0
            for component in range(d):
                total = _value(
                    p.stage("dot_update"),
                    accumulator=total,
                    left=df[i, component],
                    right=residual[component],
                )
            rhs[i] = _value(p.stage("weighted_rhs"), omega=weights[i], value=total)
            for j in range(n):
                total = 0.0
                for component in range(d):
                    total = _value(
                        p.stage("dot_update"),
                        accumulator=total,
                        left=df[i, component],
                        right=df[j, component],
                    )
                weighted = _value(
                    p.stage("weighted_gram"),
                    row_weight=weights[i],
                    column_weight=weights[j],
                    overlap=total,
                )
                beta[i, j] = (
                    _value(p.stage("diagonal_shift"), value=weighted, omega_zero=0.01)
                    if i == j
                    else weighted
                )
        # Independent vectorized contractions, not a copy of the schedule loops.
        weighted_df = weights[:, None] * df
        np.testing.assert_allclose(rhs, weighted_df @ residual, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(
            beta, weighted_df @ weighted_df.T + 1e-4 * np.eye(n), rtol=2e-14, atol=2e-14
        )


@pytest.fixture(scope="module")
def native_history(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> ctypes.CDLL:
    directory = tmp_path_factory.mktemp("history-algebra-native")
    p = ordered_history_program()
    cpu, cuda = (
        retained_history_schedule(p, n) for n in ("compact-cpu", "capacity-cuda")
    )
    fold = emit_scalar_cpp_statement(
        p.stage("correction"),
        target="value",
        form="-=",
        bindings={
            "accumulator": "value",
            "omega": "weights[j]",
            "coefficient": "coefficients[j]",
            "update": "updates[j]",
        },
    )
    dot = emit_dot_loop(
        p,
        cuda,
        accumulator="sum",
        left="first[component]",
        right="second[component]",
        index="component",
        extent="count",
        failure=CudaFlagFailure(
            "record_error", "error", "DeviceError::nonfinite", "valid"
        ),
        indent=2,
    )
    source = directory / "probe.cpp"
    source.write_text(
        "\n".join(
            (
                "#include <cmath>",
                "#include <cstddef>",
                "#include <cstdint>",
                "namespace compact {",
                emit_cpu_dot(p, cpu),
                emit_cholesky(p, cpu),
                "}",
                "namespace strided { using std::sqrt; using std::isfinite;",
                emit_cholesky(p, cuda).replace("__device__ ", "", 1),
                "}",
                'extern "C" int solve_compact(double* a, double* b, std::int64_t n) { return compact::cholesky_solve(a,b,n); }',
                'extern "C" int solve_strided(double* a, double* b, std::int64_t n, std::int64_t ld) { return strided::cholesky_solve(a,b,n,ld); }',
                'extern "C" int dot_compact(const double* a, const double* b, std::int64_t n, double* result) { return compact::dot_product(a,b,n,*result); }',
                'extern "C" int dot_strided(const double* first, const double* second, std::int64_t count, double* result) {',
                "  using std::isfinite; double sum = 0.0; int valid = 1;",
                "  enum class DeviceError { nonfinite }; std::uint32_t error_storage = 0; auto* error = &error_storage;",
                "  auto record_error = [](std::uint32_t* output, DeviceError) { *output = 1; };",
                "  auto atomicExch = [](int* output, int value) { *output = value; };",
                dot,
                "  *result = sum; return valid;",
                "}",
                'extern "C" double correction(double value, const double* weights, const double* coefficients, const double* updates, std::int64_t n) {',
                "  for (std::int64_t j=0; j<n; ++j) {",
                "    " + fold,
                "  }",
                "  return value;",
                "}",
            )
        )
    )
    library = directory / "probe.so"
    required_native_cxx.build_shared(
        [source], library, compile_args=["-std=c++17", "-O2"]
    )
    result = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_double)
    result.solve_compact.argtypes = [ptr, ptr, ctypes.c_int64]
    result.solve_strided.argtypes = [ptr, ptr, ctypes.c_int64, ctypes.c_int64]
    for name in ("dot_compact", "dot_strided"):
        getattr(result, name).argtypes = [ptr, ptr, ctypes.c_int64, ptr]
    result.correction.argtypes = [ctypes.c_double, ptr, ptr, ptr, ctypes.c_int64]
    result.correction.restype = ctypes.c_double
    return result


def _ptr(array: np.ndarray) -> ctypes._Pointer[ctypes.c_double]:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.parametrize("n", (1, 2, 4, 8, 16, 64))
def test_generated_compact_and_strided_cholesky_against_independent_solve(
    native_history: ctypes.CDLL, n: int
) -> None:
    rng = np.random.default_rng(718 + n)
    vectors = rng.normal(size=(n, n + 7))
    weights = rng.uniform(1, 3, n)
    panel = vectors * weights[:, None]
    matrix = panel @ panel.T + 1e-4 * np.eye(n)
    rhs = panel @ rng.normal(size=n + 7)
    expected = np.linalg.solve(matrix, rhs)
    compact, cb = matrix.copy(), rhs.copy()
    assert native_history.solve_compact(_ptr(compact), _ptr(cb), n)
    np.testing.assert_allclose(cb, expected, rtol=3e-12, atol=3e-13)
    for stride in (n, n + 3):
        strided = np.full((n, stride), 123456.75)
        strided[:, :n] = matrix
        gb = rhs.copy()
        assert native_history.solve_strided(_ptr(strided), _ptr(gb), n, stride)
        np.testing.assert_array_equal(gb, cb)
        np.testing.assert_array_equal(strided[:, :n], compact)
        np.testing.assert_array_equal(strided[:, n:], 123456.75)
        residual = np.linalg.norm(matrix @ gb - rhs, ord=np.inf)
        assert residual <= 2e-13 * (
            np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(gb, ord=np.inf)
            + np.linalg.norm(rhs, ord=np.inf)
        )


def test_two_dimensional_solve_against_decimal_inverse(
    native_history: ctypes.CDLL,
) -> None:
    matrix = np.array([[2.125, -0.375], [-0.375, 1.0625]])
    rhs = np.array([0.75, -0.125])
    with localcontext() as context:
        context.prec = 80
        a, b, d, x, y = (
            Decimal.from_float(value)
            for value in (matrix[0, 0], matrix[0, 1], matrix[1, 1], rhs[0], rhs[1])
        )
        determinant = a * d - b * b
        expected = np.array(
            [float((d * x - b * y) / determinant), float((a * y - b * x) / determinant)]
        )
    assert native_history.solve_compact(_ptr(matrix), _ptr(rhs), 2)
    # A small solution component is cancellation-sensitive. Bound forward
    # error in ULPs of the solution's infinity norm, not componentwise relative
    # error against a tiny value.
    np.testing.assert_allclose(
        rhs, expected, rtol=0, atol=8 * np.spacing(np.max(np.abs(expected)))
    )


@pytest.mark.parametrize("backend", ("compact", "strided"))
@pytest.mark.parametrize(
    "fault",
    (
        "zero",
        "negative",
        "nonfinite",
        "offdiagonal-overflow",
        "rhs-overflow",
        "late-pivot",
    ),
)
def test_cholesky_rejects_unusable_values_without_repair(
    native_history: ctypes.CDLL, backend: str, fault: str
) -> None:
    matrix = np.eye(2)
    rhs = np.array([3.0, 4.0])
    if fault == "zero":
        matrix[0, 0] = 0
    elif fault == "negative":
        matrix[0, 0] = -1
    elif fault == "nonfinite":
        matrix[0, 0] = np.inf
    elif fault == "offdiagonal-overflow":
        matrix[0, 0], matrix[1, 0] = np.finfo(float).tiny, np.finfo(float).max
    elif fault == "rhs-overflow":
        matrix[0, 0], rhs[0] = np.finfo(float).tiny, np.finfo(float).max
    else:
        matrix[0, 0], matrix[1, 0], matrix[1, 1] = 4, 3, 1
    initial_rhs = rhs.copy()
    call = getattr(native_history, "solve_" + backend)
    arguments = (_ptr(matrix), _ptr(rhs), 2) + ((2,) if backend == "strided" else ())
    assert not call(*arguments)
    if fault != "rhs-overflow":
        np.testing.assert_array_equal(rhs, initial_rhs)
    if fault == "late-pivot":
        # Factorization writes scratch before failure and does not repair it.
        assert matrix[0, 0] == 2 and matrix[1, 0] == 1.5 and matrix[1, 1] == 1


def test_native_ordered_dot_overflow_and_failure_publication(
    native_history: ctypes.CDLL,
) -> None:
    first, second = np.array([1e16, 1.0, -1e16]), np.ones(3)
    for name in ("dot_compact", "dot_strided"):
        out = ctypes.c_double(73)
        assert getattr(native_history, name)(
            _ptr(first), _ptr(second), 3, ctypes.byref(out)
        )
        assert out.value == 0.0
    first = np.array([np.finfo(float).max, np.finfo(float).max, -np.finfo(float).max])
    for name in ("dot_compact", "dot_strided"):
        out = ctypes.c_double(73)
        assert not getattr(native_history, name)(
            _ptr(first), _ptr(second), 3, ctypes.byref(out)
        )
        assert out.value == 73 if name == "dot_compact" else np.isinf(out.value)


def test_native_seeded_correction_is_not_a_sum_then_subtract(
    native_history: ctypes.CDLL,
) -> None:
    weights, coefficients, updates = np.ones(2), np.ones(2), np.array([1e16, 1.0])
    actual = native_history.correction(
        1e16, _ptr(weights), _ptr(coefficients), _ptr(updates), 2
    )
    assert actual == -1.0
    assert 1e16 - np.dot(weights * coefficients, updates) == 0.0
    minus_zero = native_history.correction(
        -0.0, _ptr(weights), _ptr(coefficients), _ptr(np.zeros(2)), 2
    )
    assert minus_zero == 0 and np.signbit(minus_zero)


def test_chronological_overlay_oracle_is_independent_of_ring_formula(
    native_history: ctypes.CDLL,
) -> None:
    # Keep a chronological Python list; build ring storage independently by
    # insertion, and poison each slot being replaced before its tentative read.
    from generativeqc_compiler.tensor.ordered_history import chronological_slots

    for capacity in (1, 2, 4):
        chronological = []
        ring = [np.nan] * capacity
        head = 0
        for iteration in range(1, 3 * capacity + 3):
            tentative = float(iteration)
            chronological.append(tentative)
            chronological = chronological[-capacity:]
            ring[head] = np.nan
            new, slots = chronological_slots(iteration, capacity)
            assert new == head
            selected = [tentative if slot == new else ring[slot] for slot in slots]
            assert selected == chronological
            updates = np.asarray(selected)
            ones = np.ones(len(selected))
            actual = native_history.correction(
                1.25, _ptr(ones), _ptr(ones), _ptr(updates), len(selected)
            )
            expected = 1.25
            for item in chronological:
                expected -= item
            assert actual == expected
            ring[head] = tentative
            head = (head + 1) % capacity
