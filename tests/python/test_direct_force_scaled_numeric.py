"""Compile the emitted scalar contraction and compare independent ERI orbits."""

from __future__ import annotations

import ast
import ctypes as ct
import math
import shutil
import subprocess
from itertools import chain, product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _emitted_scalar_source() -> str:
    path = ROOT / "python/vibeqc_compiler/integral/lowering/dispatch.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_shell_class_fused_cuda"
    )
    template = next(
        node.value
        for node in owner.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "source"
            for target in node.targets
        )
        and isinstance(node.value, ast.JoinedStr)
    )
    # These scalar helpers are literal parts of the production f-string. AST
    # decoding unescapes its braces without importing or executing the emitter.
    # A future interpolated helper must update this fixture, not invent values.
    literal = "".join(
        part.value for part in template.values if isinstance(part, ast.Constant)
    )
    task_start = literal.index("struct GeneratedDpppShellTask {")
    task_end = literal.index("};", task_start) + 2
    begin = literal.index(
        "__device__ __forceinline__ std::size_t generated_dppp_matrix_index("
    )
    end = literal.index("/** Combine two reusable shell-pair records", begin)
    return literal[task_start:task_end] + "\n" + literal[begin:end]


@pytest.fixture(scope="module")
def coefficient_library(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for generated scalar checks")
    directory = tmp_path_factory.mktemp("direct-force-coefficients")
    source = directory / "coefficient.cpp"
    library = directory / "coefficient.so"
    source.write_text(
        "#include <cstddef>\n#include <cstdint>\n"
        "#define __device__\n#define __forceinline__ inline\n"
        + _emitted_scalar_source()
        + '''
extern "C" double scaled(int unrestricted, const double* density,
    std::size_t n, std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double coulomb, double exchange) {
  GeneratedDpppShellTask task{};
  task.matrix_order = static_cast<std::uint32_t>(n);
  task.density_offset = 3;
  task.spin_offset = 3 + n * n;
  return unrestricted
      ? generated_dppp_density_coefficient_scaled<true>(
          task, i, j, k, l, density, coulomb, exchange)
      : generated_dppp_density_coefficient_scaled<false>(
          task, i, j, k, l, density, coulomb, exchange);
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    native = ct.CDLL(str(library))
    native.scaled.argtypes = [
        ct.c_int,
        ct.POINTER(ct.c_double),
        *([ct.c_size_t] * 5),
        ct.c_double,
        ct.c_double,
    ]
    native.scaled.restype = ct.c_double
    return native


def _orbit_reference(
    matrices: tuple[list[float], list[float], list[float]],
    n: int,
    indices: tuple[int, int, int, int],
    unrestricted: bool,
    coulomb: float,
    exchange: float,
) -> float:
    total, alpha, beta = matrices
    i, j, k, l = indices
    orbit = dict.fromkeys(
        (
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        )
    )
    terms = []
    for a, b, c, d in orbit:
        ab, cd, ac, bd = a + b * n, c + d * n, a + c * n, b + d * n
        if coulomb:
            left = alpha[ab] + beta[ab] if unrestricted else total[ab]
            right = alpha[cd] + beta[cd] if unrestricted else total[cd]
            terms.append(0.5 * coulomb * left * right)
        if exchange:
            terms.append(
                0.5
                * exchange
                * (
                    alpha[ac] * alpha[bd] + beta[ac] * beta[bd]
                    if unrestricted
                    else total[ac] * total[bd]
                )
            )
    return math.fsum(terms)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize(
    "coefficients",
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, -0.5), (1.0, -1.0), (0.25, -0.1)],
)
def test_scaled_coefficient_matches_independent_unique_orbits(
    coefficient_library: ct.CDLL,
    unrestricted: bool,
    coefficients: tuple[float, float],
) -> None:
    n = 3
    matrices = tuple(
        [(min(i, j) + 2 * max(i, j) + 1) * scale for j in range(n) for i in range(n)]
        for scale in (0.13, 0.07, -0.02)
    )
    buffer = (ct.c_double * (3 + 3 * n * n))(
        -7.0, -8.0, -9.0, *chain.from_iterable(matrices)
    )
    for indices in product(range(n), repeat=4):
        actual = coefficient_library.scaled(
            unrestricted, buffer, n, *indices, *coefficients
        )
        expected = _orbit_reference(matrices, n, indices, unrestricted, *coefficients)
        assert actual == pytest.approx(expected, rel=2e-13, abs=2e-14)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("disabled", ["exchange", "coulomb", "both"])
def test_disabled_operator_never_evaluates_overflowing_intermediates(
    coefficient_library: ct.CDLL, unrestricted: bool, disabled: str
) -> None:
    n = 4
    total, alpha, beta = ([0.125] * (n * n) for _ in range(3))
    if disabled == "exchange":
        coefficients = (1.0, 0.0)
        if unrestricted:
            alpha[:] = [1e200] * (n * n)
            beta[:] = [-1e200] * (n * n)
        else:
            for i, j in ((0, 2), (1, 3), (0, 3), (1, 2)):
                total[i + j * n] = total[j + i * n] = 1e200
    elif disabled == "coulomb":
        coefficients = (0.0, -0.5)
        for matrix in (total, alpha, beta):
            for i, j in ((0, 1), (2, 3)):
                matrix[i + j * n] = matrix[j + i * n] = 1e308
    else:
        coefficients = (0.0, -0.0)
        for matrix in (total, alpha, beta):
            matrix[:] = [1e308] * (n * n)
    matrices = (total, alpha, beta)
    buffer = (ct.c_double * (3 + 3 * n * n))(
        -7.0, -8.0, -9.0, *chain.from_iterable(matrices)
    )
    indices = (0, 1, 2, 3)
    expected = _orbit_reference(matrices, n, indices, unrestricted, *coefficients)
    actual = coefficient_library.scaled(
        unrestricted, buffer, n, *indices, *coefficients
    )
    assert math.isfinite(expected)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-14)
