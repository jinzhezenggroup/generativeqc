"""Host regressions for the actual fused CUDA RSH density-weight block.

The production permutation helpers and weight statements are compiled unchanged;
only the CUDA annotation/header is stubbed. This is not a GPU integral or complete
force qualification. Real-device derivative and endpoint gates remain required.
"""

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def weights_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/scf/cuda/direct_jk_kernels.cu").read_text()
    kernel = source.index("__global__ void independent_rsh_derivative_kernel(")
    begin = source.index("    double j_weight =", kernel)
    end = source.index("    if (j_weight ==", begin)
    statements = source[begin:end]
    directory = tmp_path_factory.mktemp("cuda-rsh-weights")
    (directory / "cuda_runtime.h").write_text("#pragma once\n#define __device__\n")
    cpp, library = directory / "probe.cpp", directory / "probe.so"
    cpp.write_text(
        '#include "scf/cuda/direct_eri_symmetry.cuh"\n'
        "using namespace generativeqc::scf::cuda_execution;\n"
        'extern "C" void weights(std::size_t n, std::size_t offset,\n'
        "    std::size_t i, std::size_t j, std::size_t k, std::size_t l,\n"
        "    bool unrestricted, double cj, double short_ck, double long_ck,\n"
        "    const double* density, const double* beta, double* out) {\n"
        + statements
        + "    out[0] = j_weight; out[1] = short_weight; out[2] = long_weight;\n}\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I" + str(directory),
            "-I" + str(ROOT / "src"),
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    probe = ct.CDLL(str(library))
    probe.weights.argtypes = (
        [ct.c_size_t] * 6
        + [ct.c_bool]
        + [ct.c_double] * 3
        + [ct.POINTER(ct.c_double)] * 3
    )
    probe.weights.restype = None
    return probe


def evaluate(
    probe: ct.CDLL,
    a: np.ndarray,
    b: np.ndarray,
    quartet: tuple[int, int, int, int],
    coefficients: tuple[float, float, float],
    unrestricted: bool,
    *,
    offset: int = 0,
) -> np.ndarray:
    n = a.shape[0]
    alpha = np.concatenate((np.full(offset, -17.0), a.ravel()))
    beta = np.concatenate((np.full(offset, 23.0), b.ravel()))
    out = np.full(3, np.nan)
    probe.weights(
        n,
        offset,
        *quartet,
        unrestricted,
        *coefficients,
        *(x.ctypes.data_as(ct.POINTER(ct.c_double)) for x in (alpha, beta, out)),
    )
    return out


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("offset", [0, 9])
@pytest.mark.parametrize(
    "coefficients",
    [
        (1.0, -0.075, -0.5),
        (1.0, 0.0, 0.0),
        (0.0, -0.37, 0.0),
        (0.0, 0.0, -0.37),
    ],
)
def test_weights_match_distinct_ordered_quartets(
    weights_probe: ct.CDLL,
    unrestricted: bool,
    offset: int,
    coefficients: tuple[float, float, float],
) -> None:
    row, column = np.indices((3, 3))
    a = np.cos(0.3 * (row + column)) / 3
    b = np.sin(0.4 * (row + column)) / 6
    total = a + b if unrestricted else a
    pairs = [(i, j) for i in range(3) for j in range(i + 1)]
    cj, short_ck, long_ck = coefficients
    for first, (i, j) in enumerate(pairs):
        for k, l in pairs[: first + 1]:
            # Enumerate the distinct ordered domain independently of the helper.
            orbit = {
                (aa, bb, cc, dd)
                for (u, v), (w, x) in (((i, j), (k, l)), ((k, l), (i, j)))
                for aa, bb in ((u, v), (v, u))
                for cc, dd in ((w, x), (x, w))
            }
            j_weight = sum(
                0.5 * total[aa, bb] * total[cc, dd] for aa, bb, cc, dd in orbit
            )
            exchange = sum(
                0.5
                * (
                    a[aa, cc] * a[bb, dd]
                    + (b[aa, cc] * b[bb, dd] if unrestricted else 0.0)
                )
                for aa, bb, cc, dd in orbit
            )
            actual = evaluate(
                weights_probe,
                a,
                b,
                (i, j, k, l),
                coefficients,
                unrestricted,
                offset=offset,
            )
            np.testing.assert_allclose(
                actual,
                [cj * j_weight, short_ck * exchange, long_ck * exchange],
                rtol=1e-14,
                atol=1e-14,
            )


def test_disabled_exchange_does_not_overflow_spin_quadratic(
    weights_probe: ct.CDLL,
) -> None:
    a = np.array([[1.0, 1e200], [1e200, 1.0]])
    b = np.array([[1.0, -1e200], [-1e200, 1.0]])
    # Total density is finite and small; unused individual-spin squares overflow.
    actual = evaluate(weights_probe, a, b, (1, 1, 0, 0), (1.0, 0.0, 0.0), True)
    np.testing.assert_array_equal(actual, [4.0, 0.0, 0.0])


def test_disabled_coulomb_does_not_form_overflowing_total_density(
    weights_probe: ct.CDLL,
) -> None:
    a = np.array([[1e308, 1.0], [1.0, 1e308]])
    # This quartet's exchange uses only the unit off-diagonal elements.
    actual = evaluate(weights_probe, a, a, (1, 1, 0, 0), (0.0, -0.075, -0.5), True)
    np.testing.assert_allclose(actual, [0.0, -0.15, -1.0], rtol=0, atol=0)
