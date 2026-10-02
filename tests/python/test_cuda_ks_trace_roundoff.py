"""Stress the actual scalar KS diagnostic body against exact analytic sums.

This host harness removes only the CUDA entry annotation; real-device molecular
SCF and force gates remain separate qualification requirements.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def diagnostic_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the production reduction, not an independently rewritten sum."""
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/dft/cuda_ks_kernels.cu").read_text()
    start = source.find("__device__ void accumulate_diagnostic_trace")
    if start < 0:
        start = source.index("__global__ void diagnostic_kernel")
    body = source[start : source.index("__global__ void advance_kernel")]
    header = (ROOT / "src/dft/cuda_ks_kernels.hpp").read_text()
    descriptor = header[
        header.index("struct Scalars {") : header.index("struct Control {")
    ]
    directory = tmp_path_factory.mktemp("cuda-ks-trace")
    path = directory / "probe.cpp"
    path.write_text(
        "#include <cmath>\n#include <cstddef>\n#include <cstdint>\n"
        "#include <cstdlib>\n#include <iomanip>\n#include <iostream>\n"
        "#include <vector>\n#define __global__\n#define __device__\n"
        "using std::isfinite;\n"
        "double __dadd_rn(double left, double right) { return left + right; }\n"
        + descriptor
        + body
        + r"""
int main(int argc, char** argv) {
  const std::size_t matrix = std::strtoul(argv[1], nullptr, 10);
  const unsigned spins = std::strtoul(argv[2], nullptr, 10);
  std::vector<double> density(spins * matrix, 1.0), residual(spins * matrix, 0.0);
  std::vector<double> hcore(matrix), overlap(matrix, 0.0), coulomb(matrix);
  std::vector<double> exchange(spins * matrix), range_exchange(spins * matrix);
  const double reference = (matrix / 3) * 0x1p-54 * spins;
  for (std::size_t index = 0; index < matrix; ++index) {
    const double term = index % 3 == 0 ? 1.0 : index % 3 == 1 ? 0x1p-54 : -1.0;
    hcore[index] = term;
    coulomb[index] = 2 * term;
    for (unsigned spin = 0; spin < spins; ++spin) {
      exchange[spin * matrix + index] = -8 * term;
      range_exchange[spin * matrix + index] = -4 * term;
    }
  }
  const double totals[3] = {};
  const int status[2] = {};
  Scalars output;
  diagnostic_kernel(matrix, spins, density.data(), density.data(), residual.data(),
                    hcore.data(), overlap.data(), coulomb.data(), exchange.data(), -0.25,
                    range_exchange.data(), -0.5, totals, status, status, nullptr,
                    nullptr, nullptr, status, nullptr, &output);
  std::cout << std::setprecision(17)
            << output.one_electron - reference << ' '
            << output.hartree - reference << ' '
            << output.exact_exchange - 2 * reference << ' '
            << output.failure << '\n';
}
"""
    )
    executable = directory / "probe"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    return executable


@pytest.mark.parametrize("aos", (24, 384, 768))
@pytest.mark.parametrize("spins", (1, 2))
def test_diagnostic_energy_traces_do_not_lose_sub_ulp_terms(
    diagnostic_probe: Path, aos: int, spins: int
) -> None:
    completed = subprocess.run(
        [str(diagnostic_probe), str(aos * aos), str(spins)],
        check=True,
        capture_output=True,
        text=True,
    )
    one, hartree, exchange, failure = map(float, completed.stdout.split())
    assert failure == 0
    assert (one, hartree, exchange) == pytest.approx((0, 0, 0), abs=1e-15)


@pytest.fixture(scope="module")
def cuda_diagnostic_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the unmodified CUDA reduction with the required cache launcher."""
    if os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    cupy = pytest.importorskip("cupy")
    cache, compiler = shutil.which("ccache"), shutil.which("nvcc")
    if cache is None or compiler is None:
        pytest.fail("CUDA diagnostic qualification requires ccache and nvcc")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("cuda-ks-trace-device")
    launcher = directory / "nvcc"
    launcher.write_text(f'#!/bin/sh\nexec "{cache}" "{compiler}" "$@"\n')
    launcher.chmod(0o755)
    source = (ROOT / "src/dft/cuda_ks_kernels.cu").read_text()
    start = source.index("__device__ void accumulate_diagnostic_trace")
    body = source[start : source.index("__global__ void advance_kernel")]
    header = (ROOT / "src/dft/cuda_ks_kernels.hpp").read_text()
    descriptor = header[
        header.index("struct Scalars {") : header.index("struct Control {")
    ]
    code = (
        "#include <cstddef>\n#include <cstdint>\n#include <cmath>\n"
        + descriptor
        + "static_assert(sizeof(Scalars) <= 128);\n"
        + "static_assert(offsetof(Scalars, failure) == 14 * sizeof(double));\n"
        + body.replace(
            "__global__ void diagnostic_kernel",
            'extern "C" __global__ void diagnostic_kernel',
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NVCC", str(launcher))
        patch.setenv("CCACHE_BASEDIR", str(ROOT))
        module = cupy.RawModule(code=code, backend="nvcc", options=("-std=c++17",))
        kernel = module.get_function("diagnostic_kernel")
    return kernel


@pytest.mark.parametrize("aos", (24, 384, 768))
@pytest.mark.parametrize("spins", (1, 2))
def test_cuda_diagnostic_traces_match_exact_sum(
    cuda_diagnostic_probe: Any, aos: int, spins: int
) -> None:
    """Check NVCC's actual FP64 contraction/rounding behavior on the GPU."""
    import cupy as cp
    import numpy as np

    matrix = aos * aos
    terms = cp.tile(cp.asarray([1.0, 2.0**-54, -1.0]), matrix // 3)
    density = cp.ones(spins * matrix)
    residual = cp.zeros_like(density)
    overlap = cp.zeros(matrix)
    coulomb = 2 * terms
    exchange = -8 * cp.tile(terms, spins)
    range_exchange = -4 * cp.tile(terms, spins)
    totals = cp.zeros(3)
    status = cp.zeros(2, dtype=cp.int32)
    output = cp.zeros(16)
    cuda_diagnostic_probe(
        (1,),
        (1,),
        (
            np.uint64(matrix),
            np.uint32(spins),
            density,
            density,
            residual,
            terms,
            overlap,
            coulomb,
            exchange,
            np.float64(-0.25),
            range_exchange,
            np.float64(-0.5),
            totals,
            status,
            status,
            np.uint64(0),
            np.uint64(0),
            np.uint64(0),
            status,
            np.uint64(0),
            output,
        ),
    )
    reference = (matrix // 3) * 2.0**-54 * spins
    np.testing.assert_allclose(
        cp.asnumpy(output[:3]),
        [reference, reference, 2 * reference],
        atol=1e-15,
        rtol=0,
    )
    assert int(cp.asnumpy(output).view(np.int32)[28]) == 0


def _cuda_trace_result(
    kernel: Any,
    density: Any,
    hcore: Any,
    coulomb: Any,
    exchange: Any,
    range_exchange: Any,
    *,
    xc_energy: float = 0.0,
    solver_failure: bool = False,
) -> tuple[Any, int]:
    """Keep pointer-null and failure-bit tests on the production CUDA entry."""
    import cupy as cp
    import numpy as np

    spins, matrix = density.shape
    device_density = cp.asarray(density)
    residual = cp.zeros_like(device_density)
    overlap = cp.zeros(matrix)
    device_hcore = cp.asarray(hcore)
    device_coulomb = cp.asarray(coulomb)
    device_exchange = cp.asarray(exchange) if exchange is not None else np.uint64(0)
    device_range = (
        cp.asarray(range_exchange) if range_exchange is not None else np.uint64(0)
    )
    totals = cp.asarray([xc_energy, 0.0, 0.0])
    status = cp.zeros(2, dtype=cp.int32)
    solver = cp.asarray([int(solver_failure)] * spins, dtype=cp.int32)
    output = cp.zeros(16)
    kernel(
        (1,),
        (1,),
        (
            np.uint64(matrix),
            np.uint32(spins),
            device_density,
            device_density,
            residual,
            device_hcore,
            overlap,
            device_coulomb,
            device_exchange,
            np.float64(-0.193),
            device_range,
            np.float64(0.471),
            totals,
            status,
            status,
            np.uint64(0),
            np.uint64(0),
            np.uint64(0),
            solver,
            np.uint64(0),
            output,
        ),
    )
    copied = cp.asnumpy(output)
    return copied[:3], int(copied.view(np.int32)[28])


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("exchange_kind", ("none", "full", "range", "both"))
def test_cuda_nonbinary_products_match_independent_rounded_sum(
    cuda_diagnostic_probe: Any, spins: int, exchange_kind: str
) -> None:
    """Sum rounded FP64 products, including canceling full/range exchange."""
    import numpy as np

    random = np.random.default_rng(61003)
    matrix = 257**2
    density = random.normal(size=(spins, matrix)) * (np.arange(spins)[:, None] + 0.37)
    hcore = random.normal(size=matrix)
    coulomb = random.normal(size=matrix)
    exchange = random.normal(size=(spins, matrix))
    range_exchange = (
        -exchange * (-0.193 / 0.471) * (1 + 3e-7 * (np.arange(spins)[:, None] + 1))
    )
    if exchange_kind not in ("full", "both"):
        exchange = None
    if exchange_kind not in ("range", "both"):
        range_exchange = None
    one_terms = np.multiply(density, hcore).ravel()
    hartree_terms = np.multiply(np.multiply(0.5, density), coulomb).ravel()
    exchange_terms = []
    for coefficient, values in ((-0.193, exchange), (0.471, range_exchange)):
        if values is not None:
            rounded = np.multiply(
                np.multiply(np.multiply(0.5, density), coefficient), values
            )
            exchange_terms.extend(rounded.ravel())
    terms = (one_terms, hartree_terms, exchange_terms)
    expected = [math.fsum(values) for values in terms]
    epsilon = np.finfo(np.float64).eps
    bounds = [
        2 * epsilon * abs(value)
        + 8 * epsilon**2 * len(values) * math.fsum(abs(term) for term in values)
        for value, values in zip(expected, terms, strict=True)
    ]
    actual, failure = _cuda_trace_result(
        cuda_diagnostic_probe, density, hcore, coulomb, exchange, range_exchange
    )
    assert failure == 0
    assert np.all(np.abs(actual - expected) <= bounds), (actual, expected, bounds)


@pytest.mark.parametrize(
    "invalid", ("density", "hcore", "coulomb", "exchange", "range", "xc", "solver")
)
def test_cuda_trace_failure_bits_reject_nonfinite_inputs(
    cuda_diagnostic_probe: Any, invalid: str
) -> None:
    """Compensation cannot hide invalid traces behind finite diagnostics."""
    import numpy as np

    density = np.full((2, 9), 0.37)
    hcore = np.full(9, 1.13)
    coulomb = np.full(9, 0.71)
    exchange = np.full((2, 9), 0.29)
    range_exchange = np.full((2, 9), -0.67)
    arrays = {
        "density": density,
        "hcore": hcore,
        "coulomb": coulomb,
        "exchange": exchange,
        "range": range_exchange,
    }
    if invalid in arrays:
        arrays[invalid].flat[-1] = np.inf if invalid in ("hcore", "range") else np.nan
    actual, failure = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        hcore,
        coulomb,
        exchange,
        range_exchange,
        xc_energy=np.nan if invalid == "xc" else 0.0,
        solver_failure=invalid == "solver",
    )
    assert failure & (4 if invalid == "solver" else 8)
    if invalid == "solver":
        assert np.isfinite(actual).all()
