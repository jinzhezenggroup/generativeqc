"""Stress the actual KS diagnostic block against independent trace/metric sums.

The host harness emulates CUDA lanes/barriers; device tests execute the same
block body. Molecular SCF and force gates remain separate requirements.
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
        "#include <cmath>\n#include <cstddef>\n#include <cstdint>\n#include <algorithm>\n"
        "#include <cstdlib>\n#include <iomanip>\n#include <iostream>\n"
        "#include <vector>\n#include <thread>\n#include <barrier>\n"
        "#define __global__\n#define __device__\n#define __shared__ static\n"
        "struct Index { unsigned x; };\n"
        "thread_local Index threadIdx{};\nconst Index blockDim{256};\n"
        "std::barrier team_barrier(256);\n"
        "void __syncthreads() { team_barrier.arrive_and_wait(); }\n"
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
  double totals[3] = {};
  if (argc > 3) {
    std::fill(hcore.begin(), hcore.end(), 0.0);
    std::fill(coulomb.begin(), coulomb.end(), 0.0);
    std::fill(exchange.begin(), exchange.end(), 0.0);
    std::fill(range_exchange.begin(), range_exchange.end(), 0.0);
    hcore[0] = -4096.0;
    hcore[1] = 3e-13;
    coulomb[0] = 8192.0;
    totals[0] = 0.25;
  }
  const int status[2] = {};
  Scalars output;
  std::vector<std::thread> lanes;
  for (unsigned lane = 0; lane < blockDim.x; ++lane) lanes.emplace_back([&, lane] {
    threadIdx.x = lane;
    diagnostic_kernel(matrix, spins, density.data(), density.data(), residual.data(),
                    hcore.data(), overlap.data(), coulomb.data(), exchange.data(), -0.25,
                    range_exchange.data(), -0.5, totals, status, status, nullptr,
                      nullptr, nullptr, status, nullptr, &output);
  });
  for (auto& lane : lanes) lane.join();
  if (argc > 3) {
    std::cout << std::setprecision(17) << output.electronic_energy << ' '
              << output.electronic_energy_correction << '\n';
    return 0;
  }
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
            "-std=c++20",
            "-O2",
            "-pthread",
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


def test_electronic_energy_keeps_unrounded_component_words(
    diagnostic_probe: Path,
) -> None:
    """A large individual component must not quantize away the physical delta."""
    completed = subprocess.run(
        [str(diagnostic_probe), "9", "1", "expanded"],
        check=True,
        capture_output=True,
        text=True,
    )
    high, correction = map(float, completed.stdout.split())
    assert math.fsum((high, correction, -0.25)) == pytest.approx(
        3e-13, rel=1e-12, abs=0
    )


@pytest.mark.parametrize("matrix", (3, 255, 258, 513, 771))
@pytest.mark.parametrize("spins", (1, 2))
def test_host_diagnostic_handles_incomplete_lane_tiles(
    diagnostic_probe: Path, matrix: int, spins: int
) -> None:
    """Exercise idle lanes and partial final tiles in the production body."""
    completed = subprocess.run(
        [str(diagnostic_probe), str(matrix), str(spins)],
        check=True,
        capture_output=True,
        text=True,
    )
    one, hartree, exchange, failure = map(float, completed.stdout.split())
    assert failure == 0
    assert (one, hartree, exchange) == pytest.approx((0, 0, 0), abs=1e-15)


def test_parallel_diagnostic_is_the_bounded_production_default() -> None:
    """The public runtime must select the qualified owner without an opt-in."""
    source = (ROOT / "src/dft/cuda_ks_kernels.cu").read_text()
    compact = "".join(source.split())
    assert "constexprunsignedkDiagnosticThreads=256;" in compact
    assert "__shared__DiagnosticPartialpartials[kDiagnosticThreads];" in compact
    assert "diagnostic_kernel<<<1,kDiagnosticThreads,0,stream>>>" in compact
    assert "diagnostic_kernel<<<1,1," not in compact
    assert (
        "atomicAdd"
        not in source[
            source.index("__global__ void diagnostic_kernel") : source.index(
                "__global__ void advance_kernel"
            )
        ]
    )


def test_compensated_diagnostic_merge_keeps_both_lane_words() -> None:
    """Dropping lane corrections would silently undo signed-trace accuracy."""
    source = (ROOT / "src/dft/cuda_ks_kernels.cu").read_text()
    compact = "".join(source.split())
    for word in ("energy", "correction"):
        assert (
            f"accumulate_diagnostic_trace(part.{word}[term],"
            "total.energy[term],total.correction[term]);"
        ) in compact
    assert "for(unsignedindex=0;index<blockDim.x;++index)" in compact


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
        + "static_assert(sizeof(Scalars) <= 136);\n"
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
    output = cp.zeros(18)
    cuda_diagnostic_probe(
        (1,),
        (256,),
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
    proposal: Any = None,
    residual: Any = None,
    overlap: Any = None,
    enabled: int | None = None,
    full_output: bool = False,
    errors: tuple[int, ...] = (0, 0, 0, 0, 0),
) -> tuple[Any, int]:
    """Keep pointer-null and failure-bit tests on the production CUDA entry."""
    import cupy as cp
    import numpy as np

    spins, matrix = density.shape
    device_density = cp.asarray(density)
    device_proposal = device_density if proposal is None else cp.asarray(proposal)
    device_residual = (
        cp.zeros_like(device_density) if residual is None else cp.asarray(residual)
    )
    device_overlap = cp.zeros(matrix) if overlap is None else cp.asarray(overlap)
    device_hcore = cp.asarray(hcore)
    device_coulomb = cp.asarray(coulomb)
    device_exchange = cp.asarray(exchange) if exchange is not None else np.uint64(0)
    device_range = (
        cp.asarray(range_exchange) if range_exchange is not None else np.uint64(0)
    )
    totals = cp.asarray([xc_energy, 0.0, 0.0])
    status = [cp.asarray([value], dtype=cp.int32) for value in errors]
    solver = cp.asarray([int(solver_failure)] * spins, dtype=cp.int32)
    mask = np.uint64(0) if enabled is None else cp.asarray([enabled], dtype=cp.uint8)
    output = cp.full(18, -99.0)
    kernel(
        (1,),
        (256,),
        (
            np.uint64(matrix),
            np.uint32(spins),
            device_density,
            device_proposal,
            device_residual,
            device_hcore,
            device_overlap,
            device_coulomb,
            device_exchange,
            np.float64(-0.193),
            device_range,
            np.float64(0.471),
            totals,
            *status,
            solver,
            mask,
            output,
        ),
    )
    copied = cp.asnumpy(output)
    return copied if full_output else copied[:3], int(copied.view(np.int32)[28])


def test_cuda_electronic_energy_retains_component_cancellation(
    cuda_diagnostic_probe: Any,
) -> None:
    """Check the actual NVCC reduction retains low words until the energy gate."""
    import numpy as np

    density = np.ones((1, 9))
    hcore, coulomb = np.zeros(9), np.zeros(9)
    hcore[0], hcore[1], coulomb[0] = -4096.0, 3e-13, 8192.0
    actual, failure = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        hcore,
        coulomb,
        None,
        None,
        xc_energy=0.25,
        full_output=True,
    )
    assert failure == 0
    assert math.fsum((actual[15], actual[16], -0.25)) == pytest.approx(
        3e-13, rel=1e-12, abs=0
    )


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


@pytest.mark.parametrize("matrix", (1, 255, 256, 257, 384**2, 768**2))
@pytest.mark.parametrize("spins", (1, 2))
def test_cuda_parallel_diagnostics_preserve_all_physical_metrics(
    cuda_diagnostic_probe: Any, matrix: int, spins: int
) -> None:
    """Independent sums cover inactive lanes, tails, spin maxima and RMS gates."""
    import numpy as np

    rng = np.random.default_rng(1895)
    density = rng.normal(size=(spins, matrix))
    proposal = density + rng.normal(size=density.shape) * 3e-7
    residual = rng.normal(size=density.shape) * 2e-10
    overlap = rng.normal(size=matrix)
    zero = np.zeros(matrix)
    actual, failure = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        zero,
        zero,
        None,
        None,
        proposal=proposal,
        residual=residual,
        overlap=overlap,
        full_output=True,
    )
    error2 = [math.fsum(row * row) / matrix for row in residual]
    change2 = [math.fsum(row * row) / matrix for row in proposal - density]
    electrons = [math.fsum(row * overlap) for row in density]
    if spins == 1:
        electrons = [electrons[0] / 2] * 2
    expected = [
        math.sqrt(max(error2)),
        math.sqrt(max(change2)),
        math.sqrt(math.fsum(error2) / spins),
        math.sqrt(math.fsum(change2) / spins),
        np.max(np.abs(residual)),
        *electrons,
    ]
    assert failure == 0
    np.testing.assert_allclose(actual[4:9], expected[:5], rtol=3e-13, atol=0)
    np.testing.assert_allclose(actual[9:11], expected[5:], rtol=3e-13, atol=1e-13)
    assert actual[8] == expected[4]
    # The private result includes both energy words; guard its 136-byte extent.
    assert actual[17] == -99.0


@pytest.mark.parametrize("invalid", ("residual", "proposal", "overlap"))
def test_cuda_parallel_metric_nonfinite_is_not_hidden_by_maximum(
    cuda_diagnostic_probe: Any, invalid: str
) -> None:
    """NaNs on a nonzero lane must survive even though fmax ignores a NaN."""
    import numpy as np

    density = np.ones((2, 257))
    metrics = {
        "residual": np.zeros_like(density),
        "proposal": density.copy(),
        "overlap": np.ones(257),
    }
    metrics[invalid].flat[-2] = np.nan
    _, failure = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        np.ones(257),
        np.ones(257),
        None,
        None,
        **metrics,
    )
    assert failure & 8


def test_cuda_parallel_diagnostics_mask_and_failure_union(
    cuda_diagnostic_probe: Any,
) -> None:
    """Inactive work leaves output untouched; every upstream failure is retained."""
    import numpy as np

    density = np.ones((2, 257))
    values = np.ones(257)
    masked, _ = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        values,
        values,
        None,
        None,
        enabled=0,
        full_output=True,
    )
    assert np.all(masked == -99.0)
    _, failure = _cuda_trace_result(
        cuda_diagnostic_probe,
        density,
        values,
        values,
        None,
        None,
        errors=(1, 1, 1, 1, 1),
        solver_failure=True,
        enabled=1,
    )
    assert failure == 1 | 2 | 4 | 16 | 32 | 64
