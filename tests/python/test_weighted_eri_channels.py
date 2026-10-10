"""Shared recurrence numerics, zero operators, publication and bounded subsets."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_force_density_channels,
    emit_direct_force_density_coefficient,
)
from generativeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from generativeqc_compiler.integral.weighted_eri_channels import (
    emit_weighted_eri_channel_function,
)
from generativeqc_compiler.integral.weighted_eri_cuda import emit_weighted_eri_header
from test_direct_force_scaled_numeric import _orbit_reference
from test_weighted_eri import CENTERS, EXPONENTS, evaluate

from tools.generativeqc_validation.weighted_eri import primitive_variables

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def channel_library(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    """Compile the actual source lowering; no CUDA execution is claimed."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host channel qualification requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    kernel = build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 0, 0)))
    subset = build_weighted_eri_kernel(kernel.integral, (0, 2))
    source = emit_weighted_eri_header(((kernel, "single"),), backend="cpu")
    source += "\nnamespace generativeqc::scf::generated_weighted_eri {\n"
    source += "struct IndependentGradient { double center[3][3]; };\n"
    source += emit_weighted_eri_channel_function(kernel, "full", backend="cpu")
    source += emit_weighted_eri_channel_function(subset, "subset", backend="cpu")
    source += "}\n"
    source += r"""
#include "integrals/eri_geometry.hpp"
#include <limits>
namespace weighted = generativeqc::scf::generated_weighted_eri;
extern "C" int channels(int subset, int fault, const double* exponents,
    const double* centers, const double* weights, const bool* active, double* output) {
  weighted::Geometry geometry{};
  if (!generativeqc::integrals::make_eri_geometry(exponents, centers, 2,
      generativeqc::integrals::CoulombRange::Full, 0.0, geometry)) return 2;
  if (fault == 1) geometry.prefactor = std::numeric_limits<double>::infinity();
  const auto& input = *reinterpret_cast<const double (*)[3][3]>(weights);
  const auto& mask = *reinterpret_cast<const bool (*)[3]>(active);
  auto& result = *reinterpret_cast<weighted::IndependentGradient (*)[3]>(output);
  const bool success = subset ? weighted::subset_channels(geometry, input, mask, result)
                             : weighted::full_channels(geometry, input, mask, result);
  return success ? 0 : 1;
}
"""
    source += "\n#define __device__\n#define __forceinline__ inline\n"
    source += '#include "scf/cuda/direct_eri_symmetry.cuh"\n'
    source += '#include "scf/cuda/matrix_index.cuh"\n'
    source += "namespace generativeqc::scf::cuda_execution {\n"
    source += emit_direct_force_density_coefficient()
    source += emit_direct_force_density_channels()
    source += "}\n"
    source += r"""
extern "C" void weight_channels(int unrestricted, std::size_t n,
    std::size_t first, std::size_t second, std::size_t third, std::size_t fourth,
    const double* density, const double* coulomb, const double* exchange, double* output) {
  const auto& cj = *reinterpret_cast<const double (*)[3]>(coulomb);
  const auto& ck = *reinterpret_cast<const double (*)[3]>(exchange);
  auto& result = *reinterpret_cast<double (*)[3]>(output);
  using namespace generativeqc::scf::cuda_execution;
  if (unrestricted)
    direct_force_density_channels_scaled<true>(n, 3, 3+n*n, density,
        first, second, third, fourth, cj, ck, result);
  else
    direct_force_density_channels_scaled<false>(n, 3, 3+n*n, density,
        first, second, third, fourth, cj, ck, result);
}
"""
    directory = tmp_path_factory.mktemp("weighted-eri-channels")
    (directory / "cuda_runtime.h").write_text("#pragma once\n")
    path, library = directory / "channels.cpp", directory / "channels.so"
    path.write_text(source)
    result = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            f"-I{directory}",
            f"-I{ROOT / 'src'}",
            str(path),
            "-o",
            str(library),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    binding = ct.CDLL(str(library))
    binding.channels.argtypes = [ct.c_int, ct.c_int] + [ct.c_void_p] * 5
    binding.channels.restype = ct.c_int
    binding.weight_channels.argtypes = (
        [ct.c_int] + [ct.c_size_t] * 5 + [ct.c_void_p] * 4
    )
    binding.weight_channels.restype = None
    return binding


def run_channels(
    library: ct.CDLL,
    weights: np.ndarray,
    active: tuple[bool, bool, bool],
    *,
    subset: bool = False,
    fault: int = 0,
) -> tuple[int, np.ndarray]:
    """Keep borrowed test arrays alive across the synchronous generated call."""
    exponents = np.array(EXPONENTS, dtype=np.float64)
    centers = np.ascontiguousarray(CENTERS)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    mask = np.array(active, dtype=np.bool_)
    output = np.full((3, 3, 3), 19.25)
    status = library.channels(
        int(subset),
        fault,
        exponents.ctypes.data,
        centers.ctypes.data,
        weights.ctypes.data,
        mask.ctypes.data,
        output.ctypes.data,
    )
    return status, output


@pytest.mark.parametrize("subset", [False, True])
def test_three_independent_weight_channels_match_weight_first_dag(
    channel_library: ct.CDLL, subset: bool
) -> None:
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    kernel = build_weighted_eri_kernel(integral, (0, 2) if subset else None)
    weights = np.array([[0.3, -0.7, 0.9], [-0.2, 1.5, -0.6], [0.0, 0.0, 0.0]])
    status, output = run_channels(
        channel_library, weights, (True, True, True), subset=subset
    )
    assert status == 0
    variables = primitive_variables(EXPONENTS, CENTERS, integral.maximum_coulomb_order)
    for channel in range(3):
        expected = evaluate(kernel, variables, weights[channel])[1][:3]
        np.testing.assert_allclose(output[channel], expected, atol=3e-12, rtol=3e-12)


def test_inactive_channels_do_not_read_nonfinite_weights(
    channel_library: ct.CDLL,
) -> None:
    weights = np.array([[0.2, -0.4, 0.7], [np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0]])
    status, output = run_channels(channel_library, weights, (True, False, True))
    assert status == 0
    np.testing.assert_array_equal(output[1:], 0.0)


@pytest.mark.parametrize("fault", [0, 1])
def test_failed_active_channel_preserves_all_publication_planes(
    channel_library: ct.CDLL, fault: int
) -> None:
    weights = np.array([[0.2, -0.4, 0.7], [np.nan, 0.8, -0.3], [0.0, 0.0, 0.0]])
    status, output = run_channels(
        channel_library, weights, (True, True, False), fault=fault
    )
    assert status == 1
    np.testing.assert_array_equal(output, 19.25)


def test_exact_zero_components_do_not_multiply_nonfinite_integrals(
    channel_library: ct.CDLL,
) -> None:
    status, output = run_channels(
        channel_library, np.zeros((3, 3)), (True, True, True), fault=1
    )
    assert status == 0
    np.testing.assert_array_equal(output, 0.0)


def test_channel_source_is_deterministic_and_uses_one_cse_state() -> None:
    kernel = build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 1, 0)))
    first = emit_weighted_eri_channel_function(kernel, "shared")
    assert first == emit_weighted_eri_channel_function(kernel, "shared")
    assert first.count("std::isfinite") == 1
    assert "component_weight_" not in first
    assert "ChannelCount" in first
    assert "output[channel] = candidate[channel]" in first
    with pytest.raises(ValueError, match="ASCII"):
        emit_weighted_eri_channel_function(kernel, "not-a-name")
    with pytest.raises(ValueError, match="cpu or cuda"):
        emit_weighted_eri_channel_function(kernel, "shared", backend="other")


@pytest.mark.parametrize("unrestricted", [False, True])
def test_generated_density_channels_match_independent_unique_orbits(
    channel_library: ct.CDLL, unrestricted: bool
) -> None:
    order = 3
    matrices = tuple(
        [
            (min(row, column) + 2 * max(row, column) + 1) * scale
            for column in range(order)
            for row in range(order)
        ]
        for scale in (0.13, 0.07, -0.02)
    )
    density = np.array([-7.0, -8.0, -9.0, *np.array(matrices).ravel()])
    coulomb = np.array([1.7, 0.0, 0.4])
    exchange = np.array([0.0, -0.23, -0.9])
    output = np.zeros(3)
    for indices in product(range(order), repeat=4):
        first, second, third, fourth = indices
        if first < second or third < fourth or (first, second) < (third, fourth):
            continue
        channel_library.weight_channels(
            unrestricted,
            order,
            *indices,
            density.ctypes.data,
            coulomb.ctypes.data,
            exchange.ctypes.data,
            output.ctypes.data,
        )
        for channel in range(3):
            expected = _orbit_reference(
                matrices,
                order,
                indices,
                unrestricted,
                coulomb[channel],
                exchange[channel],
            )
            assert output[channel] == pytest.approx(expected, rel=2e-13, abs=2e-14)


def test_low_order_channels_match_independent_displaced_hermite_values(
    tmp_path: Path,
) -> None:
    """Reuse the independent order-four oracle scaffold for every low-order class."""
    from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
        emit_direct_cartesian_contraction_headers,
    )
    from generativeqc_compiler.integral.direct_pair_support_cuda import (
        emit_direct_pair_support_headers,
    )
    from generativeqc_compiler.integral.direct_recurrence_cuda import (
        emit_direct_recurrence_headers,
    )
    from test_direct_low_order_sources import prepare_source_probe
    from test_direct_order4_sources import PREFIX

    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("independent host qualification requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = prepare_source_probe(tmp_path)
    headers = {
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        **emit_direct_cartesian_contraction_headers(),
    }
    for name, contents in headers.items():
        (tmp_path / name).write_text(contents)
    prefix = (
        PREFIX.replace(
            "direct_force_order4_sources.cuh", "direct_force_low_order_sources.cuh"
        )
        .replace('#include "scf/cuda/direct_force_quartet.cuh"\n', "")
        .replace("Order4SourceRoots", "LowOrderSourceRoots")
        .replace("boys_values<5>", "boys_values<A+B+C+D+1>")
    )
    source.write_text(
        prefix
        + r"""
int main() {
  independent<kSsssShellClass,0,0,0,0>();
  independent<kPsssShellClass,1,0,0,0>();
  independent<kPspsShellClass,1,0,1,0>();
  independent<kPpssShellClass,1,1,0,0>();
  independent<kDsssShellClass,2,0,0,0>();
  independent<kPppsShellClass,1,1,1,0>();
  independent<kDspsShellClass,2,0,1,0>();
  independent<kDpssShellClass,2,1,0,0>();
  independent<kFsssShellClass,3,0,0,0>();
  std::printf("independent low-order shared channels PASS; coordinates=%llu maximum_error=%.17g\n",
              checks, max_independent_error);
}
"""
    )
    executable = tmp_path / "independent"
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(source),
            "-o",
            str(executable),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "coordinates=15840 " in result.stdout
    print(result.stdout)
