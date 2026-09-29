"""Host-execute production VV10 kernels to verify signed-weight semantics.

Lane indices and atomics are scalar stand-ins; this is not GPU scheduling
qualification. Both pair variants run their actual source arithmetic.
"""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def signed_weight_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    pieces = [
        _definition(source, "struct PairKernelValues") + ";",
        _definition(
            source, "template <Vv10Variant Variant, bool Features, bool Geometry>"
        ),
        _definition(source, "template <Vv10Variant Variant, bool Features>"),
        _definition(
            source,
            "template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows>",
        ),
        _definition(source, "__global__ void molecular_domain_kernel("),
        _definition(source, "__global__ void pack_force_seeds_kernel("),
    ]
    prefix = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
using std::isfinite;
using std::signbit;
#define __device__
#define __global__
constexpr double kPi = 3.141592653589793238462643383279502884;
enum class Vv10Variant { vv10, rvv10 };
struct Index { std::size_t x{}; } blockIdx, threadIdx, blockDim{1}, gridDim{1};
void atomicExch(int* out, int value) { *out = value; }
double __longlong_as_double(unsigned long long value) {
  double result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
"""
    wrapper = r"""
template <Vv10Variant Variant, bool Mask>
int evaluate(double first_weight, double first_density, double* out) {
  constexpr std::size_t n = 3;
  const double weights[n] = {first_weight, 0.7, 1.1};
  const double density[n] = {first_density, 0.9, 1.2};
  const double gradient[3*n] = {0.1, 0.02, 0.03, 0.05, 0.04, 0.02, 0.03, 0.06, 0.07};
  const double points[3*n] = {0.0, 0.0, 0.0, 0.8, 0.1, 0.0, -0.3, 0.7, 0.2};
  double ew[n]{}, rho[n]{}, grad[3*n]{}, omega[n]{}, kappa[n]{};
  double wrho[n]{}, wsigma[n]{}, krho[n]{}, weighted[n]{}, energy[n]{};
  double vrho[n]{}, vsigma[n]{}, point_derivative[3*n]{}, weight_derivative[n]{};
  int domain_error = 0, pair_error = 0, collect_error = 0;
  threadIdx.x = 0;
  molecular_domain_kernel(n, 1e-12, weights, density, gradient, ew, rho, grad, &domain_error);
  for (std::size_t i = 0; i < n; ++i) {
    threadIdx.x = i;
    local_scales_kernel<Variant, true>(n, 6.0, 0.01, ew, rho, grad, omega, kappa,
                                      wrho, wsigma, krho, weighted, &pair_error);
  }
  std::uint64_t active[n]{}, active_count = 0;
  for (std::size_t j = 0; j < n; ++j)
    if (weighted[j] != 0.0) active[active_count++] = j;
  const double beta = std::pow(3.0/36.0, 0.75)/32.0;
  for (std::size_t i = 0; i < n; ++i) {
    threadIdx.x = i;
    pair_kernel_ordered<Variant, true, true, Mask>(0, n, 1.0, points, rho, omega, kappa,
        wrho, wsigma, krho, weighted, active, &active_count, beta, energy, vrho, vsigma,
        point_derivative, weight_derivative, &pair_error);
  }
  double seeds[6*n]{};
  for (std::size_t i = 0; i < n; ++i) {
    seeds[i] = vrho[i];
    seeds[n+i] = vsigma[i];
    seeds[5*n+i] = weight_derivative[i];
    threadIdx.x = i;
    pack_force_seeds_kernel(n, ew, point_derivative, seeds,
                           &collect_error, &domain_error, &pair_error);
  }
  out[0] = energy[0] + energy[1] + energy[2];
  out[1] = vrho[0]; out[2] = vsigma[0];
  out[3] = point_derivative[0]; out[4] = weight_derivative[0];
  out[5] = std::signbit(ew[0]); out[6] = std::signbit(weighted[0]);
  for (std::size_t row = 0; row < 6; ++row) out[7+row] = seeds[row*n];
  return domain_error || pair_error || collect_error;
}
extern "C" int run(int variant, int mask, double weight, double density, double* out) {
  if (variant)
    return mask ? evaluate<Vv10Variant::rvv10, true>(weight, density, out)
                : evaluate<Vv10Variant::rvv10, false>(weight, density, out);
  return mask ? evaluate<Vv10Variant::vv10, true>(weight, density, out)
              : evaluate<Vv10Variant::vv10, false>(weight, density, out);
}
"""
    directory = tmp_path_factory.mktemp("vv10-signed-weights")
    cpp, library = directory / "probe.cpp", directory / "probe.so"
    cpp.write_text(prefix + "\n".join(pieces) + wrapper)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(cpp),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    native = ct.CDLL(str(library))
    native.run.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.POINTER(ct.c_double),
    ]
    native.run.restype = ct.c_int
    return native


def _run(
    probe: ct.CDLL,
    variant: int,
    weight: float,
    density: float = 0.4,
    *,
    mask: int = 1,
) -> list[float]:
    result = (ct.c_double * 13)()
    assert probe.run(variant, mask, weight, density, result) == 0
    return list(result)


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
@pytest.mark.parametrize("weight", [-0.2, -1e-200, 0.0, -0.0, 0.2])
def test_active_signed_weights_match_unmasked_pair_arithmetic(
    signed_weight_probe: ct.CDLL, variant: int, weight: float
) -> None:
    actual = _run(signed_weight_probe, variant, weight)
    unmasked = _run(signed_weight_probe, variant, weight, mask=0)
    assert actual[:5] == pytest.approx(unmasked[:5], rel=1e-13, abs=1e-15)
    assert actual[7:9] == pytest.approx(actual[1:3], rel=1e-13, abs=1e-15)
    assert actual[12] == actual[4]
    assert actual[4] != 0.0


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
def test_underflow_does_not_forge_inactive_negative_zero(
    signed_weight_probe: ct.CDLL, variant: int
) -> None:
    actual = _run(signed_weight_probe, variant, -5e-324, density=0.1)
    assert actual[5] == 1.0  # The effective integration weight remains negative.
    assert actual[6] == 0.0  # Only a real inactive marker may survive as -0.
    assert actual[4] != 0.0
    assert actual[12] == actual[4]


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
@pytest.mark.parametrize("weight", [-0.2, 0.0, 0.2])
def test_weight_derivative_matches_three_step_kernel_energy_differences(
    signed_weight_probe: ct.CDLL, variant: int, weight: float
) -> None:
    expected = _run(signed_weight_probe, variant, weight)[4]
    for step in (1e-4, 1e-5, 1e-6):
        plus = _run(signed_weight_probe, variant, weight + step)[0]
        minus = _run(signed_weight_probe, variant, weight - step)[0]
        assert (plus - minus) / (2 * step) == pytest.approx(
            expected, rel=2e-8, abs=2e-11
        )


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
def test_density_screened_row_keeps_negative_zero_and_zero_seeds(
    signed_weight_probe: ct.CDLL, variant: int
) -> None:
    actual = _run(signed_weight_probe, variant, -0.2, density=1e-15)
    assert actual[5:7] == [1.0, 1.0]
    assert actual[1:5] == [0.0] * 4
    assert actual[7:] == [0.0] * 6


def test_pair_kernel_consumes_stable_compacted_partner_domain() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    pair = _definition(
        source,
        "template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows>",
    )
    assert "active_indices[slot]" in pair
    assert "slot < nactive" in pair
    assert "for (std::size_t j = 0; j < npoint; ++j)" not in pair
    assert "count_active_partner_blocks_kernel" in source
    assert "prefix_active_partner_blocks_kernel" in source
    assert "scatter_active_partners_ordered_kernel" in source
