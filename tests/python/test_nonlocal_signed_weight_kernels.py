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

from tools.generate_nonlocal_pair_native import native_header

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
        _definition(
            source, "template <Vv10Variant Variant, bool Features, bool Geometry>"
        ),
        _definition(source, "template <Vv10Variant Variant, bool Features>"),
        _definition(
            source,
            "template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows,",
        ),
        _definition(source, "__global__ void molecular_domain_kernel("),
        _definition(source, "__global__ void pack_force_seeds_kernel("),
        _definition(source, "__global__ void admit_molecular_pair_domain_kernel("),
    ]
    # Observe executed partner visits without changing the production arithmetic
    # or inferring saved work from zeros in the final output.
    marker = "const auto j = static_cast<std::size_t>(active_indices[slot]);"
    assert pieces[2].count(marker) == 1
    pieces[2] = pieces[2].replace(marker, "++pair_visits;\n" + marker)
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
namespace generativeqc::dft::nlc {
enum class Vv10Variant { vv10, rvv10 };
}
using namespace generativeqc::dft::nlc;
struct Index { std::size_t x{}; } blockIdx, threadIdx, blockDim{1}, gridDim{1};
std::uint64_t pair_visits{};
int last_admission_rejected = -1;
void atomicExch(int* out, int value) { *out = value; }
double __longlong_as_double(unsigned long long value) {
  double result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
"""
    wrapper = r"""
template <Vv10Variant Variant, bool Mask, bool Geometry = true>
int evaluate(double first_weight, double first_density, double* out, double first_gradient_x = 0.1, bool preflight = false, double shift = 0.0, double coefficient = 1.0) {
  pair_visits = 0;
  constexpr std::size_t n = 3;
  const double weights[n] = {first_weight, 0.7, 1.1};
  const double density[n] = {first_density, 0.9, 1.2};
  const double gradient[3*n] = {first_gradient_x, 0.02, 0.03, 0.05, 0.04, 0.02, 0.03, 0.06, 0.07};
  double points[3*n] = {0.0, 0.0, 0.0, 0.8, 0.1, 0.0, -0.3, 0.7, 0.2};
  for (std::size_t i = 0; i < n; ++i) points[3*i] += shift;
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
  int rejected = 0;
  const int* gate = nullptr;
  if (preflight && std::fabs(coefficient) <= 0x1p32) {
    gate = &rejected;
    for (std::size_t i = 0; i < n; ++i) {
      threadIdx.x = i;
      admit_molecular_pair_domain_kernel(n, points, rho, omega, kappa, wrho, wsigma,
                                        krho, weighted, &rejected);
    }
  }
  last_admission_rejected = gate ? rejected : -1;
  const double beta = std::pow(3.0/36.0, 0.75)/32.0;
  for (std::size_t i = 0; i < n; ++i) {
    threadIdx.x = i;
    pair_kernel_ordered<Variant, true, Geometry, Mask>(0, n, coefficient, points, rho, omega, kappa,
        wrho, wsigma, krho, weighted, active, &active_count, beta, energy, vrho, vsigma,
        point_derivative, weight_derivative, &pair_error, gate);
    if constexpr (Variant == Vv10Variant::vv10 && Mask) {
      if (gate)
        pair_kernel_ordered<Variant, true, Geometry, Mask, true>(0, n, coefficient, points,
            rho, omega, kappa, wrho, wsigma, krho, weighted, active, &active_count, beta,
            energy, vrho, vsigma, point_derivative, weight_derivative, &pair_error, gate);
    }
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
  out[13] = static_cast<double>(pair_visits);
  return domain_error || pair_error || collect_error;
}
extern "C" int run(int variant, int mask, double weight, double density, double* out) {
  if (variant)
    return mask ? evaluate<Vv10Variant::rvv10, true>(weight, density, out)
                : evaluate<Vv10Variant::rvv10, false>(weight, density, out);
  return mask ? evaluate<Vv10Variant::vv10, true>(weight, density, out)
              : evaluate<Vv10Variant::vv10, false>(weight, density, out);
}
extern "C" int run_gradient(double weight, double density, double gradient_x, double* out) {
  return evaluate<Vv10Variant::vv10, true>(weight, density, out, gradient_x, true);
}
extern "C" int admission_state() { return last_admission_rejected; }
extern "C" int run_admission(int geometry, int enabled, double weight, double density,
    double gx, double shift, double coefficient, double* out) {
  return geometry
      ? evaluate<Vv10Variant::vv10, true, true>(weight, density, out, gx, enabled, shift, coefficient)
      : evaluate<Vv10Variant::vv10, true, false>(weight, density, out, gx, enabled, shift, coefficient);
}
extern "C" int run_scf(int variant, int mask, double weight, double density, double* out) {
  if (variant)
    return mask ? evaluate<Vv10Variant::rvv10, true, false>(weight, density, out)
                : evaluate<Vv10Variant::rvv10, false, false>(weight, density, out);
  return mask ? evaluate<Vv10Variant::vv10, true, false>(weight, density, out)
              : evaluate<Vv10Variant::vv10, false, false>(weight, density, out);
}
"""
    directory = tmp_path_factory.mktemp("vv10-signed-weights")
    cpp, library = directory / "probe.cpp", directory / "probe.so"
    cpp.write_text(
        prefix
        + native_header()
        + "\nusing PairKernelValues = generated::PairValues;\n"
        + "\n".join(pieces)
        + wrapper
    )
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
    native.run_scf.argtypes = native.run.argtypes
    native.run_scf.restype = ct.c_int
    native.run_gradient.argtypes = [
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.POINTER(ct.c_double),
    ]
    native.run_gradient.restype = ct.c_int
    native.admission_state.restype = ct.c_int
    native.run_admission.argtypes = [
        ct.c_int,
        ct.c_int,
        *([ct.c_double] * 5),
        ct.POINTER(ct.c_double),
    ]
    native.run_admission.restype = ct.c_int
    return native


def _run(
    probe: ct.CDLL,
    variant: int,
    weight: float,
    density: float = 0.4,
    *,
    mask: int = 1,
    scf: bool = False,
    preflight: bool = False,
) -> list[float]:
    result = (ct.c_double * 14)()
    run = probe.run_scf if scf else probe.run
    if preflight:
        assert variant == 0 and mask == 1
        assert (
            probe.run_admission(not scf, 1, weight, density, 0.1, 0.0, 1.0, result) == 0
        )
    else:
        assert run(variant, mask, weight, density, result) == 0
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
    assert actual[7:13] == [0.0] * 6


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
def test_scf_screened_rows_reduce_executed_pair_work(
    signed_weight_probe: ct.CDLL, variant: int
) -> None:
    actual = _run(signed_weight_probe, variant, -0.2, density=1e-15, scf=True)
    unmasked = _run(signed_weight_probe, variant, -0.2, density=1e-15, mask=0, scf=True)
    assert actual[0] == unmasked[0]
    assert actual[1:3] == [0.0, 0.0]
    assert actual[13] == 4  # Two active rows, two compacted partners.
    assert unmasked[13] == 6  # Previously the screened row also traversed both.


@pytest.mark.parametrize("variant", [0, 1], ids=["vv10", "rvv10"])
@pytest.mark.parametrize("weight", [-0.2, -5e-324, -0.0, 0.0, 0.2])
def test_scf_active_signed_rows_keep_energy_potential_and_work(
    signed_weight_probe: ct.CDLL, variant: int, weight: float
) -> None:
    actual = _run(signed_weight_probe, variant, weight, scf=True)
    unmasked = _run(signed_weight_probe, variant, weight, mask=0, scf=True)
    assert actual[:5] == pytest.approx(unmasked[:5], rel=2e-15, abs=1e-16)
    assert actual[5:7] == unmasked[5:7]
    assert actual[7:13] == pytest.approx(unmasked[7:13], rel=2e-15, abs=1e-16)
    assert actual[13] == unmasked[13]
    assert actual[1] != 0.0


def test_pair_kernel_consumes_stable_compacted_partner_domain() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    pair = _definition(
        source,
        "template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows,",
    )
    assert "active_indices[slot]" in pair
    assert "slot < nactive" in pair
    assert "for (std::size_t j = 0; j < npoint; ++j)" not in pair
    assert "count_active_partner_blocks_kernel" in source
    assert "prefix_active_partner_blocks_kernel" in source
    assert "scatter_active_partners_ordered_kernel" in source


@pytest.mark.parametrize("scf", [False, True])
def test_row_parameter_sums_match_density_energy_differences(
    signed_weight_probe: ct.CDLL, scf: bool
) -> None:
    weight, density = -0.2, 0.4
    actual = _run(signed_weight_probe, 0, weight, density, scf=scf, preflight=True)
    for step in (1e-4, 1e-5, 1e-6):
        plus = _run(
            signed_weight_probe, 0, weight, density + step, scf=scf, preflight=True
        )[0]
        minus = _run(
            signed_weight_probe, 0, weight, density - step, scf=scf, preflight=True
        )[0]
        assert (plus - minus) / (2 * step) == pytest.approx(
            weight * actual[1], rel=2e-8, abs=2e-11
        )


@pytest.mark.parametrize("scf", [False, True])
@pytest.mark.parametrize("weight,density", [(2.0**70, 0.4), (0.2, 2.0**100)])
def test_row_parameter_outlier_preserves_the_original_one_pass_domain(
    signed_weight_probe: ct.CDLL, scf: bool, weight: float, density: float
) -> None:
    actual = _run(signed_weight_probe, 0, weight, density, scf=scf, preflight=True)
    original = _run(signed_weight_probe, 0, weight, density, mask=0, scf=scf)
    assert actual[:-1] == original[:-1]
    # Preflight rejects the entire grid before pair execution. No prefix is
    # recomputed, and the original chain/pair program remains bitwise intact.
    assert actual[-1] == original[-1]


def test_row_parameter_sums_match_gradient_energy_differences(
    signed_weight_probe: ct.CDLL,
) -> None:
    weight, density, gx = -0.2, 0.4, 0.1
    actual = _run(signed_weight_probe, 0, weight, density, preflight=True)
    for step in (1e-4, 1e-5, 1e-6):
        energies = []
        for sign in (-1, 1):
            result = (ct.c_double * 14)()
            assert (
                signed_weight_probe.run_gradient(
                    weight, density, gx + sign * step, result
                )
                == 0
            )
            energies.append(result[0])
        assert (energies[1] - energies[0]) / (2 * step) == pytest.approx(
            weight * 2 * gx * actual[2], rel=2e-7, abs=2e-11
        )


@pytest.mark.parametrize("geometry", [0, 1])
@pytest.mark.parametrize(
    "weight,density,shift,coefficient,expected",
    [
        (-0.2, 0.4, 0.0, 1.0, 0),
        (0.0, 0.4, 0.0, 1.0, 0),
        (0.2, 1e-15, 0.0, 1.0, 0),
        (-0.2, 0.4, 2.0**14 - 1, 1.0, 0),
        (-0.2, 0.4, 2.0**14, 1.0, 1),
        (-0.2, 0.4, 2.0**20, 1.0, 1),
        (2.0**70, 0.4, 0.0, 1.0, 1),
        (0.2, 2.0**100, 0.0, 1.0, 1),
        (-0.2, 0.4, 0.0, 2.0**33, -1),
    ],
)
def test_linear_admission_preserves_outputs_order_and_executed_pair_visits(
    signed_weight_probe: ct.CDLL,
    geometry: int,
    weight: float,
    density: float,
    shift: float,
    coefficient: float,
    expected: int,
) -> None:
    """The predicate selects exactly one pair traversal and never bypasses a bound."""
    outputs = []
    for enabled in (0, 1):
        out = (ct.c_double * 14)()
        assert (
            signed_weight_probe.run_admission(
                geometry, enabled, weight, density, 0.1, shift, coefficient, out
            )
            == 0
        )
        outputs.append(tuple(out))
    assert signed_weight_probe.admission_state() == expected
    if expected == 0:
        assert outputs[1] == pytest.approx(outputs[0], rel=2e-14, abs=2e-14)
    else:
        assert outputs[1] == outputs[0]
    assert outputs[1][5:7] == outputs[0][5:7]
    assert outputs[1][-1] == outputs[0][-1]
