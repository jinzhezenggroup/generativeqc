"""Execute the fused shell RSH force consumer's coefficient contract on the host.

The complete production consumer, density algebra, indexing and force scatter
are compiled unchanged. Only integral derivatives and CUDA annotations are
substituted; independent ERI/device qualification remains in the CUDA suites.
"""

from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fused_shell_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ct.CDLL:
    directory = tmp_path_factory.mktemp("fused-shell-force-coefficients")
    (directory / "cuda_runtime.h").write_text(
        "#pragma once\n#include <algorithm>\n#include <cmath>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "using std::min; using std::max;\n"
        "inline void atomicAdd(double* p, double value) { *p += value; }\n"
    )
    (directory / "generated_direct_fock_accumulation.cuh").write_text(
        emit_direct_fock_accumulation_header()
    )
    source = (ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text()
    consumer = cpp_function_definition(
        source,
        "contract_two_electron_force_quartet_subtile_rsh_scaled",
        include_template=True,
    )
    cpp = directory / "probe.cpp"
    cpp.write_text(PREFIX + consumer + SUFFIX)
    library = directory / "probe.so"
    native_cxx.build_shared(
        [cpp],
        library,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-I" + str(directory),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        compile_timeout=60,
    )
    probe = ct.CDLL(str(library))
    probe.forces.argtypes = [
        ct.c_bool,
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.POINTER(ct.c_double),
    ]
    probe.forces.restype = None
    return probe


def check_forces(
    probe: ct.CDLL,
    unrestricted: bool,
    alpha: list[float],
    beta: list[float],
    coefficients: tuple[float, float, float],
) -> None:
    """Independently contract the two distinct ordered (11|00) permutations."""
    density = (ct.c_double * 8)(*alpha, *beta)
    output = (ct.c_double * 18)()
    probe.forces(unrestricted, density, *coefficients, output)
    cj, short_ck, long_ck = coefficients
    # Do not form an inactive channel's potentially overflowing intermediates.
    coulomb = 0.0
    if cj:
        total_00 = alpha[0] + (beta[0] if unrestricted else 0.0)
        total_11 = alpha[3] + (beta[3] if unrestricted else 0.0)
        coulomb = cj * total_00 * total_11
    exchange = 0.0
    if short_ck or long_ck:
        exchange = alpha[1] * alpha[2]
        if unrestricted:
            exchange += beta[1] * beta[2]
    weights = (coulomb, short_ck * exchange, long_ck * exchange)
    full = (2.0, 3.0, 5.0)
    long = (0.5, 0.75, 1.25)
    gradients = (full, tuple(a - b for a, b in zip(full, long, strict=True)), long)
    expected = []
    for weight, gradient in zip(weights, gradients, strict=True):
        response = [weight * value for value in gradient]
        expected.extend((*response, *(-value for value in response)))
    assert all(math.isfinite(value) for value in output)
    assert list(output) == pytest.approx(expected, rel=2e-14, abs=2e-14)
    for channel, coefficient in enumerate(coefficients):
        if coefficient == 0.0:
            assert list(output)[channel * 6 : (channel + 1) * 6] == [0.0] * 6


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize(
    "coefficients",
    (
        (1.7, -0.15, -0.63),
        (1.0, 0.0, 0.0),
        (0.0, -0.37, 0.0),
        (0.0, 0.0, -0.37),
        (0.0, -0.15, -0.63),
        (0.0, 0.0, 0.0),
    ),
)
def test_fused_shell_force_coefficients_match_independent_sources(
    fused_shell_probe: ct.CDLL,
    unrestricted: bool,
    coefficients: tuple[float, float, float],
) -> None:
    check_forces(
        fused_shell_probe,
        unrestricted,
        [0.2, 0.1, 0.1, 0.3],
        [0.13, 0.05, 0.05, 0.17],
        coefficients,
    )


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("disabled", ("exchange", "coulomb", "both"))
def test_fused_shell_disabled_sources_do_not_overflow(
    fused_shell_probe: ct.CDLL, unrestricted: bool, disabled: str
) -> None:
    if disabled == "exchange":
        alpha = [1.0, 1e200, 1e200, 1.0]
        beta = [1.0, -1e200, -1e200, 1.0]
        coefficients = (1.0, 0.0, 0.0)
    elif disabled == "coulomb":
        alpha = beta = [1e308, 1.0, 1.0, 1e308]
        coefficients = (0.0, -0.15, -0.5)
    else:
        alpha = beta = [1e308] * 4
        coefficients = (0.0, -0.0, 0.0)
    check_forces(fused_shell_probe, unrestricted, alpha, beta, coefficients)


PREFIX = r"""
#include <array>
#include <cmath>
#include "generated_direct_fock_accumulation.cuh"
#include "integrals/range_moments.hpp"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_gradient_types.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/scalar_math.cuh"
namespace generativeqc::scf::cuda_execution {

// Integral responses are transport inputs here. The real consumer must apply
// signed J/SR/LR weights and recover the second atom's translational response.
template<unsigned Order, class Scalar>
Scalar dispatch_contracted_eri_cartesian_source_shell_class(
    unsigned, const DeviceBatch&, std::int32_t, std::int32_t, std::int32_t,
    std::int32_t, std::int32_t, std::int64_t,
    integrals::CoulombRange range = integrals::CoulombRange::Full,
    double = 0.0) {
  return range == integrals::CoulombRange::Long
      ? Scalar{0.0, 0.5, 0.75, 1.25} : Scalar{0.0, 2.0, 3.0, 5.0};
}
// The order-zero instantiation does not call the higher-order integral owners.
CartesianQuartetGradient contracted_eri_cartesian_source_order2_generated_gradient(
    const DeviceBatch&, int, int, int, int, int);
CartesianQuartetGradient contracted_eri_cartesian_source_order4_gradient(
    const DeviceBatch&, int, int, int, int, int);
CartesianQuartetGradient contracted_eri_cartesian_source_order5_gradient(
    const DeviceBatch&, int, int, int, int, int);
CartesianQuartetGradient contracted_eri_cartesian_source_order6_gradient(
    const DeviceBatch&, int, int, int, int, int);
"""

SUFFIX = r"""
}  // namespace generativeqc::scf::cuda_execution
extern "C" void forces(bool unrestricted, const double* density, double cj,
                       double short_ck, double long_ck, double* output) {
  using namespace generativeqc::scf::cuda_execution;
  const std::int32_t systems[]{0,0}, shells[]{1,0}, atoms[]{0,1};
  const std::int64_t offsets[]{0,1,2};
  const std::uint8_t angular[]{0,0}, active[]{1};
  const double bounds[]{1,1,1,1};
  DeviceBatch batch{};
  batch.direct_nbf = 2;
  batch.total_atoms = 2;
  batch.shell_pair_systems = systems;
  batch.shell_pair_first = batch.shell_pair_second = shells;
  batch.shell_direct_ao_offsets = offsets;
  batch.shell_atoms = atoms;
  batch.shell_angular = angular;
  const ActiveShellQuartetTile task{0,1,0};
  const std::uint32_t count = 1;
  if (unrestricted)
    contract_two_electron_force_quartet_subtile_rsh_scaled<true,0>(
        batch,&count,&task,0.0,bounds,density,active,output,cj,short_ck,long_ck,0.37,0,0);
  else
    contract_two_electron_force_quartet_subtile_rsh_scaled<false,0>(
        batch,&count,&task,0.0,bounds,density,active,output,cj,short_ck,long_ck,0.37,0,0);
}
"""
