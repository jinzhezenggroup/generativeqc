"""Host lowering of the shared bounded full/SR/LR Cartesian derivative owner."""

from .direct_cartesian_contraction_cuda import emit_direct_cartesian_cpu_header
from .range_separation import CoulombKernel, CoulombKernelFamily


def _emit_primitive_program(radial: CoulombKernel) -> str:
    """Share the full/SR/LR Cartesian recurrence and its forward derivative."""
    range_name = {
        CoulombKernelFamily.FULL_RANGE: "Full",
        CoulombKernelFamily.LONG_RANGE: "Long",
        CoulombKernelFamily.SHORT_RANGE: "Short",
    }[radial.family]
    dispatch = "\n".join(
        f"    case {order}: return evaluate_order<{order}>(exponents, centers, angular, mask);"
        for order in range(13)
    )
    return (
        emit_direct_cartesian_cpu_header()
        + r"""
#include "integrals/range_exchange_runtime.hpp"
namespace {
using namespace generativeqc::scf::cuda_execution;
struct Program {
  template <unsigned Order>
  static Dual3 evaluate_order(const double* exponents, const double* centers,
                             const Angular* angular, unsigned mask) {
    Vec3<Dual3> positions[4];
    for (unsigned slot = 0; slot < 4; ++slot) {
      const double seed = (mask & (1U << slot)) ? 1.0 : 0.0;
      positions[slot] = {{centers[3*slot], seed, 0.0, 0.0},
                         {centers[3*slot+1], 0.0, seed, 0.0},
                         {centers[3*slot+2], 0.0, 0.0, seed}};
    }
    return primitive_eri_cartesian<Order>(
        exponents[0], positions[0], angular[0], exponents[1], positions[1], angular[1],
        exponents[2], positions[2], angular[2], exponents[3], positions[3], angular[3],
        generativeqc::integrals::CoulombRange::@RANGE@, @OMEGA@);
  }
  static Dual3 evaluate(const double* exponents, const double* centers,
                        const Angular* angular, unsigned mask) {
    unsigned order = 0;
    for (unsigned slot = 0; slot < 4; ++slot) order += angular_total(angular[slot]);
    switch (order) {
@DISPATCH@
      default: return {NAN, NAN, NAN, NAN};
    }
  }
};
}
""".replace("@RANGE@", range_name)
        .replace("@OMEGA@", float(radial.omega).hex())
        .replace("@DISPATCH@", dispatch)
    )


def emit_range_exchange_cpu(
    radial: CoulombKernel, *, entry_name: str = "generativeqc_range_exchange_cpu"
) -> str:
    """Freeze an SR/LR operator and stream normalized public AO primitives.

    The finite s/p/d/f dispatch shares all recurrence/forward-AD arithmetic
    with CUDA's exact exchange and the CPU full-range primitive owner.
    """
    if not isinstance(radial, CoulombKernel) or radial.family not in (
        CoulombKernelFamily.SHORT_RANGE,
        CoulombKernelFamily.LONG_RANGE,
    ):
        raise ValueError("CPU range exchange requires an explicit SR/LR kernel")
    if not entry_name.isidentifier() or not entry_name.isascii():
        raise ValueError("range exchange entry must be an ASCII C identifier")
    return (
        _emit_primitive_program(radial)
        + r"""
extern "C" int generativeqc_range_exchange_cpu(
    const double* packed, std::size_t natom, std::size_t nprimitive, std::size_t nao,
    const std::int64_t* indices, const double* weights, std::size_t count,
    bool independent_slots, double* output, std::uint64_t* visits) {
  return generativeqc::integrals::range_exchange_records<Program>(
      packed, natom, nprimitive, nao, indices, weights, count,
      independent_slots, output, visits);
}
""".replace("generativeqc_range_exchange_cpu", entry_name)
    )


def emit_full_range_primitive_derivative_cpu(
    *, entry_name: str = "generativeqc_full_range_primitive_derivative_cpu"
) -> str:
    """Emit bounded unnormalized s/p/d/f derivatives in independent slot order.

    Native shell contraction owns public-AO normalization and reuses each
    primitive result across its Coulomb/exchange weight channels. Recover
    the fourth center by translation, as in the shared range consumer.
    Invalid input or numerical failure leaves the caller's output untouched.
    """
    if not entry_name.isidentifier() or not entry_name.isascii():
        raise ValueError("primitive derivative entry must be an ASCII C identifier")
    radial = CoulombKernel(CoulombKernelFamily.FULL_RANGE, 0.0)
    return (
        _emit_primitive_program(radial)
        + r"""
extern "C" int generativeqc_full_range_primitive_derivative_cpu(
    const double* exponents, const double* centers, const unsigned* powers, double* output) {
  if (!exponents || !centers || !powers || !output) return 1;
  Angular angular[4];
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (!std::isfinite(exponents[slot]) || exponents[slot] <= 0) return 1;
    unsigned order = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (powers[3*slot+axis] > 3 || !std::isfinite(centers[3*slot+axis])) return 1;
      order += powers[3*slot+axis];
    }
    if (order > 3) return 1;
    angular[slot] = {powers[3*slot], powers[3*slot+1], powers[3*slot+2]};
  }
  double candidate[12]{};
  for (unsigned slot = 0; slot < 3; ++slot) {
    const auto result = Program::evaluate(exponents, centers, angular, 1U << slot);
    if (!std::isfinite(result.value)) return 2;
    const double derivative[3]{result.derivative_x, result.derivative_y, result.derivative_z};
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(derivative[axis])) return 2;
      candidate[3*slot+axis] = derivative[axis];
      candidate[9+axis] -= derivative[axis];
    }
  }
  for (double value : candidate)
    if (!std::isfinite(value)) return 2;
  std::copy_n(candidate, 12, output);
  return 0;
}
""".replace("generativeqc_full_range_primitive_derivative_cpu", entry_name)
    )
