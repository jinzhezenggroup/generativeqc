"""Host lowering of the shared bounded Cartesian range-ERI derivative owner."""

from .direct_cartesian_contraction_cuda import emit_direct_cartesian_cpu_header
from .range_separation import CoulombKernel, CoulombKernelFamily


def emit_range_exchange_cpu(radial: CoulombKernel) -> str:
    """Freeze the radial operator, not a functional or a Cartesian component.

    The native consumer expands already normalized public AO records and
    streams contracted primitives. All recurrence/forward-AD arithmetic is
    the same owner used by CUDA's exact range exchange. Its finite angular
    dispatch covers s/p/d/f without component-specific artifacts.
    """
    if not isinstance(radial, CoulombKernel) or radial.family not in (
        CoulombKernelFamily.SHORT_RANGE,
        CoulombKernelFamily.LONG_RANGE,
    ):
        raise ValueError("CPU range exchange requires an explicit SR/LR kernel")
    range_name = "Long" if radial.family == CoulombKernelFamily.LONG_RANGE else "Short"
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
extern "C" int generativeqc_range_exchange_cpu(
    const double* packed, std::size_t natom, std::size_t nprimitive, std::size_t nao,
    const std::int64_t* indices, const double* weights, std::size_t count,
    bool independent_slots, double* output, std::uint64_t* visits) {
  return generativeqc::integrals::range_exchange_records<Program>(
      packed, natom, nprimitive, nao, indices, weights, count,
      independent_slots, output, visits);
}
""".replace("@RANGE@", range_name)
        .replace("@OMEGA@", float(radial.omega).hex())
        .replace("@DISPATCH@", dispatch)
    )
