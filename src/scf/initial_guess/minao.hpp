#ifndef GENERATIVEQC_SCF_INITIAL_GUESS_MINAO_HPP
#define GENERATIVEQC_SCF_INITIAL_GUESS_MINAO_HPP

#include <cstddef>

#include "scf/initial_guess/density.hpp"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::integrals {
struct IntegralData;
}

namespace generativeqc::scf::initial_guess {

struct MinaoDensityResult {
  Matrix density;
  std::size_t source_aos{};
  std::size_t source_primitives{};
  double source_electrons{};
  double projected_electrons{};
};

/** PySCF-compatible occupied ANO MINAO projection for all-electron H-Ar systems.
 *
 * The returned density is the raw projected MINAO density. Target electron-count
 * normalization remains the existing seed-boundary policy, not part of this
 * projection.
 */
MinaoDensityResult minao_density(const core::System& system,
                                 const integrals::IntegralData& target,
                                 const Matrix& target_orthogonalizer);

std::size_t minao_source_ao_count(const core::System& system);
std::size_t minao_source_primitive_count(const core::System& system);

}  // namespace generativeqc::scf::initial_guess

#endif
