#ifndef GENERATIVEQC_INTEGRALS_MINAO_BASIS_HPP
#define GENERATIVEQC_INTEGRALS_MINAO_BASIS_HPP

#include <cstddef>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::integrals {

struct MinaoBasisSource {
  core::System system;
  std::vector<double> occupations;
  std::size_t primitive_count{};
};

/** Build the normalized occupied ANO/MINAO source used by the SCF seed layer.
 *
 * This owns basis construction/normalization so shared initial-guess code only
 * consumes core/integrals interfaces.
 */
MinaoBasisSource make_minao_basis_source(const core::System& target);
std::size_t minao_basis_ao_count(const core::System& target);
std::size_t minao_basis_primitive_count(const core::System& target);

}  // namespace generativeqc::integrals

#endif
