#ifndef GENERATIVEQC_CLI_MOLECULAR_SYSTEM_HPP
#define GENERATIVEQC_CLI_MOLECULAR_SYSTEM_HPP

#include <algorithm>

#include "cli/native_basis.hpp"
#include "generativeqc/generativeqc.hpp"

namespace generativeqc::cli {

/** Bind the complete Hamiltonian when creating a fresh molecular geometry.
 * Scalar ECPs change both electron/ionic core counts and residual potentials.
 * Core-only records must not silently become all-electron systems when their
 * residual term list is empty. Native construction copies the supplied arrays;
 * its usual validation and Context lifetime requirements remain authoritative.
 */
inline System make_molecular_system(Context& context,
                                    const generativeqc_system_descriptor& geometry,
                                    const NativeBasisData& basis) {
  if (!basis.core_electrons.empty() && basis.core_electrons.size() != geometry.atom_count)
    throw Error(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                "molecular ECP core counts must match atom count");
  const bool has_ecp = !basis.ecp_terms.empty() ||
                       std::any_of(basis.core_electrons.begin(), basis.core_electrons.end(),
                                   [](std::int32_t core) { return core != 0; });
  return has_ecp ? System(context, geometry, basis.core_electrons, basis.ecp_terms)
                 : System(context, geometry);
}

}  // namespace generativeqc::cli

#endif
