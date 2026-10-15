#ifndef GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP
#define GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::molecule {

/** Unnormalized descriptor storage; system creation owns validation/normalization.
 * Local canonical records also use this storage for their ECP and provenance data.
 */
struct ExpandedBasis {
  std::vector<generativeqc_primitive> primitives;
  std::vector<generativeqc_shell> shells;
  generativeqc_basis_representation representation{GENERATIVEQC_BASIS_CARTESIAN};
  std::vector<std::int32_t> core_electrons;
  std::vector<generativeqc_ecp_term> ecp_terms;
  std::string identity;
};

/** Canonical JSON checksum of the complete build-time bundled source pack. */
const char* bundled_basis_catalog_sha256() noexcept;

/** Names borrow immutable compiled storage; no file/network/context is accessed. */
std::vector<std::string_view> bundled_basis_names();
bool bundled_basis_supports(std::string_view name, int atomic_number);

/** Expand raw contractions in atom/source order, preserving zeros and coefficients.
 * Case and underscore aliases are accepted. Unknown names/elements, invalid
 * representations and dimensions outside the descriptor domain are rejected.
 * This is metadata work, not integral evaluation or scientific normalization.
 */
ExpandedBasis expand_bundled_basis(std::string_view name, std::span<const generativeqc_atom> atoms,
                                   generativeqc_basis_representation representation);

}  // namespace generativeqc::molecule

#endif
