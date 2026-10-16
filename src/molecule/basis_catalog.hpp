#ifndef GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP
#define GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP

#include <cstddef>
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

/** Shape-only expansion counts, not complete normalized-system admission.
 * buffer_bytes covers detached orbital shell/primitive buffers only; per-atom
 * core counts, scalar ECP terms, normalization and allocator overhead are separate.
 */
struct BasisExpansionCapacity {
  std::size_t shell_count{}, primitive_count{}, ao_count{}, buffer_bytes{}, ecp_term_count{};
};

/** Canonical JSON checksum of the complete build-time bundled source pack. */
const char* bundled_basis_catalog_sha256() noexcept;

/** Names borrow immutable compiled storage; no file/network/context is accessed. */
std::size_t bundled_basis_count() noexcept;
std::string_view bundled_basis_name(std::size_t index) noexcept;
std::string_view bundled_basis_pack_checksum() noexcept;
std::string_view canonical_basis_name(std::string_view name);
std::vector<std::string_view> bundled_basis_names();
bool bundled_basis_supports(std::string_view name, int atomic_number);

/** Query the same general-template preflight used by actual bundled expansion.
 * Coordinates, charge and spin do not participate in these detached-buffer counts.
 */
BasisExpansionCapacity bundled_basis_capacity(std::string_view name,
                                              std::span<const generativeqc_atom> atoms,
                                              generativeqc_basis_representation representation);

/** Expand raw contractions in atom/source order, preserving zeros and coefficients.
 * Case and underscore aliases are accepted. Unknown names/elements, invalid
 * representations and dimensions outside the descriptor domain are rejected.
 * This is metadata work, not integral evaluation or scientific normalization.
 */
ExpandedBasis expand_bundled_basis(std::string_view name, std::span<const generativeqc_atom> atoms,
                                   generativeqc_basis_representation representation);

}  // namespace generativeqc::molecule

#endif
