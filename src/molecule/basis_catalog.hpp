#ifndef GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP
#define GENERATIVEQC_MOLECULE_BASIS_CATALOG_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::molecule::catalog {

/** Detached native basis buffers. Built-in data is compiled from the audited
 * decimal pack; canonical custom records may additionally supply ECP metadata. */
struct BasisData {
  std::vector<generativeqc_primitive> primitives;
  std::vector<generativeqc_shell> shells;
  generativeqc_basis_representation representation{GENERATIVEQC_BASIS_CARTESIAN};
  std::vector<std::int32_t> core_electrons;
  std::vector<generativeqc_ecp_term> ecp_terms;
  std::string identity;
};

/** Result-buffer capacity, not allocator overhead or complete system admission.
 * Coordinate values are irrelevant to these shape/element-only counts. */
struct Capacity {
  std::size_t shell_count{}, primitive_count{}, ao_count{}, buffer_bytes{};
};

/** Borrowed names refer to immutable AOT storage, independently of CLI/Python. */
GENERATIVEQC_API std::vector<std::string_view> bundled_basis_names();
GENERATIVEQC_API bool bundled_basis_supports(std::string_view name, int atomic_number);

/** Validate native index widths and representation before allocating output.
 * Names retain the established case/underscore aliases; missing elements refuse. */
GENERATIVEQC_API Capacity bundled_basis_capacity(std::string_view name,
                                                 std::span<const generativeqc_atom> atoms,
                                                 generativeqc_basis_representation representation);

/** Expand in caller atom order and original packed shell/primitive order.
 * No runtime JSON, compiler import, normalization or coefficient pruning occurs. */
GENERATIVEQC_API BasisData expand_bundled_basis(std::string_view name,
                                                std::span<const generativeqc_atom> atoms,
                                                generativeqc_basis_representation representation);

}  // namespace generativeqc::molecule::catalog
#endif
