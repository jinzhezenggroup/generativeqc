#ifndef GENERATIVEQC_MOLECULE_BASIS_TEMPLATE_HPP
#define GENERATIVEQC_MOLECULE_BASIS_TEMPLATE_HPP

#include <span>
#include <vector>

#include "molecule/basis_catalog.hpp"

namespace generativeqc::molecule {

/** Owned element-local adapter storage, not a second normalized basis owner.
 * descriptor() borrows these vectors; do not mutate them until its consumer returns.
 */
struct BasisTemplateData {
  std::vector<generativeqc_basis_element_template_v1> elements;
  std::vector<generativeqc_basis_shell_template_v1> shells;
  std::vector<double> exponents;
  std::vector<double> coefficients;
  std::vector<generativeqc_basis_ecp_template_v1> ecp_terms;

  generativeqc_basis_template_descriptor_v1 descriptor() const;
};

/** Validate element-local FP64 data, preflight expanded dimensions, then map atoms.
 * Native metadata only: normalization and scientific/backend admission remain in
 * system_create[_ecp]. Input ranges can be shared, but element labels are unique.
 */
ExpandedBasis expand_basis_template(const generativeqc_basis_template_descriptor_v1& basis,
                                    std::span<const generativeqc_atom> atoms,
                                    generativeqc_basis_representation representation);

}  // namespace generativeqc::molecule

#endif
