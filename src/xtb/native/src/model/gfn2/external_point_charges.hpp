#ifndef GENERATIVEQC_XTB_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GENERATIVEQC_XTB_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "runtime/types.hpp"

namespace generativeqc::xtb::detail::gfn2 {

/*
 * Geometry-independent layout and shell hardnesses for GFN2 external point
 * charges. Point sites are a ragged, per-molecule partition and never interact
 * with sites or QM atoms from another batch member.
 *
 * shell_hardness stores the Hartree-valued parameter
 *
 *   element_hardness * shell_hubbard_scale
 *
 * in BasisPlan shell order. Plan construction may allocate. All evaluation
 * routines below only read this plan and do not allocate on successful calls.
 */
struct ExternalPointChargePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_point_charges = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
};

/*
 * Create a reusable external-charge plan for an existing BasisPlan.
 * point_charge_offsets has batch_size + 1 entries and ends at
 * total_point_charges. For a batch with no point sites, a NULL offsets pointer
 * is accepted and interpreted as an all-zero partition.
 *
 * atomic_numbers must describe the BasisPlan atoms in the same order. Besides
 * selecting generated GFN2 hardnesses, this is checked against the basis shell
 * layout so an accidentally mismatched element list fails during setup.
 */
generativeqc_xtb_status_t make_external_point_charge_plan(const BasisPlan& basis,
                                                 const std::int32_t* atomic_numbers,
                                                 std::int64_t total_point_charges,
                                                 const std::int64_t* point_charge_offsets,
                                                 ExternalPointChargePlan& plan, std::string& error);

/*
 * Accumulate explicit external-charge forces for fixed converged shell
 * charges. QM and point-site force buffers use atom/site-major xyz in
 * Hartree/bohr and may independently be NULL. There is no point-charge to
 * point-charge term. Coincident QM/site positions are valid for finite
 * positive hardnesses and contribute zero force.
 */
generativeqc_xtb_status_t add_external_point_charge_forces_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, const double* shell_charges,
    double* qm_forces, double* point_forces, std::string& error);

}  // namespace generativeqc::xtb::detail::gfn2

#endif  // GENERATIVEQC_XTB_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP
