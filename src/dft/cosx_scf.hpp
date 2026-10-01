#ifndef GENERATIVEQC_DFT_COSX_SCF_HPP
#define GENERATIVEQC_DFT_COSX_SCF_HPP

#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cosx_fock_provider.hpp"
#include "scf/types.hpp"

namespace generativeqc::dft {

/** Host-controlled RHF using the prepared RI-J/COSX-K value/force provider. */
scf::ScfResult run_cosx_rhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

/** Host-controlled UHF using independent alpha/beta COSX value/force exchange. */
scf::ScfResult run_cosx_uhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

}  // namespace generativeqc::dft

namespace generativeqc::scf {

/** Internal host-controlled PBE0 RKS over explicit RI/direct-J + COSX-K.
 * Public/AUTO provider selection remains unchanged. */
ScfResult run_pbe0_cosx_rks(dft::PreparedCosxFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

/** Spin-polarized counterpart of run_pbe0_cosx_rks. */
ScfResult run_pbe0_cosx_uks(dft::PreparedCosxFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

}  // namespace generativeqc::scf

#endif
