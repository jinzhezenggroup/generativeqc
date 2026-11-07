#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "dft/vv10_rebase_history.hpp"

namespace generativeqc::dft::cuda_ks_detail {
/** One optional lease; all pointers remain stable until drained retirement. */
struct RebaseStorage {
  double *density{}, *bank_density{}, *bank_fock{}, *bank_residual{}, *potential{};
  vv10_rebase::SlotTag *tags{}, *bank_tags{};
  vv10_rebase::Transaction* transaction{};
};
void rebase_reset(cudaStream_t stream, unsigned history, RebaseStorage storage);
void rebase_refuse(cudaStream_t stream, vv10_rebase::Error reason, RebaseStorage storage);
void rebase_record_density(cudaStream_t stream, std::size_t elements, unsigned history,
                           const double* density, const std::uint8_t* enabled,
                           const std::uint32_t* count, const std::uint32_t* head,
                           vv10_rebase::Key key, RebaseStorage storage);
void rebase_snapshot(cudaStream_t stream, std::size_t elements, unsigned history,
                     const double* fock_history, const double* residual_history,
                     const double* old_potential, const std::uint32_t* count,
                     const std::uint32_t* head, vv10_rebase::Key key, RebaseStorage storage);
void rebase_prepare_target(cudaStream_t stream, std::size_t elements, unsigned history,
                           const double* new_potential, const std::uint32_t* count,
                           const std::uint32_t* head, vv10_rebase::Key key, RebaseStorage storage);
/** Checks each source/product and updated F/R for finite values. No live slot
 * is touched here; transaction.ready is sticky false after numerical refusal. */
void rebase_check_finite(cudaStream_t stream, std::size_t elements, const double* values,
                         RebaseStorage storage);
void rebase_accumulate(cudaStream_t stream, std::size_t elements, unsigned bank_slot,
                       const double* product, bool subtract, bool update_fock,
                       RebaseStorage storage);
void rebase_restore(cudaStream_t stream, std::size_t elements, unsigned history,
                    double* fock_history, double* residual_history, RebaseStorage storage);
void rebase_publish(cudaStream_t stream, unsigned history, std::uint32_t* count,
                    std::uint32_t* head, RebaseStorage storage);

/** Detect numerical core-spectrum ties touching the beta frontier. The cold
 * seed, not the physical Hamiltonian, may resolve this otherwise arbitrary
 * gauge. Both adjacent occupied/virtual clusters participate in the check. */
void core_guess_frontier_tie(cudaStream_t stream, std::size_t dimension, int occupied_beta,
                             const double* energies, int* tied);
/** Deterministic 1e-8-Eh diagonal tie-break in the orthogonal cold-guess basis.
 * Never apply this to an SCF, DIIS, final-state or response Fock matrix. */
void break_core_guess_ties(cudaStream_t stream, std::size_t dimension, double* core_guess);
/** Fix each cold orbital's arbitrary sign before mixing distinct columns.
 * The first AO among numerically tied largest components defines its phase. */
void orient_core_guess_orbitals(cudaStream_t stream, std::size_t dimension, double* coefficients);

struct Scalars {
  double one_electron{}, hartree{}, exact_exchange{}, xc{}, residual{}, density_change{};
  double residual_rms{}, density_rms{};
  // Snapshot admission checks the largest AO commutator entry, not its RMS.
  double maximum_residual{};
  double orbital_gradient{};
  double electrons[2]{}, grid_electrons[2]{};
  double energy_change{};
  int failure{};
  // Preserve the unrounded compensated electronic trace for convergence.
  double electronic_energy{}, electronic_energy_correction{};
};

struct Control {
  double previous_energy{};
  double previous_energy_correction{};
  std::uint32_t iterations{};
  int active{}, converged{}, failed{};
};

void reset_control(cudaStream_t stream, unsigned spins, int occupied_alpha, int occupied_beta,
                   Control* control, std::uint8_t* enabled, std::uint8_t* spin_enabled);
/** Merge independently integrated AO potentials and energies. Electron counts
 * remain those of the primary quadrature; either numerical error invalidates it. */
void merge_grid_contributions(cudaStream_t stream, std::size_t elements, double* potential,
                              double* totals, int* error, const double* nonlocal_potential,
                              const double* nonlocal_totals, const int* nonlocal_error);
void assemble_fock(cudaStream_t stream, std::size_t n, unsigned spins, const double* hcore,
                   const double* coulomb, const double* exchange, double exchange_coefficient,
                   const double* range_exchange, double range_exchange_coefficient,
                   const double* potential, const std::uint8_t* enabled, double* fock);
/** Retain a cold-start proposal in otherwise idle DIIS storage. Physical F[D]
 * is immutable; the first/last undamped sweeps never read uninitialized history.
 * The owner must retire this scratch role before admitting Pulay history. */
void cold_start_fock(cudaStream_t stream, std::size_t elements, const double* physical_fock,
                     double* previous_proposal, double* proposal, bool damp);
void stabilize_uks_proposal(cudaStream_t stream, std::size_t n, const double* overlap,
                            const double* occupied_projector, const std::uint8_t* enabled,
                            double* proposal_fock);
void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* exchange,
                 double exchange_coefficient, const double* range_exchange,
                 double range_exchange_coefficient, const double* xc_totals, const int* xc_error,
                 const int* jk_error, const int* range_jk_error, const int* nonlocal_domain_error,
                 const int* nonlocal_pair_error, const int* solver_info,
                 const std::uint8_t* enabled, Scalars* output,
                 bool retain_orbital_gradient = false);
/** Reduce X^T(FDS-SDF)X without retaining an extra resident matrix. */
void orbital_gradient(cudaStream_t stream, std::size_t n, const double* transformed_residual,
                      const std::uint8_t* enabled, Scalars* output);
void advance(cudaStream_t stream, std::size_t n, unsigned spins, double nuclear_repulsion,
             int occupied_alpha, int occupied_beta, double energy_tolerance,
             double density_tolerance, unsigned max_iterations, bool warm_updates, Scalars* current,
             Control* control, const double* proposal, double* density, double* warm,
             std::uint8_t* enabled, std::uint8_t* spin_enabled);
}  // namespace generativeqc::dft::cuda_ks_detail
