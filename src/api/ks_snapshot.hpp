#pragma once

#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"

/** Private ctypes bridge for #163; deliberately absent from the installed API.
 * A snapshot owns its token and arrays, but borrows no calculation pointer.
 * Reads/checks require a live batch and compare its current #162 token. */
struct generativeqc_ks_snapshot;
struct generativeqc_ks_xc_response;
struct generativeqc_nonlocal_cuda_force;
namespace generativeqc::dft {
struct GridTaskView;
}

extern "C" {
/** Private, explicitly partial #2151 native single-system derivative bridge.
 * Only a successfully executed prepared KS calculation with a current verified
 * final state may provide source-major [H', Pulay, J', K'] +dE/dR blocks.
 * Source size is exactly 4*3*Natom doubles; work has exactly nine uint64 slots.
 * CPU DF and qualified CUDA DF/Direct are admitted; CPU Direct and ECP/range/
 * nonlocal/dispersion domains fail closed. Caller output is unchanged on
 * failure. XC/Becke/nuclear terms are absent; this is NOT an analytic force. */
generativeqc_status generativeqc_ks_calculation_integral_sources_v1(
    generativeqc_calculation* calculation, double* values, std::size_t count,
    std::size_t maximum_bytes, std::uint64_t* work, std::size_t work_count);

generativeqc_status generativeqc_ks_snapshot_create_v1(generativeqc_batch* batch, std::size_t index,
                                                       generativeqc_ks_snapshot** output,
                                                       std::uint64_t* metadata,
                                                       std::size_t metadata_count);
generativeqc_status generativeqc_ks_snapshot_check_v1(const generativeqc_batch* batch,
                                                      const generativeqc_ks_snapshot* snapshot);
generativeqc_status generativeqc_ks_snapshot_copy_v1(const generativeqc_batch* batch,
                                                     const generativeqc_ks_snapshot* snapshot,
                                                     double* values, std::size_t count);
void generativeqc_ks_snapshot_destroy_v1(generativeqc_ks_snapshot* snapshot);
generativeqc_status generativeqc_ks_snapshot_cuda_full_range_derivatives_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count);
generativeqc_status generativeqc_ks_snapshot_cuda_integral_gradient_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count, std::size_t maximum_bytes, std::uint64_t* work, std::size_t work_count);

/** Explicit output layout: combined=1 publishes [H', Pulay, J'+K'];
 * combined=0 retains the v1 independent-source layout. No raw J/K source is
 * fabricated for total-force consumers. Snapshot/token/budget gates are shared. */
generativeqc_status generativeqc_ks_snapshot_cuda_integral_gradient_v2(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, int combined,
    double* values, std::size_t count, std::size_t maximum_bytes, std::uint64_t* work,
    std::size_t work_count);

generativeqc_status generativeqc_ks_snapshot_density_fitted_integral_gradient_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count, std::size_t maximum_bytes, std::uint64_t* work, std::size_t work_count);
generativeqc_status generativeqc_ks_snapshot_cuda_resident_density_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, int* device,
    const double** alpha, const double** beta, std::size_t* matrix_elements, unsigned* spins,
    void** source_stream);
/** Four CUDA-event milliseconds [J, full-K, range-K, semilocal-XC] plus a
 * presence bitmask. This private diagnostic replays the exact resident D and
 * never runs an SCF iteration. */
generativeqc_status generativeqc_ks_snapshot_cuda_fixed_density_profile_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* milliseconds,
    std::size_t count, std::uint32_t* present_mask);
generativeqc_status generativeqc_ks_snapshot_cuda_resident_grid_v2(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, int* device,
    const double** points, const double** weights, const double** atomic_weights,
    std::size_t* point_count);
/** Seed the resident nonlocal force owner directly from the exact current
 * CUDA KS rho/grad-rho binding on the borrowed grid task stream. No pointer is
 * published to Python and no host transfer or synchronization occurs. */
generativeqc_status generativeqc_ks_snapshot_cuda_seed_nonlocal_force_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot,
    generativeqc_nonlocal_cuda_force* owner, const generativeqc::dft::GridTaskView* view);
generativeqc_status generativeqc_ks_snapshot_energy_v1(const generativeqc_batch* batch,
                                                       const generativeqc_ks_snapshot* snapshot,
                                                       double* energy);
/** Live native proof: 0=all-electron, 1=ECP; never inferred from electron count. */
/** Current-owner proof of a complete range-exchange/nonlocal model, including its domain. */
generativeqc_status generativeqc_ks_snapshot_nonlocal_model_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count);
/** Compatibility alias for the original WB97M-V-specific symbol. */
generativeqc_status generativeqc_ks_snapshot_wb97mv_model_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count);

generativeqc_status generativeqc_ks_snapshot_hamiltonian_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, std::uint32_t* kind);
/** Current primary Fock-provider proof. Approximation values follow the private
 * FockApproximation enum; an absent exchange term is UINT32_MAX. */
generativeqc_status generativeqc_ks_snapshot_fock_provider_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot,
    std::uint32_t* coulomb_approximation, std::uint32_t* exchange_approximation,
    double* metric_relative_threshold);
/** Private bounded CUDA XC response owner. It copies the successful state's
 * exact density/basis/grid and retains its token. Every execute requires the
 * live batch; no snapshot pointer is borrowed by the native owner. */
generativeqc_status generativeqc_ks_xc_response_create_v1(generativeqc_batch* batch,
                                                          const generativeqc_ks_snapshot* snapshot,
                                                          std::size_t tile_points,
                                                          std::size_t budget_bytes,
                                                          generativeqc_ks_xc_response** output);
generativeqc_status generativeqc_ks_xc_response_apply_v1(generativeqc_batch* batch,
                                                         generativeqc_ks_xc_response* response,
                                                         const double* direction, std::size_t count,
                                                         double* output, std::size_t output_count);
/** Twelve uint64 values: device bytes, setup H2D, action H2D, D2H, syncs,
 * enqueues, spin blocks, AO count, grid points, preparation snapshot-export
 * D2H/reads/syncs. Enqueues count submitted actions, including later numerical
 * rejection; syncs also include the owner's explicit failure-cleanup fences. */
generativeqc_status generativeqc_ks_xc_response_diagnostic_v1(
    const generativeqc_ks_xc_response* response, std::uint64_t* values, std::size_t count);
void generativeqc_ks_xc_response_destroy_v1(generativeqc_ks_xc_response* response);
/** Private CPU RKS directional potential bridge; rho/gradient use total density.
 * On failure, callers must discard the output buffer, including completed rows. */
generativeqc_status generativeqc_xc_rks_response_batch_v1(
    std::uint32_t pbe, const double* rho, const double* gradient, const double* delta_rho,
    const double* delta_gradient, std::size_t point_count, double* values, std::size_t value_count);
/** Spin-major rho[2,n] and gradient[2,n,3]; outputs point-major rho[2], gradient[2,3].
 * As for the restricted bridge, discard the complete output after any failure. */
generativeqc_status generativeqc_xc_uks_response_batch_v1(
    std::uint32_t pbe, const double* rho, const double* gradient, const double* delta_rho,
    const double* delta_gradient, std::size_t point_count, double* values, std::size_t value_count);
generativeqc_status generativeqc_ks_snapshot_ecp_derivatives_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count);
}
