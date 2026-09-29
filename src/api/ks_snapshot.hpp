#pragma once

#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"

/** Private ctypes bridge for #163; deliberately absent from the installed API.
 * A snapshot owns its token and arrays, but borrows no calculation pointer.
 * Reads/checks require a live batch and compare its current #162 token. */
struct generativeqc_ks_snapshot;
struct generativeqc_ks_xc_response;

extern "C" {
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
generativeqc_status generativeqc_ks_snapshot_cuda_integral_gradient_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count, std::size_t maximum_bytes, std::uint64_t* work, std::size_t work_count);
/** Method-neutral prepared Direct shell J'/K' consumer.
 * values is [2, Natom, 3]; work is retained bytes + final-state D2H/read/sync deltas. */
generativeqc_status generativeqc_ks_snapshot_cuda_shell_full_range_gradient_v1(
    generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count, std::uint64_t* work, std::size_t work_count);
generativeqc_status generativeqc_ks_snapshot_energy_v1(const generativeqc_batch* batch,
                                                       const generativeqc_ks_snapshot* snapshot,
                                                       double* energy);
/** Live native proof: 0=all-electron, 1=ECP; never inferred from electron count. */
/** Current-owner proof of the complete RSH/VV10 model, including its density domain. */
generativeqc_status generativeqc_ks_snapshot_wb97mv_model_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, double* values,
    std::size_t count);

generativeqc_status generativeqc_ks_snapshot_hamiltonian_v1(
    const generativeqc_batch* batch, const generativeqc_ks_snapshot* snapshot, std::uint32_t* kind);
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
