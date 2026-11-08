#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "runtime/cuda_lowering_profile.hpp"

namespace generativeqc::scf {

/** Explicit packed-projection controls remain available for paired ablations. */
inline bool df_packed_projection_blas_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_PACKED_PROJECTION");
  return value && std::strcmp(value, "blas") == 0;
}

inline bool df_packed_projection_auto_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_PACKED_PROJECTION");
  return !value || std::strcmp(value, "auto") == 0;
}

inline bool df_packed_projection_policy_valid() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_PACKED_PROJECTION");
  return !value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "generated") == 0 ||
         std::strcmp(value, "blas") == 0;
}

/** Conservative work/resource domain for full-throughput FP64 hardware.
 * The extra unpack traverses 2*n*n*a doubles; enough occupied columns and
 * batched rows must amortize that traffic and the library launches. Small or
 * narrow contractions and throughput-limited FP64 retain generated execution.
 * The CUDA throughput ratio is a capability, never a GPU-name allowlist.
 * This measured admission is not a universal latency model for all shapes.
 */
inline constexpr runtime::CudaLoweringProfile kDfPackedProjectionProfile{
    false, {256, 1024, 64, 8}, 2};

/** Shape-only admission avoids probing hardware for bounded incumbent plans.
 * Axes are AO columns, auxiliary functions, occupied rank and scratch rows.
 * The older projection profile qualifies scalar FP64 throughput, independently
 * of the stricter matrix-acceleration requirement used by occupied Gram.
 */
inline bool df_packed_projection_work_admitted(std::size_t ao_columns,
                                               std::size_t auxiliary_columns, std::size_t rank,
                                               std::size_t capacity) noexcept {
  return ao_columns && auxiliary_columns && rank <= ao_columns &&
         runtime::cuda_lowering_workload_rejection(
             kDfPackedProjectionProfile,
             {ao_columns, auxiliary_columns, rank, capacity / ao_columns / auxiliary_columns})
             .empty();
}

inline bool df_packed_projection_preferred(std::size_t n, std::size_t a, std::size_t rank,
                                           std::size_t capacity,
                                           const runtime::CudaTargetInfo& target) noexcept {
  return df_packed_projection_work_admitted(n, a, rank, capacity) &&
         runtime::cuda_lowering_profile_rejection(kDfPackedProjectionProfile, target,
                                                  {n, a, rank, capacity / n / a})
             .empty();
}

/** Diagnostic legacy contraction requires a new owner: captured J/K nodes
 * and the immutable raw-buffer lifetime must follow the same frozen policy. */
inline bool df_resident_exchange_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_RESIDENT_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "full") == 0 ||
         std::strcmp(value, "flat") == 0;
}

inline bool df_triangular_exchange_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_RESIDENT_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "flat") == 0;
}

/** Keep the original flattened dense contraction as an isolated ablation. */
inline bool df_flat_dense_exchange_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_RESIDENT_EXCHANGE");
  return value && std::strcmp(value, "flat") == 0;
}

/** Freeze the DIIS reduction policy with captured SCF work. */
inline bool df_cooperative_diis_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_DIIS_DOTS");
  return !value || std::strcmp(value, "auto") == 0;
}

/** An unset policy is auto; explicit dense/occupied remain comparison controls. */
inline bool df_occupied_exchange_auto_requested() noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0;
}

/** Backend-independent work policy for the resident FP64 occupied algorithm.
 * Dense K costs 4*a*n^3 FLOPs; occupied projection plus a full Gram costs
 * at most 4*a*n^2*r (SYRK can reduce this further). Require a twofold arithmetic
 * reduction to leave headroom for factor validation and smaller BLAS shapes.
 * This is a conservative work heuristic, not a device latency prediction.
 * Division keeps both the work comparison and native BLAS bounds overflow-safe.
 * Residency, storage, RHF provenance and density validity are separate gates.
 */
inline bool df_occupied_exchange_preferred(std::size_t nbf, std::size_t naux, std::size_t batch,
                                           std::size_t rank) noexcept {
  constexpr auto blas_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return batch == 1 && nbf >= 2 && nbf <= blas_limit / nbf && naux > 0 &&
         naux <= blas_limit / nbf && rank > 0 && rank <= nbf / 2;
}

/** Reserve automatic factors only with a method-authorized RHF occupation.
 * Zero means unknown reference/rank or UHF: those plans keep dense capacity.
 * Explicit occupied retains its conservative reservation for all references.
 */
inline bool df_occupied_exchange_requested(std::size_t nbf = 0, std::size_t naux = 0,
                                           std::size_t batch = 0,
                                           std::size_t automatic_rhf_rank = 0) noexcept {
  const char* value = std::getenv("GENERATIVEQC_DF_EXCHANGE");
  return (value && std::strcmp(value, "occupied") == 0) ||
         (df_occupied_exchange_auto_requested() && df_resident_exchange_requested() &&
          df_occupied_exchange_preferred(nbf, naux, batch, automatic_rhf_rank));
}

}  // namespace generativeqc::scf
