#pragma once

#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"

/** Private synchronous Python/native bridge, not a new stable public ABI.
 * The live batch supplies the accepted state and producer stream; callers must
 * not create/export a KS snapshot. All capacities include independent response
 * storage, while immutable SCF resources remain charged to their producer.
 */
struct generativeqc_ks_hessian_options_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_hessian_options_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  std::size_t device_bytes{2ULL << 30};
  std::size_t integral_bytes{64ULL << 20};
  std::size_t grid_bytes{128ULL << 20};
  std::size_t response_bytes{64ULL << 20};
  std::size_t output_bytes{64ULL << 20};
  std::uint64_t max_nonlocal_pairs{16'777'216};
  std::size_t restart{30};
  std::size_t max_iterations{200};
  unsigned reorthogonalizations{2};
  double relative_tolerance{1e-10};
  double absolute_tolerance{};
};

/** Terminal scalar/work receipt only; no orbital, density or grid panels cross
 * this boundary. Response residuals describe the last completed direction.
 */
struct generativeqc_ks_hessian_receipt_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_hessian_receipt_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  std::uint64_t device_bytes{}, completed_directions{}, induced_fock_actions{};
  std::uint64_t exact_primitive_components{}, one_electron_primitive_components{};
  std::uint64_t df_value_passes{}, df_contraction_passes{};
  std::uint64_t grid_right_directions{}, grid_mixed_contractions{}, vv10_reference_pairs{};
  std::uint64_t response_iterations{}, response_workspace_bytes{};
  std::uint64_t ecp_workspace_bytes{}, df_workspace_bytes{};
  double response_rhs_norm{}, response_residual_norm{}, response_tolerance{};
  double execution_seconds{};
};

extern "C" {
/** Zero means this library has no linked CUDA Hessian package. */
std::uint32_t generativeqc_ks_hessian_cuda_version_v1();

/** A null direction with zero count requests the raw row-major Hessian;
 * otherwise direction/output are atom-major Cartesian HVP panels. Host input,
 * output and receipt must be disjoint. Counts and capacities are checked before
 * allocation. Only the complete terminal result is downloaded, then published
 * after convergence and stream completion. Any failure preserves both caller
 * outputs. Missing linked selectors fail explicitly, never compile or fall back.
 */
generativeqc_status generativeqc_ks_hessian_cuda_v1(
    generativeqc_batch* batch, std::size_t index, const generativeqc_ks_hessian_options_v1* options,
    const double* direction, std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_hessian_receipt_v1* receipt);
}
