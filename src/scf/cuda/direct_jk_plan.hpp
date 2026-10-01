#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <vector>

#include "scf/cuda/direct_coulomb.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda_direct_jk.hpp"

namespace generativeqc::scf {

/** Select resident value sources without changing the requested mathematics.
 * Generated J/K are exact FP64 full-range consumers on the same provider stream.
 * Mixed-J, range-separated K and missing optional capacity deliberately retain
 * the generic source.
 */
struct DirectJkValueDispatch {
  bool generated_coulomb{}, generated_exchange{}, generic_coulomb{}, generic_exchange{};
};
constexpr DirectJkValueDispatch direct_jk_value_dispatch(bool generated_coulomb_available,
                                                         bool generated_exchange_available,
                                                         bool want_coulomb, bool want_exchange,
                                                         bool mixed_coulomb) noexcept {
  const bool generated_coulomb = generated_coulomb_available && want_coulomb && !mixed_coulomb;
  const bool generated_exchange = generated_exchange_available && want_exchange && !mixed_coulomb;
  return {generated_coulomb, generated_exchange, want_coulomb && !generated_coulomb,
          want_exchange && !generated_exchange};
}
constexpr DirectJkValueDispatch direct_jk_value_dispatch(bool generated_coulomb_available,
                                                         bool want_coulomb, bool want_exchange,
                                                         bool mixed_coulomb) noexcept {
  return direct_jk_value_dispatch(generated_coulomb_available, false, want_coulomb, want_exchange,
                                  mixed_coulomb);
}

/** Own one exact public-AO provider source and its density/output scratch.
 * Kernel consumers borrow the packed view; the plan drains its stream before
 * releasing buffers. Layout and explicit budget accounting remain unchanged.
 */
struct CudaDirectJkPlan {
  int device_id{-1};
  cuda_execution::DeviceBatch batch{};
  cudaStream_t stream{};
  unsigned derivative_order{};
  std::size_t matrix_elements{}, coordinates_per_item{}, coordinate_elements{};
  double screening_tolerance{};
  double *density{}, *beta{}, *coulomb{}, *alpha_exchange{}, *beta_exchange{}, *bounds{},
      *derivative{};
  int* numerical_failure{};
  std::vector<void*> allocations;
  std::size_t device_bytes{};
  CudaDirectJkDiagnostic diagnostic{};
  std::unique_ptr<cuda_execution::GeneratedCoulombPlan> generated_coulomb;
  std::unique_ptr<cuda_execution::GeneratedExchangePlan> generated_exchange;
  /** Automatic symmetry-canonical source for through-f plans. Angular buckets
   * keep each kernel's recurrence order fixed, including f-shell quartets.
   * All storage is charged to the existing optional provider budget. */
  const std::int32_t* canonical_pairs{};
  /** Cartesian consumers borrow HF's normalized source/projection ABI. Public
   * matrix dimensions remain in batch/diagnostic; source strides live here. */
  cuda_execution::DeviceBatch canonical_batch{};
  bool canonical_cartesian{};
  double* canonical_bounds{};
  const double* canonical_transform{};
  /** Shell-local transform support; null keeps the shared HF dense projection. */
  const std::int32_t* canonical_projection_spans{};
  std::uint8_t* canonical_active{};
  double *canonical_public_density{}, *canonical_public_output{}, *canonical_projection{},
      *canonical_zero{};
  std::vector<std::array<std::size_t, 8>> canonical_pair_offsets;
  /** Geometry-bound descending Schwarz order and inclusive ket-row spans.
   * Optional O(NAO^2) metadata removes rejected quartets before traversal.
   * If its charged workspace does not fit, the dense canonical source remains. */
  const std::int32_t* canonical_pair_order{};
  const std::uint64_t* canonical_row_prefix{};
  /** Derivative-capable canonical plans may retain shell AO offsets/pairs in
   * batch so HF's one-electron kernel can borrow metadata and derivative scratch. */
  double *canonical_density{}, *canonical_coulomb{}, *canonical_exchange{};
  /** Borrowed test/profiler census: candidate quartets and radial evaluations.
   * Null in production. The observer owns storage and stream-ordered lifetime. */
  std::uint64_t* canonical_work_count{};
  ~CudaDirectJkPlan();
};

/** Complete full-range shell value ownership may be either entirely
 * generated/native or generated/native plus HF's bounded higher-l fallback. */
inline bool direct_jk_generated_full_range_value_available(const CudaDirectJkPlan& plan) noexcept {
  return plan.generated_exchange != nullptr && plan.generated_exchange->shared != nullptr &&
         (plan.generated_exchange->shared->value_capability ||
          plan.generated_exchange->bounded_value_capability);
}

/** A derivative-capable provider owner may still use its full-range value
 * route for a zero-order request. The owner's maximum derivative capability
 * does not select the SCF value schedule. */
inline bool direct_jk_generated_exchange_value_available(const CudaDirectJkPlan& plan,
                                                         const FockBuildSpec& spec) noexcept {
  return direct_jk_generated_full_range_value_available(plan) && spec.derivative_order == 0 &&
         spec.exchange.present && spec.exchange.op == FockOperator::FullRange;
}

}  // namespace generativeqc::scf
