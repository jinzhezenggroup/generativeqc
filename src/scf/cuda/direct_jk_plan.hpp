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
 * Through-f SR/LR K requires the explicit bounded value qualification opt-in;
 * Each channel retains its own fallback: mixed J does not change strict K's
 * provider, and an uncovered exchange operator does not discard generated J.
 */
struct DirectJkValueDispatch {
  bool generated_coulomb{}, generated_exchange{}, generic_coulomb{}, generic_exchange{};
  bool canonical_coulomb{}, canonical_exchange{};
};
constexpr DirectJkValueDispatch direct_jk_value_dispatch(
    bool generated_coulomb_available, bool generated_exchange_available, bool want_coulomb,
    bool want_exchange, bool mixed_coulomb, bool canonical_available = false) noexcept {
  const bool generated_coulomb = generated_coulomb_available && want_coulomb && !mixed_coulomb;
  const bool generated_exchange = generated_exchange_available && want_exchange;
  const bool canonical_coulomb =
      canonical_available && want_coulomb && !generated_coulomb && !mixed_coulomb;
  const bool canonical_exchange = canonical_available && want_exchange && !generated_exchange;
  return {generated_coulomb,
          generated_exchange,
          want_coulomb && !generated_coulomb && !canonical_coulomb,
          want_exchange && !generated_exchange && !canonical_exchange,
          canonical_coulomb,
          canonical_exchange};
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
  /** Internal qualification switch, deliberately disabled for production.
   * Availability of the bounded through-f value lease does not qualify it as
   * a faster default. Set before enqueueing to compare retained sources without
   * changing capability, precision, derivative ownership or public API. */
  bool bounded_value_opt_in{false};
  /** Optional symmetry-canonical source for uncovered operators/classes. SPD
   * full-range generated owners retain priority; SR/LR also use this source.
   * Angular buckets
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
  /** Optional two-spin Cartesian range-output matrix for a joint full/range
   * value traversal. Allocated after existing owners, charged to the same
   * budget, and absent on policy/budget/allocation fallback. Presence freezes
   * admission; execution never rereads the environment or allocates storage.
   */
  double* canonical_range_exchange{};
  /** Borrowed test/profiler census: candidate quartets and radial evaluations.
   * Null in production. The observer owns storage and stream-ordered lifetime. */
  std::uint64_t* canonical_work_count{};
  /** Optional immutable, full-range symmetry-unique AO source values. Their
   * order is exactly the geometry-bound canonical bucket traversal. This
   * lease never changes range, screening or derivative fallback semantics. */
  double* resident_values{};
  std::size_t resident_value_count{};
  ~CudaDirectJkPlan();
};

/** Bounded value selection is separate from the owner's retained capability.
 * Other value joins must use this gate too; derivative consumers do not. */
inline bool direct_jk_bounded_value_enabled(const CudaDirectJkPlan& plan) noexcept {
  return plan.bounded_value_opt_in && plan.generated_exchange != nullptr &&
         plan.generated_exchange->shared != nullptr &&
         plan.generated_exchange->bounded_value_capability;
}

/** Complete generated/native coverage stays automatic. Incomplete coverage
 * uses the canonical/generic production source unless qualification explicitly
 * opts into HF's bounded higher-l value fallback. */
inline bool direct_jk_generated_full_range_value_available(const CudaDirectJkPlan& plan) noexcept {
  return plan.generated_exchange != nullptr && plan.generated_exchange->shared != nullptr &&
         (plan.generated_exchange->shared->value_capability ||
          direct_jk_bounded_value_enabled(plan));
}

/** A derivative-capable provider owner may still use its full-range value
 * route for a zero-order request. The owner's maximum derivative capability
 * does not select the SCF value schedule. */
inline bool direct_jk_generated_exchange_value_available(const CudaDirectJkPlan& plan,
                                                         const FockBuildSpec& spec) noexcept {
  if (spec.derivative_order != 0 || !spec.exchange.present || !plan.generated_exchange ||
      !plan.generated_exchange->shared)
    return false;
  if (spec.exchange.op == FockOperator::FullRange)
    return direct_jk_generated_full_range_value_available(plan);
  return direct_jk_bounded_value_enabled(plan) && spec.exchange.omega > 0.0 &&
         (spec.exchange.op == FockOperator::ShortRange ||
          spec.exchange.op == FockOperator::LongRange);
}

}  // namespace generativeqc::scf
