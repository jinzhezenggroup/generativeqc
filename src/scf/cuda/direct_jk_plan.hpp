#pragma once

#include <cuda_runtime_api.h>

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
  ~CudaDirectJkPlan();
};

/** A derivative-capable provider owner may still use generated exchange for a
 * zero-order value request. The owner's maximum derivative capability must not
 * select the SCF value schedule.
 */
inline bool direct_jk_generated_exchange_value_available(const CudaDirectJkPlan& plan,
                                                         const FockBuildSpec& spec) noexcept {
  return plan.generated_exchange != nullptr && spec.derivative_order == 0 &&
         spec.exchange.present && spec.exchange.op == FockOperator::FullRange;
}

}  // namespace generativeqc::scf
