#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_wb97mv_grid.hpp"
#include "response/cuda_cpks.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}

namespace generativeqc::dft {

struct CudaKsWb97mvResponseOptions {
  CudaWb97mvGridOptions grid{};
  std::size_t max_device_bytes{512ULL << 20};
  /** Optional strict full/LR values for repeated induced actions. Values and
   * extra provider metadata remain charged to max_device_bytes; zero disables. */
  std::size_t max_resident_integral_bytes{64ULL << 20};
  bool defer_resident_integrals{false};
  /** Private dense Direct caller: audit the reference after optional row admission. */
  bool audit_after_resident_admission{false};
  std::size_t max_affine_term_visits{100'000'000};
  /** Padded products in generated shared-memory tiles, charged separately. */
  std::size_t max_matrix_terms{1'000'000'000};
  std::uint64_t max_direct_eri_evaluations{16'777'216};
  double reference_absolute_tolerance{3e-9};
  double reference_relative_tolerance{3e-9};
};

struct CudaKsWb97mvResponseWork {
  std::size_t device_bytes{}, affine_reference_terms{}, affine_action_terms{};
  std::size_t matrix_reference_terms{}, matrix_action_terms{};
  std::uint64_t reference_quartets{}, reference_eris{}, response_quartets{}, response_eris{};
  std::uint64_t actions{};
  std::size_t resident_integral_bytes{};
  std::uint64_t resident_source_evaluations{};
  std::size_t resident_pair_bytes{};
  std::uint64_t reference_pair_products{};
  std::uint64_t response_pair_product_bound{};
  std::size_t transition_workspace_bytes{};
  std::uint64_t direct_projection_products{};
  CudaWb97mvGridWork grid{}, separate_vv10{};
};

/** Authenticated fixed-geometry physical induced Fock for native WB97M-V KS.
 * Preparation verifies the complete model and reconstructs the actual accepted
 * F from H + exact/fitted full-range J/K + exact LR correction + XC/VV10. DF
 * consumes the existing compiler energy VJP and immutable raw/metric sources,
 * not occupied factors. Signed/indefinite coupled spin directions are legal.
 * Exact actions use a separate unscreened geometry-only Direct owner on the
 * producer stream; neither SCF scratch nor occupied projection leases change.
 *
 * Preparation may allocate; actions never allocate, compile, switch devices or
 * execute Python/CPU numerical work. Resource and semantic-work bounds are
 * explicit. The batch must outlive this nonmovable owner, with exclusive access.
 * Each action reauthenticates before execution/publication; failures preserve
 * output before publication. This is an induced-Fock component, not a nuclear
 * Hessian: moving-grid, integral and ECP source derivatives are not supplied.
 */
class CudaKsWb97mvResponse {
 public:
  CudaKsWb97mvResponse();
  ~CudaKsWb97mvResponse();
  CudaKsWb97mvResponse(const CudaKsWb97mvResponse&) = delete;
  CudaKsWb97mvResponse& operator=(const CudaKsWb97mvResponse&) = delete;
  CudaKsWb97mvResponse(CudaKsWb97mvResponse&&) = delete;
  CudaKsWb97mvResponse& operator=(CudaKsWb97mvResponse&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const CudaKsWb97mvResponseOptions& options, std::string& detail);
  cudaError_t apply(const double* direction, double* output, cudaStream_t stream) noexcept;
  /** Last action's static failure stage; null after a successful publication. */
  const char* failure_check() const noexcept;
  /** Read aggregate audited action counts once at the outer publication fence. */
  cudaError_t collect_work() noexcept;
  CudaKsWb97mvResponseWork work() const noexcept;
  /** Optional post-preparation admission, before the first action. Composite
   * owners call this only after all mandatory consumers have reserved storage.
   * The additional-byte cap includes extra provider metadata, not just values.
   * Dense repeated-RHS callers can also admit an AO pair layout. A
   * resource refusal keeps exact replay; numerical/device faults fail.
   * Only a dense-only owner may request certified small-value omission.
   * If the reference audit was deferred, this call must complete it before
   * the first action, even when the row layout is not admitted. */
  generativeqc_status retain_integrals(std::size_t maximum_additional_bytes, bool repeated_rhs,
                                       std::string& detail, bool dense_only = false);
  /** Spend only the remaining enclosing budget on fixed semilocal XC work. */
  cudaError_t retain_fixed_grid_points(std::size_t maximum_additional_bytes) noexcept;
  /** Bound the optional one-time RKS transition projection scratch and work. */
  cudaError_t retain_transition_builder(const response::CudaCpksShape& shape,
                                        std::size_t maximum_additional_bytes) noexcept;
  /** Check the finite arithmetic envelope of a complete pure RKS projection,
   * before the enclosing owner sets aside a CPKS matrix. */
  bool can_seed_fitted_operator(const response::CudaCpksShape& shape) const noexcept;
  bool can_seed_direct_operator(const response::CudaCpksShape& shape) const noexcept;
  bool has_transition_builder() const noexcept;
  /** Reserve optional batched metric RHS storage over the interleaved
   * [coordinate][core/overlap][AO][AO] lease of the dense Hessian. */
  cudaError_t retain_metric_projection(const double* frozen_core, std::size_t max_bytes) noexcept;
  const double* metric_projection(std::size_t column) const noexcept;
  static cudaError_t seed_fitted_operator(void* context, const response::CudaCpksShape& shape,
                                          const response::CudaCpksReference& reference,
                                          double* matrix, cudaStream_t stream) noexcept;
  static cudaError_t seed_direct_operator(void* context, const response::CudaCpksShape& shape,
                                          const response::CudaCpksReference& reference,
                                          double* matrix, cudaStream_t stream) noexcept;
  static cudaError_t seed_local_xc(void* context, const response::CudaCpksShape& shape,
                                   const response::CudaCpksReference& reference, double* matrix,
                                   cudaStream_t stream) noexcept;
  static cudaError_t induced_nonlocal_fock(void* context, const double* direction, double* output,
                                           const response::CudaCpksShape& shape,
                                           cudaStream_t stream) noexcept;
  /** Enqueue into CPKS's private output; audit failure poisons the panel.
   * CPKS owns stream completion and failure draining. Do not expose this panel
   * without checking it. Call collect_work at the outer endpoint boundary. */
  static cudaError_t induced_fock(void* context, const double* direction, double* output,
                                  const response::CudaCpksShape& shape,
                                  cudaStream_t stream) noexcept;

 private:
  cudaError_t apply_impl(const double* direction, double* output, cudaStream_t stream,
                         bool private_panel, bool local_xc = true) noexcept;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
