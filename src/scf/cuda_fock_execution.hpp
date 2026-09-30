#ifndef GENERATIVEQC_SCF_CUDA_FOCK_EXECUTION_HPP
#define GENERATIVEQC_SCF_CUDA_FOCK_EXECUTION_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "scf/fock_prepared.hpp"

namespace generativeqc::scf {

/** Device-resident execution binding for one prepared Fock owner.
 *
 * Method consumers use this seam instead of borrowing a concrete Direct-J/K
 * or DF handle. The opaque source identity participates only in owner-local
 * replay invalidation; scientific and provider identity remain in the prepared
 * plan. Unsupported compositions return an empty binding rather than silently
 * selecting another provider.
 */
struct PreparedCudaFockBinding {
  int device_id{-1};
  cudaStream_t stream{};
  const void* source_identity{};
  std::size_t nbf{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && stream != nullptr && source_identity != nullptr && nbf != 0;
  }
};

/** Return the ordinary-stream device binding when the prepared plan can
 * execute its complete requested value-side Fock model through one resident
 * provider. The value seam covers exact full-range Coulomb plus exact full-/
 * short-/long-range exchange, and full-range density-fitted J/K when every
 * requested term shares the fitted owner. Qualified exact SR/LR value
 * execution may reuse the owner's conservative full-range Schwarz screening.
 * Derivatives use the separate capability binding below so value identity does
 * not become derivative identity. Mixed-provider compositions fail closed
 * rather than selecting or staging a different source.
 */
PreparedCudaFockBinding prepared_cuda_fock_binding(const PreparedFockPlan& plan) noexcept;

/** Device-resident occupied-factor execution binding for one full-range fitted
 * provider. This is a capability view only: the method consumer still owns the
 * proof that its canonical occupied coefficients reconstruct the supplied
 * density. No final-state or force-response lease is implied by this binding. */
struct PreparedCudaOccupiedFockBinding {
  int device_id{-1};
  cudaStream_t stream{};
  const void* source_identity{};
  std::size_t nbf{};
  bool unrestricted{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && stream != nullptr && source_identity != nullptr && nbf != 0;
  }
};

PreparedCudaOccupiedFockBinding prepared_cuda_occupied_fock_binding(
    const PreparedFockPlan& plan) noexcept;

/** Canonical occupied coefficients already resident on the prepared provider's
 * device. Coefficients are column-major AO x orbital matrices. Restricted
 * execution uses occupation two; unrestricted alpha/beta use occupation one. */
struct PreparedCudaOccupiedFockInput {
  const double* alpha_coefficients{};
  const double* beta_coefficients{};
  std::size_t alpha_rank{}, beta_rank{};
};

/** Enqueue full-range fitted J plus occupied-factor RI-K without staging the
 * canonical coefficients through host memory. The caller must prove that the
 * factors belong to the exact density pointers supplied here. Unsupported
 * provider/model compositions fail closed; this call never creates a final
 * projection lease by itself. */
generativeqc_status enqueue_prepared_cuda_occupied_fock(
    const PreparedFockPlan& plan, const double* density, const double* beta,
    std::size_t matrix_elements, const PreparedCudaOccupiedFockInput& occupied, double* coulomb,
    double* alpha_exchange, double* beta_exchange, int* numerical_error, std::string& detail);

/** Borrow the exact Direct source only when this prepared owner retained
 * first-derivative capability. The binding is an execution capability, not a
 * second scientific/provider owner; source_identity therefore matches the
 * ordinary value binding for exact plans.
 */
struct PreparedCudaDirectDerivativeBinding {
  int device_id{-1};
  cudaStream_t stream{};
  const void* source_identity{};
  std::size_t nbf{}, coordinates_per_item{}, retained_device_bytes{};
  unsigned maximum_derivative_order{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && stream != nullptr && source_identity != nullptr && nbf != 0 &&
           coordinates_per_item != 0 && maximum_derivative_order >= 1;
  }
};

PreparedCudaDirectDerivativeBinding prepared_cuda_direct_derivative_binding(
    const PreparedFockPlan& plan) noexcept;

/** Execute the fused fixed-density J'/SR-K'/LR-K' derivative through the
 * retained Direct owner. Scientific coefficients and omega are derived from
 * the prepared primary strategy plus a validated long-range correction; callers
 * cannot reinterpret the borrowed source with arbitrary operator parameters.
 */
generativeqc_status execute_prepared_cuda_direct_rsh_energy_derivatives(
    const PreparedFockPlan& plan, const ResolvedFockBuild& long_range_correction,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& derivatives, std::string& detail);

/** Same scientific derivative as above, but borrow a validated native KS
 * density that is already resident on the prepared Direct owner's device. */
generativeqc_status execute_prepared_cuda_direct_rsh_energy_derivatives_device(
    const PreparedFockPlan& plan, const ResolvedFockBuild& long_range_correction,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail);

/** Execute the prepared primary model's full-range J'/K' through the retained
 * shell derivative lease. Output is source-major [J,K]. Absent exchange
 * produces an all-zero K block while preserving the two-source shape. */
generativeqc_status execute_prepared_cuda_direct_shell_full_range_derivatives_device(
    const PreparedFockPlan& plan, const double* density, const double* beta,
    std::size_t matrix_elements, std::vector<double>& derivatives, std::string& detail);

/** Enqueue the prepared plan's complete raw J/K request on caller-owned device
 * buffers. Output pointers follow FockBuildSpec presence/spin semantics.
 * mixed_coulomb changes only the qualified exact-Coulomb recurrence precision;
 * fitted execution remains strict FP64. It never changes K, scientific
 * coefficients, screening, or provider selection. When supplied for a mixed
 * Direct execution, mixed_coulomb_work_count is a caller-owned device counter
 * for actually evaluated mixed Coulomb AO-ERI recurrences. Supplying it for a
 * strict execution fails closed.
 */
generativeqc_status enqueue_prepared_cuda_fock(const PreparedFockPlan& plan, const double* density,
                                               const double* beta, std::size_t matrix_elements,
                                               double* coulomb, double* alpha_exchange,
                                               double* beta_exchange, int* numerical_error,
                                               bool mixed_coulomb, std::string& detail,
                                               std::uint64_t* mixed_coulomb_work_count = nullptr);

/** Enqueue a separately resolved long-range exact-exchange correction through
 * the same resident direct-J/K source as the primary prepared owner. The
 * correction must preserve spin and screening identity and contain no Coulomb
 * term. This keeps RSH composition on one stream/source without creating a
 * second CUDA provider owner.
 */
generativeqc_status enqueue_prepared_cuda_exchange_correction(
    const PreparedFockPlan& plan, const ResolvedFockBuild& correction, const double* density,
    const double* beta, std::size_t matrix_elements, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, std::string& detail);

}  // namespace generativeqc::scf

#endif
