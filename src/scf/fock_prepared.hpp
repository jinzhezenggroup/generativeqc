#ifndef GENERATIVEQC_SCF_FOCK_PREPARED_HPP
#define GENERATIVEQC_SCF_FOCK_PREPARED_HPP

#include <memory>

#include "runtime/cuda_provider.hpp"
#include "scf/cuda_fock_provider.hpp"
#include "scf/initial_guess/eigen_operation.hpp"

namespace generativeqc::scf {
namespace initial_guess {
class OverlapOrthogonalizer;
}

/** Backend variants frozen into the prepared source; mathematical identity
 * remains in ResolvedFockBuild::spec. These switches never authorize DF. */
struct FockExecutionVariant {
  runtime::CudaProviderKind cuda_provider{runtime::CudaProviderKind::None};
  unsigned one_electron_value_mapping{}, df_value_mapping{}, df_derivative_mapping{};
  bool one_electron_value_override{}, one_electron_value_capability_fallback{};
  DfPairStorage df_pair_storage{DfPairStorage::Dense};
  bool operator==(const FockExecutionVariant&) const = default;
};

/** Source preparation diagnostics, separate from the semantic request and
 * subsequent SCF convergence controls. Counts describe explicit native
 * buffers; CUDA modules, driver/library-private storage and recurrence stacks
 * are outside these counts. A source-backed DF plan retains bounded tiles.
 */
struct FockPreparationDiagnostic {
  ResolvedFockBuild strategy;
  std::size_t nbf{}, ncoord{}, device_bytes{}, device_budget_bytes{};
  /** Additional bounded DF response capacity, separate from retained values.
   * The response owner consumes this reservation synchronously; enclosing
   * compositions must retain it even when no response buffer is live yet. */
  std::size_t response_device_bytes{}, peak_device_bytes{};
  CudaDirectJkDiagnostic direct;
  FockExecutionVariant variant;
  CudaDensityFittingSourceDiagnostic fitted_source;
  std::vector<CudaDensityFittingMetricDiagnostic> fitted;
};

/** Optional capacity requested by a method with integral restricted occupations.
 * This reserves single-B packed U scratch, not a density/factor identity or
 * permission to borrow a completed projection. Generic callers leave it empty;
 * execution must still establish exact Cocc/D and owner/generation provenance.
 */
struct FockOccupiedProjectionReservation {
  std::size_t restricted_rank{};
  bool operator==(const FockOccupiedProjectionReservation&) const = default;
};

/** Immutable geometry/semantic owner for the common Fock provider views.
 * CPU numerical sources use the existing reference integrals; CUDA uses
 * the existing direct evaluator and source-backed DF tile generator. The
 * owner also supplies one-electron data to mean-field consumers, avoiding
 * duplicated preparation in SCF and fixed-density endpoints. It is not
 * concurrently reentrant because CUDA scratch and streams are shared.
 *
 * A positive device budget bounds explicit provider buffers and is never
 * enlarged. For DF-backed CUDA owners, zero resolves through the shared
 * workload/device-aware DF policy; exact-only CUDA owners keep their existing
 * allowance. CPU storage retains its reference representation and outer policy.
 */
class PreparedFockPlan {
 public:
  PreparedFockPlan(const core::System& orbital, const core::System* auxiliary,
                   ResolvedFockBuild strategy, int device_id = -1,
                   std::size_t device_budget_bytes = 0,
                   unsigned retained_direct_derivative_order = 0,
                   unsigned retained_fitted_derivative_order = 0,
                   FockOccupiedProjectionReservation projection_reservation = {});
  ~PreparedFockPlan();
  PreparedFockPlan(const PreparedFockPlan&) = delete;
  PreparedFockPlan& operator=(const PreparedFockPlan&) = delete;

  const ResolvedFockBuild& strategy() const noexcept;
  const core::System& system() const noexcept;
  const integrals::IntegralData& one_electron() const noexcept;
  /** The stage selects only explicit diagnostic provider controls; it never
   * changes the mathematical Fock operator or authorizes a reference retry. */
  enum class EigenUse { Setup, Iteration, Finalization };
  /** Borrow the qualified ordinary device provider from a CUDA fitted owner.
   * CPU/exact-only owners return an empty callback for the independent oracle.
   * The callback shares this owner's serialized lifetime and reserved scratch.
   * An iterative DIIS frame is not a verified physical final-state snapshot. */
  initial_guess::EigenOperation eigen_operation(EigenUse use) const;
  /** CUDA fitted SCF retains validated X on this immutable source. A containing
   * calculation may supply its own cache to survive value/force source replans;
   * its ordered basis and device must remain fixed. CPU/exact plans preserve
   * the reference solve and do not acquire persistent overlap storage. */
  std::vector<double> overlap_orthogonalizer(
      initial_guess::OverlapOrthogonalizer* external_cache = nullptr) const;
  /** CPU accounting borrows the actual DF owner, including its AO data. */
  const DensityFittingScfData* cpu_fitted_data() const noexcept;
  /** Host numerical bytes counted by the CPU SCF observer. Shared AO data is
   * charged once; metadata, CUDA buffers and opaque libraries have other owners.
   * This preserves the reference observer's capacity (not logical-size) contract.
   */
  std::size_t cpu_observation_capacity() const noexcept;
  const FockPreparationDiagnostic& diagnostic() const noexcept;
  /** Borrow the already selected exact CUDA source for an ordinary-stream
   * device consumer. Null means that this prepared strategy has no such
   * source. The PreparedFockPlan still owns selection, lifetime and resources. */
  CudaDirectJkPlan* cuda_direct_source() const noexcept;
  /** Borrow the fitted CUDA source on its own stream. Null for CPU/exact
   * sources; the immutable preparation and memory budget remain owned here. */
  CudaDensityFittingJkPlan* cuda_fitted_source() const noexcept;
  /** Exact normalized scientific identity, independent of execution budget. */
  bool matches_system(const core::System& system) const noexcept;
  DirectJkMatrices build(const std::vector<double>& density,
                         const std::vector<double>& beta = {}) const;
  std::vector<double> energy_derivative(const std::vector<double>& density,
                                        const std::vector<double>& beta = {}) const;
  FockEnergyDerivativeComponents energy_derivative_components(
      const std::vector<double>& density, const std::vector<double>& beta = {}) const;
  /** CUDA-DF component response using a method-owned exact final-K Cocc/U lease.
   * Exchange consumes the one-shot projection first; Coulomb then executes the
   * ordinary exact response. CPU/non-fitted owners reject this internal path. */
  FockEnergyDerivativeComponents energy_derivative_components_with_fitted_projection(
      const std::vector<double>& density, const std::vector<double>& beta,
      const CudaDfBorrowedFittedProjection& projection) const;
  /** CUDA-DF final-state response. Coulomb may consume the exact resident
   * restricted density while exchange optionally consumes the exact final-K
   * fitted projection. Host density remains the validation/scientific witness. */
  FockEnergyDerivativeComponents energy_derivative_components_with_cuda_df_state(
      const std::vector<double>& density, const std::vector<double>& beta,
      const CudaDfBorrowedResponseDensity* response_density,
      const CudaDfBorrowedFittedProjection* projection = nullptr) const;
  /** Execute a first derivative retained alongside a value-only prepared model.
   * The underlying provider is revalidated at derivative_order=1, preserving the
   * exact scientific approximation while keeping the SCF identity value-only. */
  std::vector<double> retained_energy_derivative(const std::vector<double>& density,
                                                 const std::vector<double>& beta = {}) const;
  /** Exact comparison of immutable source inputs and execution controls.
   * Convergence thresholds, DIIS history and warm density are deliberately
   * excluded: they do not alter the prepared mathematical operator. */
  bool matches(const core::System& orbital, const core::System* auxiliary,
               const ResolvedFockBuild& strategy, int device_id, std::size_t device_budget_bytes,
               unsigned minimum_direct_derivative_order = 0,
               unsigned minimum_fitted_derivative_order = 0,
               FockOccupiedProjectionReservation projection_reservation = {}) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::scf
#endif
