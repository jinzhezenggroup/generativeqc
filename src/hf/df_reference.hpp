#pragma once

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hf/rhf_frame_identity.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/density_fitting.hpp"
#include "scf/types.hpp"

namespace generativeqc::hf {

/** Immutable same-geometry RI-JK owner shared by SCF and molecular response.
 * Its metric cutoff defines the Hamiltonian, not an approximate preconditioner.
 * Signed response densities use dense linear J/K, never occupied-factor guesses.
 * No allocation/refusal permits replacing this model with four-center RHF.
 */
class DFReference {
 public:
  DFReference() = default;
  ~DFReference();
  DFReference(const DFReference&) = delete;
  DFReference& operator=(const DFReference&) = delete;

  scf::ScfResult solve(const core::System&, const core::System&, scf::ScfOptions, int,
                       const std::vector<double>* initial_density);
  [[nodiscard]] bool matches(const core::System&, const PhysicalReference&, int) const;
  [[nodiscard]] scf::CudaDensityFittingJkPlan* plan() const { return plan_; }
  [[nodiscard]] std::size_t storage_bytes() const { return storage_bytes_; }
  /** Contract the complete fixed-rank A/M derivative of E2(density).
   * maximum_bytes bounds additional host+device response capacity together. */
  std::vector<double> quadratic_gradient(std::span<const double>, std::size_t maximum_bytes);

 private:
  std::optional<core::System> orbital_, auxiliary_;
  std::vector<std::optional<scf::DensityFittingScfData>> data_;
  scf::CudaDensityFittingJkPlan* plan_{};
  std::optional<RHFFrameIdentity> identity_;
  /** Publication binding plus by-value identity; matching dimensions or a
   * coincidentally equal Fock cannot authorize a different auxiliary model. */
  std::shared_ptr<const PhysicalReference> reference_;
  int device_{-1};
  std::size_t storage_bytes_{};
};
}  // namespace generativeqc::hf
