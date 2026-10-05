#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "hf/reference.hpp"
#include "hf/rhf_frame_identity.hpp"
#include "hf/rhf_frame_recycle.hpp"
#include "response/native_gmres.hpp"

namespace generativeqc::core {
struct System;
}

namespace generativeqc::hf {

/** Owned, immutable same-frame DF preconditioner inputs. Transfer this owner
 * into response so resource/identity rejection can release it before falling
 * back to the ordinary diagonal solve. It contains no exact operator images.
 */
class RHFFrameDFPreconditioner {
 public:
  RHFFrameDFPreconditioner(const core::System& system, const PhysicalReference& reference,
                           std::size_t rank, std::uint64_t source_identity,
                           std::vector<double>&& diagonal, std::vector<double>&& low_rank)
      : identity_(system, reference),
        rank_(rank),
        source_identity_(source_identity),
        diagonal_(std::move(diagonal)),
        low_rank_(std::move(low_rank)) {}
  RHFFrameDFPreconditioner(const RHFFrameDFPreconditioner&) = delete;
  RHFFrameDFPreconditioner& operator=(const RHFFrameDFPreconditioner&) = delete;
  [[nodiscard]] bool matches(const core::System& system, const PhysicalReference& reference) const {
    return source_identity_ != 0 && identity_.matches(system, reference);
  }
  [[nodiscard]] std::span<const double> diagonal() const { return diagonal_; }
  [[nodiscard]] std::span<const double> low_rank() const { return low_rank_; }
  [[nodiscard]] std::size_t rank() const { return rank_; }
  [[nodiscard]] std::size_t storage_bytes() const {
    return posthf::checked_add(
        identity_.storage_bytes(),
        posthf::checked_mul(posthf::checked_add(diagonal_.capacity(), low_rank_.capacity()),
                            sizeof(double)));
  }

 private:
  RHFFrameIdentity identity_;
  std::size_t rank_{};
  std::uint64_t source_identity_{};
  std::vector<double> diagonal_, low_rank_;
};

struct RHFFrameDFPreconditionerPreparation {
  std::unique_ptr<RHFFrameDFPreconditioner> data;
  std::size_t numeric_capacity_bytes{}, contraction_terms{};
  double seconds{};
  std::string reason;
};

/** Build only a numerical accelerator from qualified same-frame factors.
 * maximum_bytes is the additional allowance after all borrowed factor/reference
 * payloads have been charged. Resource/numeric refusal returns empty data.
 */
RHFFrameDFPreconditionerPreparation prepare_rhf_frame_df_preconditioner(
    const core::System&, const PhysicalReference&, std::size_t naux, std::span<const double> boo,
    std::span<const double> bov, std::span<const double> bvv, std::uint64_t source_identity,
    std::size_t maximum_bytes);

/** Internal physical-RHF response controls. Scalar CUDA is a bounded fallback
 * for BLAS storage and an independent lowering audit, never a CPU fallback.
 * Disabling orbital relaxation exposes fixed-frame metric/nuclear weights for
 * validation; such results are not stationary molecular gradients.
 */
struct RHFFrameResponseOptions {
  std::size_t maximum_bytes{512ULL << 20};
  std::size_t caller_bytes{};
  bool matrix_blas{true};
  bool relax_orbitals{true};
  // Fixed geometry-only mask for the provisional Z solve. Its result must pass
  // the zero-screening physical residual; otherwise exact GMRES refines it.
  // Zero retains the original exact solve. Never screens final nuclear sources.
  double orbital_screening_tolerance{0.0};
  // Optional synchronized J/K timing and canonical integral census. Phase wall
  // times are always reported; J/K times are subsets, not additive phases.
  bool profile_jk{false};
  // Experimental canonical P:G'(D). Fewer passes can lose shell-level reuse,
  // so this consumer requires an explicit opt-in and a measured crossover.
  bool bilinear_derivative{false};
  // Preserve the admitted shell consumer and contract the cross term as
  // [E2'(D+P)-E2'(D-P)]/2. Both false retains legacy three-pass polarization.
  bool symmetric_polarization{true};
  // Opt-in while complete endpoint qualification is pending. The physical
  // exact J/K operator and all final acceptance gates remain unchanged.
  bool df_preconditioning{false};
  // Optional caller-owned, strictly same-operator solved-direction subspace.
  // Its complete numeric payload is charged to maximum_bytes by this owner.
  RHFFrameResponseRecycle* recycling{};
  response::GmresOptions gmres = [] {
    response::GmresOptions options;
    // Each exact action traverses J/K. Intermediate candidate checks are
    // amortized; convergence and the separate scalar audit still use exact J/K.
    options.true_residual_every = options.restart;
    return options;
  }();
};

struct RHFFrameResponseResult {
  // Electronic gradient only: add correlation-source and nuclear-repulsion
  // gradients before negating to obtain complete molecular forces.
  std::vector<double> gradient, hcore_weights, overlap_weights, fock_ao_weights;
  std::vector<double> stationarity, orbital_rhs;
  response::GmresResult orbital_response;
  double orbital_residual{}, maximum_stationarity{}, reference_residual{};
  // Residual qualification is local. No minimum Hessian eigenvalue or global
  // RHF stability certification is inferred from matrix-free convergence.
  bool global_stability_certified{false};
  std::size_t numeric_capacity_bytes{}, direct_device_bytes{}, owned_device_bytes{};
  std::size_t jk_actions{}, derivative_passes{}, orbital_actions{}, gemms{};
  std::size_t shell_derivative_passes{}, generic_derivative_passes{};
  // Generated matrix-map work and owner-managed transfers (including derivative
  // operand uploads). Provider-internal setup/execution traffic is excluded;
  // this is not a complete endpoint transfer ledger.
  std::size_t contraction_terms{}, h2d_bytes{}, d2h_bytes{}, synchronizations{};
  std::size_t explicit_hessian_elements{};  // Always zero.
  bool matrix_blas{};
  double setup_seconds{}, reference_audit_seconds{}, weights_seconds{}, solve_seconds{},
      independent_audit_seconds{}, one_electron_seconds{}, two_electron_seconds{};
  double jk_seconds{}, screened_jk_seconds{}, screened_residual{};
  double requested_screening{}, applied_screening{};
  std::size_t screened_jk_actions{}, jk_census_actions{}, exact_refinements{};
  std::size_t screened_iterations{}, screened_operator_actions{};
  std::uint64_t jk_quartet_visits{}, jk_eri_evaluations{};
  std::uint64_t derivative_quartet_visits{}, derivative_jet_evaluations{};
  bool bilinear_derivative_used{}, symmetric_polarization_used{}, derivative_census_measured{};
  bool jk_timing_measured{}, linear_screening_available{}, screened_converged{};
  std::string operator_hash;
  bool df_preconditioned{}, preconditioner_fallback{};
  std::size_t preconditioner_capacity_bytes{}, preconditioner_contraction_terms{};
  double preconditioner_setup_seconds{};
  std::string preconditioner_reason;
  bool recycled_guess{}, recycle_published{};
  std::size_t recycle_capacity_bytes{};
};

/** Compose matrix-sized TensorIR pullbacks, exact G(D)=J(D)-K(D)/2,
 * matrix-free physical Z response, AO Pulay and reference nuclear sources.
 * The reference must belong to this exact normalized all-electron system.
 * Canonical density/Fock consistency is audited before response. bar_fock_mo
 * uses full row-major Frobenius coordinates; bar_frame is dL/dC in the original
 * AO frame. Both must already include all correlation/triples sources.
 *
 * No dense MO ERI, (ov)^2 Hessian, CPU integral or CPU scientific response is
 * constructed. GMRES control remains on host. Same-space stationarity is
 * checked without same-space gaps. The final ov solve is qualified by fresh
 * scalar-CUDA action and full frame stationarity, not a stability certificate.
 * Complete numeric admission includes borrowed system/reference/seed payloads,
 * bounded host Krylov/output buffers, provider metadata and device storage;
 * caller_bytes adds other simultaneously live owners.
 */
RHFFrameResponseResult rhf_frame_response_cuda(
    const core::System& system, const PhysicalReference& reference,
    std::span<const double> bar_fock_mo, std::span<const double> bar_frame, int device,
    const RHFFrameResponseOptions& options = {},
    std::unique_ptr<RHFFrameDFPreconditioner> preconditioner = {});

}  // namespace generativeqc::hf
