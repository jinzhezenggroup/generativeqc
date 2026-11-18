#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "dft/host_xc_response.hpp"
#include "dft/ks_final_state.hpp"
#include "response/unrestricted.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {

struct HostKsResponseResources {
  std::size_t persistent_bytes{}, action_array_bound_bytes{}, gmres_workspace_bytes{};
  bool admitted{};
  std::size_t batch_workspace_bytes{};
};

struct HostKsResponseRequest {
  std::span<const double> rhs;
  std::span<double> solution;
  std::span<const double> initial_guess{};
};

struct HostKsResponseDiagnostic {
  response::GmresStatus status{response::GmresStatus::nonfinite_input};
  double residual_norm{}, relative_residual{};
  std::size_t iterations{}, operator_actions{}, preconditioner_actions{};
  std::size_t gmres_workspace_bytes{}, measured_gmres_peak_bytes{};
  [[nodiscard]] bool converged() const noexcept {
    return status == response::GmresStatus::converged ||
           status == response::GmresStatus::initial_residual;
  }
};

/** Native fixed-geometry CPKS, sharing the ordinary prepared CPU J/K source.
 * The verified state, normalized system, immutable Fock source and all borrowed
 * frame buffers must outlive this consumer. validate_current authenticates the
 * exact native KS identity/solve epoch, not merely a matching numerical frame.
 * The linked MethodIR binding must match the physical model, spin, SCF domain
 * and exact full-range exchange; no named-method solver or reference fallback
 * is introduced. RKS is the singlet sector of the shared two-spin operator.
 *
 * max_numeric_bytes covers this adapter/XC's owned panels, a conservative bound
 * for the shared orbital action's returned/transform arrays, and native GMRES.
 * Borrowed SCF/Fock/grid storage, internal provider/BLAS workspace, scalar stack,
 * metadata and allocator overhead remain their own owners, not process RSS.
 * A complete endpoint must admit those owners separately. This is not nuclear
 * RHS/metric/Pulay assembly, an integral derivative provider or a full Hessian.
 * All calls are synchronous and may not overlap. Failed solves never publish a
 * solution; failures release the invocation guard and permit a later valid call.
 * Output may alias the input vector but never a borrowed stationary-frame array.
 */
class HostKsResponse {
 public:
  using ValidateCurrent = std::function<void(const KsFinalStateIdentity&)>;

  HostKsResponse(const VerifiedKsFinalState& state, const core::System& system,
                 const scf::PreparedFockPlan& fock, HostGgaResponseShape grid_shape,
                 std::string_view method_identity, HostGgaTileReader reader,
                 ValidateCurrent validate_current, std::size_t max_numeric_bytes,
                 double minimum_orbital_gap = 1e-12);

  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] HostKsResponseResources resources(const response::GmresOptions& options,
                                                  std::size_t directions = 1) const;
  /** Additional caller-owned batching panels; one direction retains the scalar
   * arena. All simultaneous Krylov payload is priced separately by resources().
   */
  [[nodiscard]] std::size_t batch_workspace_bytes(std::size_t directions) const;
  [[nodiscard]] HostGgaResponseWork xc_work() const noexcept { return xc_->work(); }
  [[nodiscard]] std::size_t fock_actions() const noexcept { return fock_actions_; }
  /** Borrow the same authenticated frame used by the orbital action. */
  [[nodiscard]] core::ElectronicReferenceView reference() const;

  /** Optional exact point-partial bank, allocated before Fock actions and inside
   * both the existing owner cap and caller's extra allowance after GMRES. Returns
   * newly owned bytes; zero retains streamed evaluation or an existing bank.
   */
  std::size_t admit_xc_cache(const response::GmresOptions& options, std::size_t maximum_extra_bytes,
                             std::size_t directions = 1);

  void apply(std::span<const double> input, std::span<double> output);
  /** One to four independent same-frame actions, transactionally published.
   * Workspace is explicit, disjoint and included in the owner's resource cap.
   */
  void apply_batch(std::span<const response::LinearAction> actions, std::span<double> workspace);
  /** Physical AO Fock derivative for metric/solved density directions. RKS
   * takes one occupation-two total matrix; UKS takes both occupation-one spin
   * matrices. This reuses the exact same prepared J/K/XC action as apply(). */
  void apply_density(std::span<const double> density, std::span<double> output);
  HostKsResponseDiagnostic solve(std::span<const double> rhs, std::span<double> solution,
                                 const response::GmresOptions& options,
                                 std::span<const double> initial_guess = {});
  /** Independent GMRES states share only physical source execution. Every
   * requested true residual is freshly evaluated. No solution is published if
   * any request fails; the scalar solve remains the bounded resource fallback.
   */
  std::vector<HostKsResponseDiagnostic> solve_batch(std::span<const HostKsResponseRequest> requests,
                                                    const response::GmresOptions& options);

 private:
  void validate() const;
  void validate_output(std::span<double> output) const;
  void apply_current(std::span<const double> input, std::span<double> output);
  void apply_batch_current(std::span<const response::LinearAction> actions,
                           std::span<double> workspace);
  response::SpinFockAction::result_type induced(const std::array<std::vector<double>, 2>& density);
  response::SpinFockAction::result_type induced_current(
      const std::array<std::vector<double>, 2>& density);
  std::vector<response::SpinFockAction::result_type> induced_batch(
      std::span<const response::SpinFockAction::result_type> density);

  const VerifiedKsFinalState& state_;
  const core::System& system_;
  const scf::PreparedFockPlan& fock_;
  KsFinalStateIdentity identity_;
  core::ElectronicReferenceView reference_;
  ValidateCurrent validate_current_;
  HostGgaResponseShape shape_;
  std::size_t dimension_{}, matrix_size_{}, max_numeric_bytes_{};
  std::size_t persistent_bytes_{}, action_array_bound_bytes_{}, fock_actions_{};
  std::unique_ptr<double[]> panels_;
  std::span<double> density_, direction_, xc_output_, zero_, half_density_;
  std::span<double> diagonal_, staging_, singlet_input_, singlet_output_;
  std::span<double> active_batch_workspace_;
  std::unique_ptr<HostGgaXcResponse> xc_;
  std::optional<response::SpinResponseProblem> full_problem_;
  std::optional<response::LinearResponseProblem> problem_;
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};

}  // namespace generativeqc::dft
