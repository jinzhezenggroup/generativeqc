#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hf/reference.hpp"
#include "response/native_gmres.hpp"

namespace generativeqc::core {
struct System;
}

namespace generativeqc::hf {

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
  // Prefer canonical P:G'(D) over generic or bounded through-f fallback.
  // Specialized SPD keeps its lease. False retains three-pass polarization.
  bool bilinear_derivative{true};
  response::GmresOptions gmres{};
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
  // Generated matrix-map work and its host boundary only. These transfer
  // counters exclude integral-provider setup and nuclear derivative consumers.
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
  bool bilinear_derivative_used{}, derivative_census_measured{};
  bool jk_timing_measured{}, linear_screening_available{}, screened_converged{};
  std::string operator_hash;
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
RHFFrameResponseResult rhf_frame_response_cuda(const core::System& system,
                                               const PhysicalReference& reference,
                                               std::span<const double> bar_fock_mo,
                                               std::span<const double> bar_frame, int device,
                                               const RHFFrameResponseOptions& options = {});

}  // namespace generativeqc::hf
