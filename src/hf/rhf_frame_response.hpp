#pragma once

#include <cstddef>
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
  // Generated matrix-map work and its host boundary only. These transfer
  // counters exclude integral-provider setup and nuclear derivative consumers.
  std::size_t contraction_terms{}, h2d_bytes{}, d2h_bytes{}, synchronizations{};
  std::size_t explicit_hessian_elements{};  // Always zero.
  bool matrix_blas{};
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
