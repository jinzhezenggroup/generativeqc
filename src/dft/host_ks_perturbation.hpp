#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "dft/host_ks_response.hpp"

namespace generativeqc::dft {

struct HostKsPerturbationOptions {
  std::size_t max_numeric_bytes{}, max_map_terms{};
  double symmetry_absolute_tolerance{1e-12}, symmetry_relative_tolerance{1e-10};
};

struct HostKsPerturbationResources {
  std::size_t owned_numeric_bytes{}, map_workspace_bytes{}, response_array_bound_bytes{};
  std::size_t gmres_workspace_bytes{}, scheduled_map_terms{};
  bool admitted{};
};

enum class HostKsPerturbationStatus {
  converged,
  workspace_limit,
  linear_solve_failed,
  physical_residual_failed
};

struct HostKsPerturbationDiagnostic {
  HostKsPerturbationStatus status{HostKsPerturbationStatus::linear_solve_failed};
  HostKsResponseDiagnostic response;
  double physical_residual_norm{}, relative_physical_residual{};
  [[nodiscard]] bool converged() const noexcept {
    return status == HostKsPerturbationStatus::converged;
  }
};

struct HostKsPerturbationWork {
  std::size_t directions_started{}, directions_completed{}, map_evaluations{};
  std::size_t scheduled_map_terms{};
};

struct HostKsPerturbationRequest {
  std::span<const double> overlap_direction, frozen_fock;
  std::span<double> density_direction, weighted_density_direction;
};

/** Native metric/RHS and D/W response around an authenticated KS frame.
 * overlap_direction is one symmetric AO S' matrix; frozen_fock contains one
 * symmetric fixed-density F' per physical spin (including ALL external source
 * terms). The borrowed HostKsResponse supplies the same coupled physical
 * induced Fock and GMRES action, and must outlive this synchronous consumer.
 * Compiler-owned TensorIR maps produce metric motion, the coupled RHS, total
 * density motion and the complete occupied-energy-matrix W tangent. A fresh
 * physical residual is checked before either output is published.
 *
 * The explicit budget composes these panels/map arena, response-owned arrays
 * and GMRES; borrowed SCF/grid/Fock/provider/BLAS storage, metadata/allocator
 * overhead and scalar stack remain separately owned. max_map_terms bounds the
 * materializing TensorIR schedule, not integral/grid/J/K or whole-endpoint work.
 * This does NOT construct nuclear integral/moving-grid/ECP F' or assemble a
 * Hessian. A nuclear endpoint must provide and authenticate every such source.
 *
 * Outputs use the response frame's spin/row/column layout and must not overlap
 * one another or any borrowed stationary frame array. They may alias external
 * input directions because publication is staged. Failure/nonconvergence never
 * publishes either output. Calls on one consumer may not overlap/reenter.
 */
class HostKsPerturbation {
 public:
  HostKsPerturbation(HostKsResponse& response, HostKsPerturbationOptions options);
  ~HostKsPerturbation();
  HostKsPerturbation(const HostKsPerturbation&) = delete;
  HostKsPerturbation& operator=(const HostKsPerturbation&) = delete;

  [[nodiscard]] HostKsPerturbationResources resources(const response::GmresOptions& options,
                                                      std::size_t directions = 1) const;
  [[nodiscard]] HostKsPerturbationWork work() const noexcept;
  HostKsPerturbationDiagnostic solve(std::span<const double> overlap_direction,
                                     std::span<const double> frozen_fock,
                                     std::span<double> density_direction,
                                     std::span<double> weighted_density_direction,
                                     const response::GmresOptions& options);
  /** One to four independent perturbations share only their iterative physical
   * actions. The existing compiler-owned metric/RHS/D/W/residual maps execute
   * per direction. Every map arena, retained panel and simultaneous Krylov state
   * is admitted before source work. Publication is atomic across the batch.
   */
  std::vector<HostKsPerturbationDiagnostic> solve_batch(
      std::span<const HostKsPerturbationRequest> requests, const response::GmresOptions& options);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
