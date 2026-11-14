#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::integrals {

// Residual Gaussian terms: c r^(power-2) exp(-exponent r^2).
// The Coulomb tail -Zeff/r is evaluated by nuclear attraction separately.
struct EcpData {
  std::size_t nbf{}, ncoord{};
  std::vector<double> local, nonlocal, local_derivative, nonlocal_derivative;
};

struct EcpSpherePoint {
  double x, y, z, weight;
  double harmonics[16];
};
struct EcpRadialPoint {
  double r, weight;
};

enum class EcpDirectionBackend { Empty, Gaussian, CheckedGrid };

/** Mandatory complete-source bounds; zero is not an implicit unlimited budget. */
struct EcpDirectionLimits {
  std::size_t workspace_bytes{};
  std::uint64_t gaussian_kernel_blocks{}, grid_radial_layers{}, grid_ao_primitive_points{};
  std::uint64_t grid_contraction_terms{};
};

struct EcpDirectionPlan {
  EcpDirectionBackend backend{};
  EcpDirectionLimits limits;
};

/** Actual source work; not total instruction counts, wall time or RSS. */
struct EcpDirectionWork {
  std::uint64_t kernel_blocks{}, radial_layers{}, ao_primitive_points{}, contraction_terms{};
  std::uint64_t shell_pairs{}, primitive_groups{}, radial_frontiers{}, cartesian_values{};
};

struct EcpDirectionData {
  std::size_t nbf{};
  std::vector<double> local, nonlocal;
  EcpDirectionWork work;
};

/** Query the actual analytic or checked-grid source, without evaluating ECP integrals. */
EcpDirectionPlan ecp_direction_plan(const core::System& system);

/** One residual-ECP nuclear matrix direction, not the ionic Coulomb tail.
 * Borrow normalized immutable system and atom/xyz direction; validate_current
 * must authenticate the actual owner/epoch and outlive this synchronous call.
 * No coordinate/AO derivative tensor is retained. The general two-grid path
 * checks every physical-coordinate derivative, even for a zero direction;
 * projection or cancellation must not hide an unqualified quadrature.
 * Both local/nonlocal outputs publish together by return only after final
 * validation. No JIT, displaced evaluation or reference-engine fallback.
 */
EcpDirectionData checked_ecp_direction(const core::System& system,
                                       std::span<const double> direction, EcpDirectionLimits limits,
                                       const std::function<void()>& validate_current);

/** Arbitrary ordered AO weights, not a stationary electronic Hessian.
 * Each residual part returns [value,right,gradient(3N),mixed(3N)]. No occupancy
 * factor or ionic Coulomb tail is included. Gradient/JVP sources are reused
 * across every physical left coordinate in one complete right-direction pass.
 */
struct EcpCurvatureData {
  std::size_t nbf{};
  std::array<std::vector<double>, 2> parts;
  EcpDirectionWork work;
};

/** Actual structural backend and mandatory complete-column resource limits. */
EcpDirectionPlan ecp_curvature_plan(const core::System& system);

/** Native analytic-specialized or independently checked general-grid column.
 * Immutable system is authenticated by validate_current, including content/epoch
 * identity, throughout the call. Ordered weights, an empty (frozen) or supplied
 * weights_right, and atom/xyz right leg are snapshotted and authenticated before
 * publication. The left nuclear partial holds weights fixed; its right action
 * includes the supplied weight response exactly once without extra source work.
 * In a KS consumer these are total D and total D_right, with no occupancy factor
 * added here. This owner neither solves nor certifies supplied SCF responses.
 * Both parts publish together by return only after final validation. The grid
 * checks raw physical primal, gradient and Hessian components, before weights
 * or direction projection, even for zero weights and rigid right directions.
 * Coincident A/B/C identities scatter before checking. Only finite slot/AO
 * gate panels and the physical output column are retained, never a molecular
 * coordinate-squared AO tensor. No displaced, JIT or reference fallback.
 * Workspace is a conservative owned-capacity bound, not an RSS/time limit.
 */
EcpCurvatureData checked_ecp_curvature(const core::System& system, std::span<const double> weights,
                                       std::span<const double> weights_right,
                                       std::span<const double> right, EcpDirectionLimits limits,
                                       const std::function<void()>& validate_current);
void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere);

EcpData ecp_integrals(const core::System& system, unsigned radial = 160, unsigned angular = 32,
                      bool derivatives = true);
EcpData checked_ecp_integrals(const core::System& system, bool derivatives);
/** Conservative numeric workspace for checked value/first-derivative consumers.
 * Includes both grids, public matrices, radial staging and host/device metadata;
 * callers reserve it in addition to their retained reference/correlation state. */
std::size_t ecp_workspace_capacity(const core::System& system);
void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative);

}  // namespace generativeqc::integrals
