#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "core/work_budget.hpp"
#include "integrals/ecp.hpp"
#include "integrals/ecp_gaussian.hpp"
#include "integrals/ecp_grid_cpu.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::integrals {
namespace {
using runtime::size_add;
using runtime::size_mul;

void validate_system(const core::System& system) {
  if (system.atoms.empty() || system.shells.empty() ||
      (system.basis_representation != core::kCartesianBasis &&
       system.basis_representation != core::kSphericalBasis))
    throw std::invalid_argument("ECP direction requires a normalized public AO system");
  for (const auto& atom : system.atoms) {
    if (atom.ionic_charge() < 0) throw std::invalid_argument("invalid ECP direction ionic charge");
    for (const auto value : atom.position)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite ECP direction center");
  }
  for (const auto& shell : system.shells) {
    if (shell.atom_index >= system.atoms.size() || shell.angular_momentum > 3 ||
        shell.primitives.empty() || shell.primitives.size() > INT32_MAX)
      throw std::invalid_argument("ECP direction supports normalized through-f shells");
    for (const auto& primitive : shell.primitives)
      if (!(primitive.exponent > 0) || !std::isfinite(primitive.exponent) ||
          !std::isfinite(primitive.coefficient))
        throw std::invalid_argument("invalid ECP direction normalized primitive");
  }
  for (const auto& term : system.ecp_terms)
    if (term.atom_index >= system.atoms.size() || !system.atoms[term.atom_index].ecp_core ||
        term.channel < -1 ||
        term.channel > static_cast<int>(generated::ecp_max_projector_angular) || term.power > 4 ||
        !(term.exponent > 0) || !std::isfinite(term.exponent) || !std::isfinite(term.coefficient))
      throw std::invalid_argument(
          "ECP direction residual operator is outside its validated domain");
  if (!molecule::ao_count(system)) throw std::invalid_argument("empty ECP direction AO layout");
}

void enforce(const EcpDirectionLimits& needed, const EcpDirectionLimits& limits) {
  if (needed.workspace_bytes > limits.workspace_bytes)
    throw std::length_error("ECP direction exceeds workspace_bytes");
  if (needed.gaussian_kernel_blocks > limits.gaussian_kernel_blocks ||
      needed.grid_radial_layers > limits.grid_radial_layers ||
      needed.grid_ao_primitive_points > limits.grid_ao_primitive_points ||
      needed.grid_contraction_terms > limits.grid_contraction_terms)
    throw core::WorkLimitError("ECP direction exceeds complete-source work limits");
}

struct Grid {
  std::vector<EcpRadialPoint> radii;
  std::vector<EcpSpherePoint> sphere;
  Grid(unsigned radial, unsigned polar) { ecp_quadrature(radial, polar, radii, sphere); }
};

void check(double coarse, double fine, bool derivative) {
  if (!generated::ecp_grid_pair_accepted(coarse, fine, derivative))
    throw std::runtime_error("ECP quadrature convergence gate failed for the nuclear direction");
}

EcpDirectionData grid_direction(const core::System& system, std::span<const double> direction,
                                const EcpDirectionPlan& plan,
                                const std::function<void()>& validate_current) {
  const auto aos = ecp_grid_cpu::orbitals(system);
  const auto count = aos.size(), matrix = size_mul(count, count);
  const std::array<Grid, 2> grids{
      Grid(generated::ecp_coarse_radial_points, generated::ecp_coarse_polar_points),
      Grid(generated::ecp_refined_radial_points, generated::ecp_refined_polar_points)};
  const auto points = std::max(grids[0].sphere.size(), grids[1].sphere.size());
  ecp_grid_cpu::Scratch scratch{
      std::vector<ecp_grid_cpu::Jet>(size_mul(count, points)),
      std::vector<ecp_grid_cpu::Jet>(size_mul(count, generated::ecp_projector_count))};
  EcpDirectionData result{count, std::vector<double>(matrix), std::vector<double>(matrix), {}};
  // A/B physical derivatives collect all ECP centers. Every other coordinate
  // belongs to exactly one current ECP center, so its gate can be checked now.
  std::vector<double> global(size_mul(28, matrix)), current(size_mul(12, matrix));
  runtime::CpuRetainedCapacity retained(runtime::vector_capacities(
      result.local, result.nonlocal, global, current, aos, scratch.values, scratch.projections,
      grids[0].radii, grids[0].sphere, grids[1].radii, grids[1].sphere));
  runtime::sample_cpu_capacity(0);
  const auto panel = [matrix](auto& values, std::size_t grid, std::size_t part,
                              std::size_t channels, std::size_t channel,
                              std::size_t item) -> double& {
    return values[((grid * 2 + part) * channels + channel) * matrix + item];
  };
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (!system.atoms[center].ecp_core) continue;
    validate_current();
    std::fill(current.begin(), current.end(), 0);
    for (std::size_t grid = 0; grid < grids.size(); ++grid)
      for (const auto& radial : grids[grid].radii) {
        validate_current();
        ecp_grid_cpu::layer(
            system, aos, radial, grids[grid].sphere, center, true, scratch,
            [&](std::size_t first, std::size_t second, const double (&parts)[2][10]) {
              const auto item = first * count + second;
              const auto first_atom = aos[first].shell->atom_index;
              const auto second_atom = aos[second].shell->atom_index;
              for (std::size_t part = 0; part < 2; ++part) {
                panel(global, grid, part, 7, 0, item) += parts[part][0];
                for (std::size_t axis = 0; axis < 3; ++axis) {
                  panel(global, grid, part, 7, 1 + axis, item) += parts[part][1 + axis];
                  panel(global, grid, part, 7, first_atom == second_atom ? 1 + axis : 4 + axis,
                        item) += parts[part][4 + axis];
                  if (center == first_atom)
                    panel(global, grid, part, 7, 1 + axis, item) += parts[part][7 + axis];
                  else if (center == second_atom)
                    panel(global, grid, part, 7, 4 + axis, item) += parts[part][7 + axis];
                  else
                    panel(current, grid, part, 3, axis, item) += parts[part][7 + axis];
                }
                if (grid == 1) {
                  auto& output = part == 0 ? result.local : result.nonlocal;
                  output[item] += generated::ecp_frozen_direction(
                      parts[part] + 1, direction.data() + 3 * first_atom,
                      direction.data() + 3 * second_atom, direction.data() + 3 * center);
                }
              }
            },
            &result.work);
      }
    for (std::size_t first = 0; first < count; ++first)
      for (std::size_t second = 0; second <= first; ++second) {
        if (center == aos[first].shell->atom_index || center == aos[second].shell->atom_index)
          continue;
        for (std::size_t part = 0; part < 2; ++part)
          for (std::size_t axis = 0; axis < 3; ++axis)
            check(panel(current, 0, part, 3, axis, first * count + second),
                  panel(current, 1, part, 3, axis, first * count + second), true);
      }
  }
  for (std::size_t first = 0; first < count; ++first)
    for (std::size_t second = 0; second <= first; ++second) {
      const auto item = first * count + second;
      for (std::size_t part = 0; part < 2; ++part) {
        for (std::size_t channel = 0; channel < 7; ++channel)
          check(panel(global, 0, part, 7, channel, item), panel(global, 1, part, 7, channel, item),
                channel != 0);
        auto& output = part == 0 ? result.local : result.nonlocal;
        if (!std::isfinite(output[item])) throw std::domain_error("nonfinite ECP direction");
        output[second * count + first] = output[item];
      }
    }
  enforce({plan.limits.workspace_bytes, 0, result.work.radial_layers,
           result.work.ao_primitive_points, result.work.contraction_terms},
          plan.limits);
  validate_current();
  return result;
}
}  // namespace

EcpDirectionPlan ecp_direction_plan(const core::System& system) {
  validate_system(system);
  const auto count = molecule::ao_count(system), matrix = size_mul(count, count);
  if (system.ecp_terms.empty())
    return {EcpDirectionBackend::Empty, {size_mul(2 * sizeof(double), matrix), 0, 0, 0, 0}};
  if (gaussian_ecp_supported(system)) return gaussian_ecp_direction_plan(system);
  const auto centers = std::count_if(system.atoms.begin(), system.atoms.end(),
                                     [](const auto& atom) { return atom.ecp_core != 0; });
  std::size_t primitive_terms{};
  for (const auto& shell : system.shells)
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation))
      primitive_terms =
          size_add(primitive_terms, size_mul(expansion.size(), shell.primitives.size()));
  const auto pairs =
      size_mul(count % 2 ? count : count / 2, count % 2 ? (count + 1) / 2 : count + 1);
  EcpDirectionPlan plan{EcpDirectionBackend::CheckedGrid, {}};
  for (const auto& [radial, polar] :
       {std::pair{generated::ecp_coarse_radial_points, generated::ecp_coarse_polar_points},
        std::pair{generated::ecp_refined_radial_points, generated::ecp_refined_polar_points}}) {
    const auto points = size_mul(2 * polar, polar), layers = size_mul(centers, radial);
    plan.limits.grid_radial_layers = size_add(plan.limits.grid_radial_layers, layers);
    plan.limits.grid_ao_primitive_points = size_add(
        plan.limits.grid_ao_primitive_points, size_mul(layers, size_mul(primitive_terms, points)));
    const auto terms =
        size_add(system.ecp_terms.size(),
                 size_add(size_mul(size_mul(count, points), generated::ecp_projector_count),
                          size_mul(pairs, size_add(points, generated::ecp_projector_count))));
    plan.limits.grid_contraction_terms =
        size_add(plan.limits.grid_contraction_terms, size_mul(layers, terms));
  }
  const auto polar = generated::ecp_refined_polar_points;
  const auto points = size_mul(2 * polar, polar);
  auto bytes = size_mul(42 * sizeof(double), matrix);
  bytes =
      size_add(bytes, size_mul(sizeof(ecp_grid_cpu::Jet),
                               size_mul(count, size_add(points, generated::ecp_projector_count))));
  bytes = size_add(bytes, size_mul(4 * sizeof(ecp_grid_cpu::AO), count));
  bytes = size_add(bytes, size_mul(8 * sizeof(EcpSpherePoint), points));
  bytes =
      size_add(bytes, size_mul(8 * sizeof(EcpRadialPoint), generated::ecp_refined_radial_points));
  plan.limits.workspace_bytes = size_add(bytes, 65536);
  return plan;
}

EcpDirectionData checked_ecp_direction(const core::System& system,
                                       std::span<const double> direction, EcpDirectionLimits limits,
                                       const std::function<void()>& validate_current) {
  if (!validate_current || !direction.data() ||
      direction.size() != size_mul(3, system.atoms.size()) ||
      !std::all_of(direction.begin(), direction.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("ECP direction requires a finite atom/xyz owner binding");
  validate_current();
  const auto plan = ecp_direction_plan(system);
  enforce(plan.limits, limits);
  if (plan.backend == EcpDirectionBackend::Gaussian)
    return gaussian_ecp_direction(system, direction, limits, validate_current);
  if (plan.backend == EcpDirectionBackend::CheckedGrid)
    return grid_direction(system, direction, plan, validate_current);
  const auto count = molecule::ao_count(system), size = size_mul(count, count);
  EcpDirectionData result{count, std::vector<double>(size), std::vector<double>(size), {}};
  runtime::sample_cpu_capacity(runtime::vector_capacities(result.local, result.nonlocal));
  validate_current();
  return result;
}

}  // namespace generativeqc::integrals
