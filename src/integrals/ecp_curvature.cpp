#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/work_budget.hpp"
#include "integrals/ecp.hpp"
#include "integrals/ecp_curvature_cpu.hpp"
#include "integrals/ecp_gaussian.hpp"
#include "integrals/ecp_grid_cpu.hpp"

namespace generativeqc::integrals {
namespace {
using runtime::size_add;
using runtime::size_mul;
constexpr std::size_t fields = 28;

void enforce(const EcpDirectionLimits& needed, const EcpDirectionLimits& limits) {
  if (needed.workspace_bytes > limits.workspace_bytes)
    throw std::length_error("ECP column exceeds workspace_bytes");
  if (needed.gaussian_kernel_blocks > limits.gaussian_kernel_blocks ||
      needed.grid_radial_layers > limits.grid_radial_layers ||
      needed.grid_ao_primitive_points > limits.grid_ao_primitive_points ||
      needed.grid_contraction_terms > limits.grid_contraction_terms)
    throw core::WorkLimitError("ECP column exceeds complete-source work limits");
}

struct Grid {
  std::vector<EcpRadialPoint> radii;
  std::vector<EcpSpherePoint> sphere;
  Grid(unsigned radial, unsigned polar) { ecp_quadrature(radial, polar, radii, sphere); }
};

/** Symmetric relative integral Hessian storage, before physical atom binding. */
std::array<std::size_t, fields> packed_indices() {
  std::array<std::size_t, fields> result{};
  for (std::size_t index = 0; index < 7; ++index) result[index] = index;
  std::size_t field = 7;
  for (std::size_t row = 0; row < 6; ++row)
    for (std::size_t column = row; column < 6; ++column) result[field++] = 7 + 6 * row + column;
  return result;
}

using Incidence = std::array<double, 6>;

Incidence incidence(std::size_t first, std::size_t second, std::size_t center, std::size_t atom,
                    std::size_t axis) {
  Incidence result{};
  result[axis] = static_cast<double>(first == atom) - static_cast<double>(center == atom);
  result[3 + axis] = static_cast<double>(second == atom) - static_cast<double>(center == atom);
  return result;
}

void check(double coarse, double fine, unsigned order) {
  if (!generated::ecp_grid_pair_accepted(coarse, fine, order != 0))
    throw std::runtime_error("ECP column raw physical order-" + std::to_string(order) +
                             " quadrature convergence gate failed");
}

EcpCurvatureData grid_column(const core::System& system, std::span<const double> weights,
                             std::span<const double> weights_right, std::span<const double> right,
                             const EcpDirectionPlan& plan,
                             const std::function<void()>& validate_current) {
  const auto aos = ecp_grid_cpu::orbitals(system);
  const auto count = aos.size(), matrix = size_mul(count, count);
  const std::array<Grid, 2> grids{
      Grid(generated::ecp_coarse_radial_points, generated::ecp_coarse_polar_points),
      Grid(generated::ecp_refined_radial_points, generated::ecp_refined_polar_points)};
  const auto points = std::max(grids[0].sphere.size(), grids[1].sphere.size());
  ecp_grid_cpu::SecondScratch scratch{
      std::vector<ecp_grid_cpu::SecondJet>(size_mul(count, points)),
      std::vector<ecp_grid_cpu::SecondJet>(size_mul(count, generated::ecp_projector_count))};
  std::vector<double> global(size_mul(4 * fields, matrix)), current(global.size());
  runtime::CpuRetainedCapacity retained(
      runtime::vector_capacities(global, current, scratch.values, scratch.projections, aos,
                                 grids[0].radii, grids[0].sphere, grids[1].radii, grids[1].sphere));
  ecp_curvature_cpu::Column column(system.atoms.size());
  EcpDirectionWork work;
  const auto packed = packed_indices();
  const auto panel = [&](auto& values, std::size_t grid, std::size_t part, std::size_t field,
                         std::size_t item) -> double& {
    return values[((grid * 2 + part) * fields + field) * matrix + item];
  };
  const auto expand = [&](std::size_t grid, std::size_t part, std::size_t item) {
    std::array<double, 43> jet{};
    for (std::size_t field = 0; field < fields; ++field)
      jet[packed[field]] = panel(current, grid, part, field, item);
    for (std::size_t row = 0; row < 6; ++row)
      for (std::size_t other = 0; other < row; ++other)
        jet[7 + 6 * row + other] = jet[7 + 6 * other + row];
    return jet;
  };
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (!system.atoms[center].ecp_core) continue;
    validate_current();
    std::fill(current.begin(), current.end(), 0);
    for (std::size_t grid = 0; grid < grids.size(); ++grid)
      for (const auto& radial : grids[grid].radii) {
        validate_current();
        ecp_grid_cpu::layer<true>(
            system, aos, radial, grids[grid].sphere, center, true, scratch,
            [&](std::size_t first, std::size_t second, const double (&parts)[2][43]) {
              const auto item = first * count + second;
              for (std::size_t part = 0; part < 2; ++part)
                for (std::size_t field = 0; field < fields; ++field)
                  panel(current, grid, part, field, item) += parts[part][packed[field]];
            },
            &work);
      }
    for (std::size_t first = 0; first < count; ++first)
      for (std::size_t second = 0; second <= first; ++second) {
        const auto item = first * count + second;
        const auto first_atom = aos[first].shell->atom_index;
        const auto second_atom = aos[second].shell->atom_index;
        std::array<Incidence, 9> rows{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          rows[axis] = incidence(first_atom, second_atom, center, first_atom, axis);
          if (first_atom != second_atom)
            rows[3 + axis] = incidence(first_atom, second_atom, center, second_atom, axis);
          if (center != first_atom && center != second_atom)
            rows[6 + axis] = incidence(first_atom, second_atom, center, center, axis);
        }
        for (std::size_t part = 0; part < 2; ++part) {
          const std::array<std::array<double, 43>, 2> jets{expand(0, part, item),
                                                           expand(1, part, item)};
          for (std::size_t grid = 0; grid < 2; ++grid) {
            const auto* jet = jets[grid].data();
            panel(global, grid, part, 0, item) += jet[0];
            for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
              panel(global, grid, part, 1 + coordinate, item) +=
                  generated::ecp_physical_gradient(jet, rows[coordinate].data());
            std::size_t field = 7;
            for (std::size_t row = 0; row < 6; ++row)
              for (std::size_t other = row; other < 6; ++other)
                panel(global, grid, part, field++, item) +=
                    generated::ecp_physical_hessian(jet, rows[row].data(), rows[other].data());
          }
          if (center != first_atom && center != second_atom)
            for (std::size_t axis = 6; axis < 9; ++axis) {
              check(generated::ecp_physical_gradient(jets[0].data(), rows[axis].data()),
                    generated::ecp_physical_gradient(jets[1].data(), rows[axis].data()), 1);
              for (const auto& row : rows)
                check(
                    generated::ecp_physical_hessian(jets[0].data(), row.data(), rows[axis].data()),
                    generated::ecp_physical_hessian(jets[1].data(), row.data(), rows[axis].data()),
                    2);
            }
          const auto weight =
              weights[item] + (first == second ? 0 : weights[second * count + first]);
          const auto moving_weight =
              weights_right[item] + (first == second ? 0 : weights_right[second * count + first]);
          column.add(part, weight, moving_weight, jets[1].data(), first_atom, second_atom, center,
                     right);
        }
      }
  }
  for (std::size_t first = 0; first < count; ++first)
    for (std::size_t second = 0; second <= first; ++second)
      for (std::size_t part = 0; part < 2; ++part)
        for (std::size_t field = 0; field < fields; ++field)
          check(panel(global, 0, part, field, first * count + second),
                panel(global, 1, part, field, first * count + second),
                field == 0  ? 0
                : field < 7 ? 1
                            : 2);
  enforce({plan.limits.workspace_bytes, 0, work.radial_layers, work.ao_primitive_points,
           work.contraction_terms},
          plan.limits);
  auto result = column.publish(count, work);
  validate_current();
  return result;
}
}  // namespace

EcpDirectionPlan ecp_curvature_plan(const core::System& system) {
  auto plan = ecp_direction_plan(system);
  const auto coordinates = size_mul(3, system.atoms.size());
  const auto outputs = size_add(2, size_mul(2, coordinates));
  const auto count = molecule::ao_count(system), matrix = size_mul(count, count);
  if (plan.backend == EcpDirectionBackend::Empty) {
    plan.limits.workspace_bytes = size_mul(2 * sizeof(double), outputs);
  } else if (plan.backend == EcpDirectionBackend::Gaussian) {
    plan = gaussian_ecp_curvature_plan(system);
  } else {
    const auto polar = generated::ecp_refined_polar_points;
    const auto points = size_mul(2 * polar, polar);
    plan.limits.workspace_bytes =
        size_add(plan.limits.workspace_bytes, size_mul((8 * fields - 42) * sizeof(double), matrix));
    plan.limits.workspace_bytes =
        size_add(plan.limits.workspace_bytes,
                 size_mul(6 * sizeof(double),
                          size_mul(count, size_add(points, generated::ecp_projector_count))));
    plan.limits.workspace_bytes =
        size_add(plan.limits.workspace_bytes, size_mul(6 * sizeof(double), outputs));
  }
  plan.limits.workspace_bytes =
      size_add(plan.limits.workspace_bytes,
               size_mul(sizeof(double), size_add(size_mul(2, matrix), coordinates)));
  return plan;
}

EcpCurvatureData checked_ecp_curvature(const core::System& system, std::span<const double> weights,
                                       std::span<const double> weights_right,
                                       std::span<const double> right, EcpDirectionLimits limits,
                                       const std::function<void()>& validate_current) {
  const auto count = molecule::ao_count(system);
  if (!validate_current || !weights.data() || weights.size() != size_mul(count, count) ||
      (!weights_right.empty() &&
       (!weights_right.data() || weights_right.size() != weights.size())) ||
      !right.data() || right.size() != size_mul(3, system.atoms.size()))
    throw std::invalid_argument("ECP column requires ordered AO weights and a right atom/xyz leg");
  for (const auto panel : {weights, weights_right, right})
    if (!std::all_of(panel.begin(), panel.end(), [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("nonfinite ECP column input");
  validate_current();
  const auto plan = ecp_curvature_plan(system);
  enforce(plan.limits, limits);
  std::vector<double> fixed(weights.begin(), weights.end()), moving(weights.size()),
      direction(right.begin(), right.end());
  if (!weights_right.empty()) std::copy(weights_right.begin(), weights_right.end(), moving.begin());
  runtime::CpuRetainedCapacity retained(runtime::vector_capacities(fixed, moving, direction));
  EcpCurvatureData result;
  if (plan.backend == EcpDirectionBackend::Gaussian) {
    result = gaussian_ecp_curvature(system, fixed, moving, direction, limits, validate_current);
  } else if (plan.backend == EcpDirectionBackend::CheckedGrid) {
    result = grid_column(system, fixed, moving, direction, plan, validate_current);
  } else {
    const auto outputs = size_add(2, size_mul(6, system.atoms.size()));
    result = {count, {std::vector<double>(outputs), std::vector<double>(outputs)}, {}};
    runtime::sample_cpu_capacity(runtime::vector_capacities(result.parts[0], result.parts[1]));
  }
  validate_current();
  if (!std::equal(fixed.begin(), fixed.end(), weights.begin()) ||
      !std::equal(direction.begin(), direction.end(), right.begin()) ||
      (!weights_right.empty() && !std::equal(moving.begin(), moving.end(), weights_right.begin())))
    throw std::invalid_argument("ECP column input changed during invocation");
  return result;
}

}  // namespace generativeqc::integrals
