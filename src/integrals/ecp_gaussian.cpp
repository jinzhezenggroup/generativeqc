#include "integrals/ecp_gaussian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "core/work_budget.hpp"
#include "generated_ecp_ao.cuh"
#include "generated_ecp_gaussian.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::integrals {
namespace {
using Powers = generated::EcpGaussianPowers;
constexpr unsigned pair_capacity = 100, jet_capacity = 43;

struct Sum {
  double value{}, correction{};
  void add(double term) {
    const double adjusted = term - correction;
    const double next = value + adjusted;
    correction = (next - value) - adjusted;
    value = next;
  }
};

struct Group {
  unsigned center{};
  double exponent{};
  std::array<Sum, 4> coefficients;
};

struct Component {
  Powers powers{};
  unsigned index{};
  double coefficient{};
};

struct Shell {
  const core::Shell* source{};
  std::size_t ao_begin{};
  std::vector<std::vector<Component>> expansions;
};

unsigned components(unsigned angular) { return (angular + 1) * (angular + 2) / 2; }

unsigned component_index(const Powers& powers, unsigned offset) {
  const unsigned angular = powers[offset] + powers[offset + 1] + powers[offset + 2];
  unsigned index = 0;
  for (int first = static_cast<int>(angular); first >= 0; --first)
    for (int second = static_cast<int>(angular) - first; second >= 0; --second, ++index)
      if (powers[offset] == static_cast<unsigned>(first) &&
          powers[offset + 1] == static_cast<unsigned>(second))
        return index;
  throw std::invalid_argument("invalid Gaussian ECP Cartesian component");
}

std::vector<Powers> cartesian_pairs(unsigned first, unsigned second) {
  std::vector<Powers> result;
  result.reserve(components(first) * components(second));
  for (int first_x = static_cast<int>(first); first_x >= 0; --first_x)
    for (int first_y = static_cast<int>(first) - first_x; first_y >= 0; --first_y)
      for (int second_x = static_cast<int>(second); second_x >= 0; --second_x)
        for (int second_y = static_cast<int>(second) - second_x; second_y >= 0; --second_y)
          result.push_back({static_cast<unsigned>(first_x), static_cast<unsigned>(first_y),
                            first - first_x - first_y, static_cast<unsigned>(second_x),
                            static_cast<unsigned>(second_y), second - second_x - second_y});
  return result;
}

std::vector<Group> groups_for(const core::System& system) {
  std::vector<Group> groups;
  for (const auto& term : system.ecp_terms) {
    auto found = std::find_if(groups.begin(), groups.end(), [&](const Group& group) {
      return group.center == term.atom_index && group.exponent == term.exponent;
    });
    if (found == groups.end()) {
      groups.push_back({term.atom_index, term.exponent, {}});
      found = groups.end() - 1;
    }
    found->coefficients[term.channel + 1].add(term.coefficient);
  }
  return groups;
}

std::vector<Shell> shells_for(const core::System& system) {
  std::vector<Shell> shells;
  std::size_t offset = 0;
  for (const auto& source : system.shells) {
    Shell shell{&source, offset, {}};
    for (const auto& expansion :
         molecule::ao_expansions(source.angular_momentum, system.basis_representation)) {
      std::vector<Component> terms;
      for (const auto& term : expansion) {
        Powers powers{term.component[0], term.component[1], term.component[2], 0, 0, 0};
        terms.push_back({powers, component_index(powers, 0),
                         generated::ecp_component_coefficient(powers[0], powers[1], powers[2],
                                                              term.coefficient)});
      }
      shell.expansions.push_back(std::move(terms));
    }
    offset = runtime::size_add(offset, shell.expansions.size());
    shells.push_back(std::move(shell));
  }
  return shells;
}

std::size_t output_elements(const core::System& system, unsigned order, bool weighted) {
  const auto coordinates = runtime::size_mul(3, system.atoms.size());
  if (weighted)
    return runtime::size_mul(
        2, runtime::size_add(
               1, runtime::size_add(coordinates, runtime::size_mul(coordinates, coordinates))));
  const auto count = molecule::ao_count(system);
  return runtime::size_mul(2, runtime::size_mul(runtime::size_mul(count, count),
                                                order ? runtime::size_add(1, coordinates) : 1));
}

std::size_t workspace_capacity(const core::System& system, unsigned order, bool weighted) {
  auto bytes = runtime::size_mul(3 * sizeof(double), output_elements(system, order, weighted));
  bytes = runtime::size_add(
      bytes, runtime::size_mul(sizeof(double), generated::ecp_gaussian_scratch_doubles));
  bytes = runtime::size_add(bytes, 144 * (sizeof(std::vector<double>) + 225 * sizeof(double)));
  bytes = runtime::size_add(bytes, 2 * jet_capacity * pair_capacity * sizeof(Sum));
  bytes = runtime::size_add(bytes, pair_capacity * sizeof(Powers));
  bytes = runtime::size_add(bytes, runtime::size_mul(2 * sizeof(Group), system.ecp_terms.size()));
  bytes = runtime::size_add(bytes, runtime::size_mul(4096, system.shells.size()));
  return bytes;
}

EcpGaussianWork preflight(const core::System& system, unsigned order, bool weighted,
                          std::vector<Group>& groups, EcpGaussianLimits limits) {
  EcpGaussianWork work;
  work.workspace_bound_bytes = workspace_capacity(system, order, weighted);
  if (work.workspace_bound_bytes > limits.workspace_bytes)
    throw std::length_error("Gaussian ECP endpoint exceeds workspace_bytes");
  groups = groups_for(system);
  std::size_t channels = 0;
  for (const auto& group : groups)
    for (const auto& coefficient : group.coefficients)
      if (coefficient.value != 0) channels = runtime::size_add(channels, 1);
  const unsigned shifts = order == 2 ? 13 : order == 1 ? 5 : 1;
  for (std::size_t first = 0; first < system.shells.size(); ++first)
    for (std::size_t second = 0; second <= first; ++second) {
      const auto primitives = runtime::size_mul(system.shells[first].primitives.size(),
                                                system.shells[second].primitives.size());
      work.kernel_block_bound =
          runtime::size_add(work.kernel_block_bound,
                            runtime::size_mul(runtime::size_mul(primitives, channels), shifts));
    }
  if (work.kernel_block_bound > limits.kernel_blocks)
    throw core::WorkLimitError("Gaussian ECP endpoint exceeds kernel_blocks");
  return work;
}

struct PrimitiveBlocks {
  double alpha, beta, zeta;
  const double* centers;
  double radial[11]{};
  bool scaled{};
  EcpGaussianWork& work;
  std::vector<double>& scratch;
  std::array<std::vector<double>, 144> cache;

  PrimitiveBlocks(double first, double second, double exponent, const double* relative,
                  unsigned maximum_order, bool needs_radial, EcpGaussianWork& observed,
                  std::vector<double>& workspace)
      : alpha(first),
        beta(second),
        zeta(exponent),
        centers(relative),
        work(observed),
        scratch(workspace) {
    if (!std::isfinite(alpha + beta + zeta))
      throw std::invalid_argument("Gaussian ECP exponent sum is not finite");
    if (needs_radial) {
      scaled = generated::ecp_gaussian_frontier(alpha, beta, zeta, centers, maximum_order, radial);
      ++work.radial_frontiers;
    }
    ++work.primitive_groups;
  }

  double value(const Powers& powers, int channel) {
    const unsigned first = powers[0] + powers[1] + powers[2];
    const unsigned second = powers[3] + powers[4] + powers[5];
    if (first > 5 || second > 5 || first + second > 8)
      throw std::logic_error("Gaussian ECP derivative requested an unavailable shifted shell");
    auto& values = cache[36 * (channel + 1) + 6 * first + second];
    if (values.empty()) {
      const bool swap = first < second;
      double relative[6];
      std::copy_n(centers + (swap ? 3 : 0), 3, relative);
      std::copy_n(centers + (swap ? 0 : 3), 3, relative + 3);
      const auto kernel = generated::ecp_gaussian_kernel(
          std::max(first, second), std::min(first, second), channel, scaled && channel != -1);
      if (!kernel) throw std::logic_error("missing generated Gaussian ECP block");
      values.resize(components(first) * components(second));
      kernel(swap ? beta : alpha, swap ? alpha : beta, zeta, relative, radial, scratch.data(),
             values.data());
      if (!std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("nonfinite generated Gaussian ECP integral");
      ++work.kernel_blocks;
      work.cartesian_values += values.size();
      std::size_t bytes = sizeof(cache);
      for (const auto& buffer : cache)
        bytes = runtime::size_add(bytes, runtime::size_mul(sizeof(double), buffer.capacity()));
      runtime::sample_cpu_capacity(bytes);
    }
    const unsigned row = component_index(powers, 0), column = component_index(powers, 3);
    return values[first < second ? column * components(first) + row
                                 : row * components(second) + column];
  }
};

using Block = std::array<Sum, jet_capacity * pair_capacity>;
using Jet = std::array<double, jet_capacity>;

std::array<Jet, 2> transform_pair(const Shell& first, const Shell& second, unsigned row,
                                  unsigned column, unsigned channels,
                                  const std::array<Block, 2>& blocks) {
  std::array<Jet, 2> result{};
  const auto pair_size =
      components(first.source->angular_momentum) * components(second.source->angular_momentum);
  for (unsigned part = 0; part < 2; ++part)
    for (unsigned channel = 0; channel < channels; ++channel) {
      Sum sum;
      for (const auto& left : first.expansions[row])
        for (const auto& right : second.expansions[column]) {
          const auto pair = left.index * components(second.source->angular_momentum) + right.index;
          sum.add(left.coefficient * right.coefficient *
                  blocks[part][channel * pair_size + pair].value);
        }
      if (!std::isfinite(sum.value))
        throw std::runtime_error("nonfinite contracted Gaussian ECP jet");
      result[part][channel] = sum.value;
    }
  return result;
}

template <class Consumer>
void evaluate(const core::System& system, unsigned order, const std::vector<Group>& groups,
              EcpGaussianWork& work, Consumer&& consume) {
  const auto shells = shells_for(system);
  std::vector<double> scratch(generated::ecp_gaussian_scratch_doubles);
  const unsigned channels = order == 2 ? 43 : order == 1 ? 7 : 1;
  auto metadata_bytes = runtime::vector_capacities(shells, scratch, groups);
  for (const auto& shell : shells) {
    metadata_bytes =
        runtime::size_add(metadata_bytes, runtime::vector_capacities(shell.expansions));
    for (const auto& expansion : shell.expansions)
      metadata_bytes = runtime::size_add(metadata_bytes, runtime::vector_capacities(expansion));
  }
  runtime::CpuRetainedCapacity metadata(metadata_bytes);
  runtime::sample_cpu_capacity(0);
  for (std::size_t first = 0; first < shells.size(); ++first)
    for (std::size_t second = 0; second <= first; ++second) {
      const auto& left = shells[first];
      const auto& right = shells[second];
      const auto powers =
          cartesian_pairs(left.source->angular_momentum, right.source->angular_momentum);
      runtime::CpuRetainedCapacity pair_storage(runtime::vector_capacities(powers));
      ++work.shell_pairs;
      // A common center/exponent group shares its radial frontier across channels.
      // Reduce primitives before AO pullback; the consumer immediately scatters
      // raw first derivatives or contracts second derivatives with final weights.
      for (const auto& group : groups) {
        std::array<Block, 2> blocks{};
        runtime::CpuRetainedCapacity block_storage(sizeof(blocks));
        double centers[6];
        for (unsigned axis = 0; axis < 3; ++axis) {
          centers[axis] = system.atoms[left.source->atom_index].position[axis] -
                          system.atoms[group.center].position[axis];
          centers[3 + axis] = system.atoms[right.source->atom_index].position[axis] -
                              system.atoms[group.center].position[axis];
          if (!std::isfinite(centers[axis]) || !std::isfinite(centers[3 + axis]))
            throw std::invalid_argument("nonfinite Gaussian ECP relative center");
        }
        const bool radial = std::any_of(group.coefficients.begin() + 1, group.coefficients.end(),
                                        [](const Sum& sum) { return sum.value != 0; });
        for (const auto& first_primitive : left.source->primitives)
          for (const auto& second_primitive : right.source->primitives) {
            PrimitiveBlocks primitive(
                first_primitive.exponent, second_primitive.exponent, group.exponent, centers,
                left.source->angular_momentum + right.source->angular_momentum + order + 2, radial,
                work, scratch);
            for (unsigned channel = 0; channel < 4; ++channel) {
              if (group.coefficients[channel].value == 0) continue;
              const double weight = first_primitive.coefficient * second_primitive.coefficient *
                                    group.coefficients[channel].value;
              auto& destination = blocks[channel ? 1 : 0];
              for (std::size_t pair = 0; pair < powers.size(); ++pair) {
                const auto jet = generated::ecp_gaussian_component_jet(
                    powers[pair], order, first_primitive.exponent, second_primitive.exponent,
                    [&](const Powers& item) {
                      return primitive.value(item, static_cast<int>(channel) - 1);
                    });
                for (unsigned derivative = 0; derivative < channels; ++derivative)
                  destination[derivative * powers.size() + pair].add(weight * jet[derivative]);
              }
            }
          }
        for (unsigned row = 0; row < left.expansions.size(); ++row)
          for (unsigned column = 0; column < right.expansions.size(); ++column) {
            if (right.ao_begin + column > left.ao_begin + row) continue;
            consume(left, right, group.center, left.ao_begin + row, right.ao_begin + column,
                    transform_pair(left, right, row, column, channels, blocks));
          }
      }
    }
  if (work.kernel_blocks > work.kernel_block_bound)
    throw std::logic_error("Gaussian ECP exceeded its preflight work bound");
}

std::vector<double> publish(const std::vector<Sum>& values) {
  std::vector<double> result;
  result.reserve(values.size());
  for (const auto& sum : values) {
    if (!std::isfinite(sum.value))
      throw std::runtime_error("nonfinite Gaussian ECP endpoint output");
    result.push_back(sum.value);
  }
  return result;
}

template <class Scatter>
void scatter_axis(const Shell& first, const Shell& second, unsigned center, unsigned axis,
                  Scatter&& scatter) {
  const auto atom = axis < 3 ? first.source->atom_index : second.source->atom_index;
  scatter(3 * atom + axis % 3, 1.0);
  scatter(3 * center + axis % 3, -1.0);
}
}  // namespace

bool gaussian_ecp_supported(const core::System& system) {
  if (system.atoms.empty() || system.shells.empty() || system.ecp_terms.empty()) return false;
  if (system.basis_representation != core::kCartesianBasis &&
      system.basis_representation != core::kSphericalBasis)
    return false;
  for (const auto& atom : system.atoms)
    for (double coordinate : atom.position)
      if (!std::isfinite(coordinate)) return false;
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3 || shell.atom_index >= system.atoms.size() ||
        shell.primitives.empty())
      return false;
    for (const auto& primitive : shell.primitives)
      if (!(primitive.exponent > 0) || !std::isfinite(primitive.exponent) ||
          !std::isfinite(primitive.coefficient))
        return false;
  }
  for (const auto& term : system.ecp_terms)
    if (term.atom_index >= system.atoms.size() || !system.atoms[term.atom_index].ecp_core ||
        term.power != 2 || term.channel < -1 || term.channel > 2 || !(term.exponent > 0) ||
        !std::isfinite(term.exponent) || !std::isfinite(term.coefficient))
      return false;
  return true;
}

std::size_t gaussian_ecp_workspace_capacity(const core::System& system, bool derivatives) {
  return workspace_capacity(system, derivatives ? 1 : 0, false);
}

EcpData gaussian_ecp_integrals(const core::System& system, bool derivatives,
                               EcpGaussianWork* observed, EcpGaussianLimits limits) {
  if (!gaussian_ecp_supported(system))
    throw std::invalid_argument("operator outside Gaussian ECP specialization");
  std::vector<Group> groups;
  auto work = preflight(system, derivatives ? 1 : 0, false, groups, limits);
  const auto count = molecule::ao_count(system), size = runtime::size_mul(count, count);
  const auto coordinates = derivatives ? runtime::size_mul(3, system.atoms.size()) : 0;
  std::array<std::vector<Sum>, 2> values{std::vector<Sum>(size), std::vector<Sum>(size)};
  const auto derivative_size = runtime::size_mul(coordinates, size);
  std::array<std::vector<Sum>, 2> gradients{std::vector<Sum>(derivative_size),
                                            std::vector<Sum>(derivative_size)};
  runtime::CpuRetainedCapacity retained(
      runtime::vector_capacities(values[0], values[1], gradients[0], gradients[1]));
  evaluate(
      system, derivatives ? 1 : 0, groups, work,
      [&](const Shell& first, const Shell& second, unsigned center, std::size_t row,
          std::size_t column, const std::array<Jet, 2>& jets) {
        for (unsigned transpose = 0; transpose < (row == column ? 1U : 2U); ++transpose) {
          const auto item = transpose ? column * count + row : row * count + column;
          for (unsigned part = 0; part < 2; ++part) {
            values[part][item].add(jets[part][0]);
            if (derivatives)
              for (unsigned axis = 0; axis < 6; ++axis)
                scatter_axis(first, second, center, axis, [&](std::size_t coordinate, double sign) {
                  gradients[part][coordinate * size + item].add(sign * jets[part][1 + axis]);
                });
          }
        }
      });
  EcpData result{count,
                 coordinates,
                 publish(values[0]),
                 publish(values[1]),
                 publish(gradients[0]),
                 publish(gradients[1])};
  runtime::sample_cpu_capacity(runtime::vector_capacities(
      result.local, result.nonlocal, result.local_derivative, result.nonlocal_derivative));
  if (observed) *observed = work;
  return result;
}

EcpWeightedSecondOrder gaussian_ecp_weighted_second_order(const core::System& system,
                                                          std::span<const double> weights,
                                                          EcpGaussianLimits limits) {
  if (!gaussian_ecp_supported(system))
    throw std::invalid_argument("operator outside Gaussian ECP specialization");
  const auto count = molecule::ao_count(system), size = runtime::size_mul(count, count);
  if (weights.size() != size || !std::all_of(weights.begin(), weights.end(),
                                             [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("Gaussian ECP weights require a finite ordered AO matrix");
  std::vector<Group> groups;
  auto work = preflight(system, 2, true, groups, limits);
  const auto coordinates = runtime::size_mul(3, system.atoms.size());
  std::array<Sum, 2> values;
  std::array<std::vector<Sum>, 2> gradients{std::vector<Sum>(coordinates),
                                            std::vector<Sum>(coordinates)};
  const auto hessian_size = runtime::size_mul(coordinates, coordinates);
  std::array<std::vector<Sum>, 2> hessians{std::vector<Sum>(hessian_size),
                                           std::vector<Sum>(hessian_size)};
  runtime::CpuRetainedCapacity retained(
      runtime::vector_capacities(gradients[0], gradients[1], hessians[0], hessians[1]));
  evaluate(
      system, 2, groups, work,
      [&](const Shell& first, const Shell& second, unsigned center, std::size_t row,
          std::size_t column, const std::array<Jet, 2>& jets) {
        const double weight =
            weights[row * count + column] + (row == column ? 0 : weights[column * count + row]);
        for (unsigned part = 0; part < 2; ++part) {
          values[part].add(weight * jets[part][0]);
          for (unsigned axis = 0; axis < 6; ++axis) {
            const double derivative = weight * jets[part][1 + axis];
            scatter_axis(first, second, center, axis, [&](std::size_t coordinate, double sign) {
              gradients[part][coordinate].add(sign * derivative);
            });
            for (unsigned other = 0; other < 6; ++other) {
              const auto slot = 7 + 6 * axis + other;
              const double entry = weight * jets[part][slot];
              scatter_axis(first, second, center, axis, [&](std::size_t left, double first_sign) {
                scatter_axis(first, second, center, other,
                             [&](std::size_t right, double second_sign) {
                               hessians[part][left * coordinates + right].add(first_sign *
                                                                              second_sign * entry);
                             });
              });
            }
          }
        }
      });
  EcpWeightedSecondOrder result;
  result.work = work;
  for (unsigned part = 0; part < 2; ++part) {
    if (!std::isfinite(values[part].value))
      throw std::runtime_error("nonfinite weighted Gaussian ECP value");
    result.parts[part] = {values[part].value, publish(gradients[part]), publish(hessians[part])};
  }
  runtime::sample_cpu_capacity(
      runtime::vector_capacities(result.parts[0].gradient, result.parts[1].gradient,
                                 result.parts[0].hessian, result.parts[1].hessian));
  return result;
}

}  // namespace generativeqc::integrals
