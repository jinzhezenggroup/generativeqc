#include "dft/host_ks_perturbation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

#include "host_ks_cpks_maps.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_host_ks_maps;
using response::detail::response_checked_add;
using response::detail::response_checked_multiply;

struct Invocation {
  explicit Invocation(std::atomic_flag& flag) : flag(flag) {
    if (flag.test_and_set())
      throw std::logic_error("native KS perturbation calls overlap or reenter");
  }
  ~Invocation() { flag.clear(); }
  std::atomic_flag& flag;
};

bool overlaps(std::span<const double> first, std::span<const double> second) {
  return maps::overlap(first.data(), first.size(), second.data(), second.size());
}

double norm(std::span<const double> values) {
  double result = 0;
  for (double value : values) result = std::hypot(result, value);
  return result;
}

const maps::Program& program(std::string_view prefix, std::string_view stage) {
  const auto name = std::string(prefix) + "_" + std::string(stage);
  for (const auto& entry : maps::programs())
    if (entry.name == name) return entry;
  throw std::invalid_argument("linked native KS perturbation map is missing: " + name);
}

struct Input {
  std::string_view name;
  std::span<const double> values;
};
struct Output {
  std::string_view name;
  std::span<double> values;
};

}  // namespace

class HostKsPerturbation::Impl {
 public:
  struct Spin {
    maps::Extents shape;
    std::size_t offset{}, rotations{};
    std::span<const double> coefficients, occupied_energies, virtual_energies;
    std::span<double> occupied, virtuals, occupied_energy_matrix, metric_motion, orbital_motion,
        gap;
    std::array<const maps::Program*, 5> programs{};
  };

  Impl(HostKsResponse& response, HostKsPerturbationOptions options)
      : response(response), options(options), reference(response.reference()) {
    if (!options.max_numeric_bytes || !options.max_map_terms ||
        !std::isfinite(options.symmetry_absolute_tolerance) ||
        !std::isfinite(options.symmetry_relative_tolerance) ||
        options.symmetry_absolute_tolerance < 0 || options.symmetry_relative_tolerance < 0 ||
        options.symmetry_relative_tolerance >= 1)
      throw std::invalid_argument("invalid native KS perturbation resource or symmetry limits");
    const auto basis = reference.basis_functions;
    matrix = response_checked_multiply(basis, basis);
    matrices = response_checked_multiply(reference.spin_channels, matrix);
    dimension = response.dimension();
    auto elements = response_checked_add(response_checked_multiply(4, matrices),
                                         response_checked_multiply(4, dimension));
    const auto prefix = reference.spin_channels == 1 ? "rks" : "uks";
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      auto& spin = spins[index];
      const auto& channel = reference.channels[index];
      const auto occupied = channel.occupied;
      spin.shape = {basis, occupied, basis - occupied};
      spin.rotations = response_checked_multiply(occupied, basis - occupied);
      if (index) spin.offset = spins[index - 1].offset + spins[index - 1].rotations;
      spin.coefficients = channel.coefficients;
      spin.occupied_energies = channel.orbital_energies.first(occupied);
      spin.virtual_energies = channel.orbital_energies.subspan(occupied);
      elements = response_checked_add(elements, matrix);
      elements = response_checked_add(elements, response_checked_multiply(occupied, occupied));
      elements = response_checked_add(elements, response_checked_multiply(2 * basis, occupied));
      const std::array stages{"metric", "rhs", "density", "weights", "physical_residual"};
      for (std::size_t stage = 0; stage < stages.size(); ++stage) {
        spin.programs[stage] = &program(prefix, stages[stage]);
        std::size_t bytes{}, terms{};
        if (!spin.programs[stage]->workspace(spin.shape, bytes) ||
            !spin.programs[stage]->work(spin.shape, terms))
          throw std::overflow_error("native KS perturbation map layout/work overflow");
        map_bytes = std::max(map_bytes, bytes);
        map_terms = response_checked_add(map_terms, terms);
      }
    }
    if (map_terms > options.max_map_terms)
      throw std::length_error("native KS perturbation map work budget exceeded");
    elements = response_checked_add(elements, map_bytes / sizeof(double));
    owned_bytes = response_checked_multiply(elements, sizeof(double));
    const auto child = response.resources(response::GmresOptions{});
    const auto response_bytes =
        response_checked_add(child.persistent_bytes, child.action_array_bound_bytes);
    if (response_checked_add(owned_bytes, response_bytes) > options.max_numeric_bytes)
      throw std::length_error("native KS perturbation composed array budget exceeded");
    storage = std::make_unique<double[]>(elements);
    std::span<double> cursor(storage.get(), elements);
    const auto take = [&cursor](std::size_t count) {
      const auto result = cursor.first(count);
      cursor = cursor.subspan(count);
      return result;
    };
    metric_density = take(matrices);
    fock = take(matrices);
    density = take(matrices);
    weights = take(matrices);
    rhs = take(dimension);
    solution = take(dimension);
    residual = take(dimension);
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      auto& spin = spins[index];
      const auto occupied = spin.shape.nocc, virtuals = spin.shape.nvir;
      spin.occupied = take(basis * occupied);
      spin.virtuals = take(basis * virtuals);
      spin.occupied_energy_matrix = take(occupied * occupied);
      spin.metric_motion = take(basis * occupied);
      spin.orbital_motion = take(basis * occupied);
      spin.gap = take(spin.rotations);
      for (std::size_t row = 0; row < basis; ++row) {
        std::copy_n(spin.coefficients.begin() + row * basis, occupied,
                    spin.occupied.begin() + row * occupied);
        std::copy_n(spin.coefficients.begin() + row * basis + occupied, virtuals,
                    spin.virtuals.begin() + row * virtuals);
      }
      for (std::size_t diagonal = 0; diagonal < occupied; ++diagonal)
        spin.occupied_energy_matrix[diagonal * occupied + diagonal] =
            spin.occupied_energies[diagonal];
    }
    arena = take(map_bytes / sizeof(double));
    if (!cursor.empty()) throw std::logic_error("native KS perturbation array layout mismatch");
  }

  HostKsPerturbationResources resources(const response::GmresOptions& solve,
                                        std::size_t directions = 1) const {
    const auto child = response.resources(solve, directions);
    const auto response_bytes = response_checked_add(
        child.persistent_bytes,
        response_checked_add(child.action_array_bound_bytes, child.batch_workspace_bytes));
    const auto owned = response_checked_multiply(directions, owned_bytes);
    const auto base = response_checked_add(owned, response_bytes);
    return {owned,
            response_checked_multiply(directions, map_bytes),
            response_bytes,
            child.gmres_workspace_bytes,
            map_terms,
            child.admitted && base <= options.max_numeric_bytes &&
                child.gmres_workspace_bytes <= options.max_numeric_bytes - base};
  }

  void validate_outputs(std::span<double> first, std::span<double> second) const {
    if (overlaps(first, second))
      throw std::invalid_argument("native KS perturbation outputs alias");
    const auto forbidden = [first, second](std::span<const double> borrowed) {
      return overlaps(first, borrowed) || overlaps(second, borrowed);
    };
    bool alias = forbidden(reference.overlap) || forbidden(reference.hcore);
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      const auto& channel = reference.channels[index];
      alias |= forbidden(channel.coefficients) || forbidden(channel.orbital_energies) ||
               forbidden(channel.density) || forbidden(channel.fock) ||
               forbidden(channel.weighted_density);
    }
    if (alias)
      throw std::invalid_argument("native KS perturbation output aliases stationary frame");
  }

  void symmetric(std::span<const double> values) const {
    if (!maps::finite(values.data(), values.size()))
      throw std::domain_error("nonfinite native KS perturbation input");
    const auto basis = reference.basis_functions;
    for (std::size_t row = 0; row < basis; ++row)
      for (std::size_t column = 0; column < row; ++column) {
        const auto first = values[row * basis + column], second = values[column * basis + row];
        if (std::abs(first - second) >
            options.symmetry_absolute_tolerance +
                options.symmetry_relative_tolerance * std::max(std::abs(first), std::abs(second)))
          throw std::invalid_argument("native KS perturbation requires symmetric S'/F' matrices");
      }
  }

  void run(std::size_t stage, std::size_t index, std::span<const double> overlap,
           std::span<const double> frozen, std::initializer_list<Output> outputs) {
    (void)response.reference();
    const auto& spin = spins[index];
    const auto& selected = *spin.programs[stage];
    const std::array<Input, 12> values{{{"coefficients", spin.coefficients},
                                        {"occupied", spin.occupied},
                                        {"virtuals", spin.virtuals},
                                        {"occupied_energies", spin.occupied_energies},
                                        {"virtual_energies", spin.virtual_energies},
                                        {"occupied_energy_matrix", spin.occupied_energy_matrix},
                                        {"metric_motion", spin.metric_motion},
                                        {"orbital_motion", spin.orbital_motion},
                                        {"rotation", solution.subspan(spin.offset, spin.rotations)},
                                        {"overlap_direction", overlap},
                                        {"frozen_fock", frozen.subspan(index * matrix, matrix)},
                                        {"induced_fock", fock.subspan(index * matrix, matrix)}}};
    std::array<maps::Input, 12> inputs{};
    std::array<maps::Output, 3> results{};
    if (selected.inputs.size() > inputs.size() || selected.outputs.size() > results.size())
      throw std::invalid_argument("native KS perturbation map binding capacity mismatch");
    for (std::size_t slot = 0; slot < selected.inputs.size(); ++slot) {
      const auto found = std::find_if(values.begin(), values.end(), [&](const auto& item) {
        return item.name == selected.inputs[slot];
      });
      if (found == values.end())
        throw std::invalid_argument("native KS perturbation input map mismatch");
      inputs[slot] = {found->values.data(), found->values.size()};
    }
    for (std::size_t slot = 0; slot < selected.outputs.size(); ++slot) {
      const auto found = std::find_if(outputs.begin(), outputs.end(), [&](const auto& item) {
        return item.name == selected.outputs[slot];
      });
      if (found == outputs.end())
        throw std::invalid_argument("native KS perturbation output map mismatch");
      results[slot] = {found->values.data(), found->values.size()};
    }
    std::size_t terms{};
    if (!selected.work(spin.shape, terms))
      throw std::overflow_error("native KS perturbation work overflow");
    census->map_evaluations = response_checked_add(census->map_evaluations, 1);
    census->scheduled_map_terms = response_checked_add(census->scheduled_map_terms, terms);
    const auto status = selected.evaluate(spin.shape, {inputs.data(), selected.inputs.size()},
                                          {results.data(), selected.outputs.size()}, arena);
    if (status != maps::Status::success)
      throw std::domain_error("native KS perturbation map evaluation failed: " +
                              std::string(selected.name));
    (void)response.reference();
  }

  void validate_request(const HostKsPerturbationRequest& request) const {
    (void)response.reference();
    if (request.overlap_direction.size() != matrix || request.frozen_fock.size() != matrices ||
        request.density_direction.size() != matrices ||
        request.weighted_density_direction.size() != matrices)
      throw std::invalid_argument("native KS perturbation matrix shape mismatch");
    validate_outputs(request.density_direction, request.weighted_density_direction);
    symmetric(request.overlap_direction);
    for (std::size_t index = 0; index < reference.spin_channels; ++index)
      symmetric(request.frozen_fock.subspan(index * matrix, matrix));
  }

  void prepare_rhs(std::span<const double> overlap, std::span<const double> frozen) {
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      const auto& spin = spins[index];
      run(0, index, overlap, frozen,
          {{"density", metric_density.subspan(index * matrix, matrix)},
           {"metric_motion", spin.metric_motion},
           {"gap", spin.gap}});
    }
    response.apply_density(metric_density, fock);
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      const auto& spin = spins[index];
      run(1, index, overlap, frozen, {{"rhs", rhs.subspan(spin.offset, spin.rotations)}});
    }
    prepared_rhs_norm = norm(rhs);
  }

  HostKsPerturbationDiagnostic finish_response(std::span<const double> overlap,
                                               std::span<const double> frozen,
                                               const response::GmresOptions& solve_options,
                                               HostKsResponseDiagnostic diagnostic) {
    HostKsPerturbationDiagnostic result;
    result.response = diagnostic;
    if (!result.response.converged()) return result;
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      const auto& spin = spins[index];
      run(2, index, overlap, frozen,
          {{"density", density.subspan(index * matrix, matrix)},
           {"orbital_motion", spin.orbital_motion}});
    }
    response.apply_density(density, fock);
    for (std::size_t index = 0; index < reference.spin_channels; ++index) {
      const auto& spin = spins[index];
      run(4, index, overlap, frozen, {{"residual", residual.subspan(spin.offset, spin.rotations)}});
    }
    result.physical_residual_norm = norm(residual);
    result.relative_physical_residual = prepared_rhs_norm > 0
                                            ? result.physical_residual_norm / prepared_rhs_norm
                                            : result.physical_residual_norm;
    const auto tolerance = std::max(solve_options.absolute_tolerance,
                                    solve_options.relative_tolerance * prepared_rhs_norm);
    if (!std::isfinite(result.physical_residual_norm) ||
        result.physical_residual_norm > tolerance) {
      result.status = HostKsPerturbationStatus::physical_residual_failed;
      return result;
    }
    for (std::size_t index = 0; index < reference.spin_channels; ++index)
      run(3, index, overlap, frozen, {{"d_weights", weights.subspan(index * matrix, matrix)}});
    (void)response.reference();
    result.status = HostKsPerturbationStatus::converged;
    return result;
  }

  HostKsPerturbationDiagnostic solve(std::span<const double> overlap,
                                     std::span<const double> frozen,
                                     std::span<double> output_density,
                                     std::span<double> output_weights,
                                     const response::GmresOptions& solve_options) {
    Invocation invocation(busy);
    validate_request({overlap, frozen, output_density, output_weights});
    const auto plan = resources(solve_options);
    work.directions_started = response_checked_add(work.directions_started, 1);
    if (!plan.admitted) {
      HostKsPerturbationDiagnostic result;
      result.status = HostKsPerturbationStatus::workspace_limit;
      result.response.status = response::GmresStatus::workspace_limit;
      result.response.gmres_workspace_bytes = plan.gmres_workspace_bytes;
      return result;
    }
    prepare_rhs(overlap, frozen);
    auto bounded = solve_options;
    bounded.max_workspace_bytes =
        std::min(solve_options.max_workspace_bytes,
                 options.max_numeric_bytes - owned_bytes - plan.response_array_bound_bytes);
    auto result =
        finish_response(overlap, frozen, solve_options, response.solve(rhs, solution, bounded));
    if (!result.converged()) return result;
    std::copy(density.begin(), density.end(), output_density.begin());
    std::copy(weights.begin(), weights.end(), output_weights.begin());
    work.directions_completed = response_checked_add(work.directions_completed, 1);
    return result;
  }

  std::vector<HostKsPerturbationDiagnostic> solve_batch(
      std::span<const HostKsPerturbationRequest> requests,
      const response::GmresOptions& solve_options) {
    Invocation invocation(busy);
    const auto count = requests.size();
    const auto plan = resources(solve_options, count);
    for (std::size_t index = 0; index < count; ++index) {
      validate_request(requests[index]);
      for (std::size_t prior = 0; prior < index; ++prior) {
        const auto& first = requests[index];
        const auto& second = requests[prior];
        if (overlaps(first.density_direction, second.density_direction) ||
            overlaps(first.density_direction, second.weighted_density_direction) ||
            overlaps(first.weighted_density_direction, second.density_direction) ||
            overlaps(first.weighted_density_direction, second.weighted_density_direction))
          throw std::invalid_argument("native KS perturbation batch outputs overlap");
      }
    }
    std::vector<HostKsPerturbationDiagnostic> diagnostics(count);
    work.directions_started = response_checked_add(work.directions_started, count);
    if (!plan.admitted) {
      for (auto& entry : diagnostics) {
        entry.status = HostKsPerturbationStatus::workspace_limit;
        entry.response.status = response::GmresStatus::workspace_limit;
        entry.response.gmres_workspace_bytes = plan.gmres_workspace_bytes / count;
      }
      return diagnostics;
    }
    std::array<std::unique_ptr<Impl>, 3> extra;
    std::array<Impl*, 4> frames{this};
    std::array<HostKsResponseRequest, 4> solves;
    for (std::size_t index = 1; index < count; ++index) {
      extra[index - 1] = std::make_unique<Impl>(response, options);
      frames[index] = extra[index - 1].get();
      frames[index]->census = &work;
    }
    for (std::size_t index = 0; index < count; ++index) {
      frames[index]->prepare_rhs(requests[index].overlap_direction, requests[index].frozen_fock);
      solves[index] = {frames[index]->rhs, frames[index]->solution};
    }
    const auto linear = response.solve_batch({solves.data(), count}, solve_options);
    bool converged = true;
    for (std::size_t index = 0; index < count; ++index) converged &= linear[index].converged();
    // solve_batch publishes no orbital solution on any linear failure. Do not
    // run D/W maps using an uninitialized or historical per-direction solution.
    if (!converged) {
      for (std::size_t index = 0; index < count; ++index)
        diagnostics[index].response = linear[index];
      return diagnostics;
    }
    for (std::size_t index = 0; index < count; ++index) {
      diagnostics[index] =
          frames[index]->finish_response(requests[index].overlap_direction,
                                         requests[index].frozen_fock, solve_options, linear[index]);
      converged &= diagnostics[index].converged();
    }
    (void)response.reference();
    if (converged) {
      for (std::size_t index = 0; index < count; ++index) {
        std::copy(frames[index]->density.begin(), frames[index]->density.end(),
                  requests[index].density_direction.begin());
        std::copy(frames[index]->weights.begin(), frames[index]->weights.end(),
                  requests[index].weighted_density_direction.begin());
      }
      work.directions_completed = response_checked_add(work.directions_completed, count);
    }
    return diagnostics;
  }

  HostKsResponse& response;
  HostKsPerturbationOptions options;
  core::ElectronicReferenceView reference;
  std::array<Spin, 2> spins{};
  std::size_t matrix{}, matrices{}, dimension{}, owned_bytes{}, map_bytes{}, map_terms{};
  std::unique_ptr<double[]> storage;
  std::span<double> metric_density, fock, density, weights, rhs, solution, residual, arena;
  HostKsPerturbationWork work;
  HostKsPerturbationWork* census{&work};
  double prepared_rhs_norm{};
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

HostKsPerturbation::HostKsPerturbation(HostKsResponse& response, HostKsPerturbationOptions options)
    : impl_(std::make_unique<Impl>(response, options)) {}
HostKsPerturbation::~HostKsPerturbation() = default;
HostKsPerturbationResources HostKsPerturbation::resources(const response::GmresOptions& options,
                                                          std::size_t directions) const {
  return impl_->resources(options, directions);
}
HostKsPerturbationWork HostKsPerturbation::work() const noexcept { return impl_->work; }
std::vector<HostKsPerturbationDiagnostic> HostKsPerturbation::solve_batch(
    std::span<const HostKsPerturbationRequest> requests, const response::GmresOptions& options) {
  return impl_->solve_batch(requests, options);
}
HostKsPerturbationDiagnostic HostKsPerturbation::solve(std::span<const double> overlap_direction,
                                                       std::span<const double> frozen_fock,
                                                       std::span<double> density_direction,
                                                       std::span<double> weighted_density_direction,
                                                       const response::GmresOptions& options) {
  return impl_->solve(overlap_direction, frozen_fock, density_direction, weighted_density_direction,
                      options);
}

}  // namespace generativeqc::dft
