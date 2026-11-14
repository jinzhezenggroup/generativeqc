#include "dft/host_ks_response.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "dft/global_hybrid_fock.hpp"

namespace generativeqc::dft {
namespace {

using response::detail::response_checked_add;
using response::detail::response_checked_multiply;

struct Invocation {
  explicit Invocation(std::atomic_flag& flag) : flag(flag) {
    if (flag.test_and_set()) throw std::logic_error("native CPU CPKS calls overlap or reenter");
  }
  ~Invocation() { flag.clear(); }
  std::atomic_flag& flag;
};

void finite(std::span<const double> values) {
  if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
    throw std::domain_error("nonfinite native CPU CPKS frame or action");
}

bool overlaps(std::span<const double> first, std::span<const double> second) {
  if (first.empty() || second.empty()) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first.data());
  const auto right = reinterpret_cast<std::uintptr_t>(second.data());
  return left <= right ? right - left < first.size_bytes() : left - right < second.size_bytes();
}

bool same_frame(core::ElectronicReferenceView left, core::ElectronicReferenceView right) {
  if (left.basis_functions != right.basis_functions || left.spin_channels != right.spin_channels ||
      left.energy != right.energy)
    return false;
  const auto same = [](auto first, auto second) {
    return first.data() == second.data() && first.size() == second.size();
  };
  if (!same(left.overlap, right.overlap) || !same(left.hcore, right.hcore)) return false;
  for (std::size_t spin = 0; spin < left.spin_channels; ++spin) {
    const auto& first = left.channels[spin];
    const auto& second = right.channels[spin];
    if (first.occupied != second.occupied || !same(first.coefficients, second.coefficients) ||
        !same(first.orbital_energies, second.orbital_energies) ||
        !same(first.density, second.density) || !same(first.fock, second.fock) ||
        !same(first.weighted_density, second.weighted_density))
      return false;
  }
  return true;
}

HostKsResponseDiagnostic diagnostic(const response::GmresResult& result) {
  return {result.status,
          result.residual_norm,
          result.relative_residual,
          result.iterations,
          result.operator_actions,
          result.preconditioner_actions,
          result.workspace_bytes,
          result.measured_workspace_peak_bytes};
}

}  // namespace

HostKsResponse::HostKsResponse(const VerifiedKsFinalState& state, const core::System& system,
                               const scf::PreparedFockPlan& fock, HostGgaResponseShape grid_shape,
                               std::string_view method_identity, HostGgaTileReader reader,
                               ValidateCurrent validate_current, std::size_t max_numeric_bytes,
                               double minimum_orbital_gap)
    : state_(state),
      system_(system),
      fock_(fock),
      identity_(state.identity),
      validate_current_(std::move(validate_current)),
      shape_(grid_shape),
      max_numeric_bytes_(max_numeric_bytes) {
  const auto* binding = host_hessian_aot::find_method(method_identity);
  const auto& model = identity_.model;
  const auto& strategy = identity_.determinant.model;
  if (!binding || !validate_current_ || !reader || !std::isfinite(minimum_orbital_gap) ||
      minimum_orbital_gap < 0 || strategy.backend != scf::FockBackend::Cpu ||
      strategy.precision != scf::FockPrecision::Float64 || strategy != fock_.strategy() ||
      model.version != 1 || model.functional != binding->functional ||
      model.scf_domain_version != binding->scf_domain || model.spins != binding->spins ||
      model.spins != shape_.spins || model.device != -1 || model.range_correction ||
      model.nonlocal_correlation ||
      strategy.spec.coulomb.approximation != scf::FockApproximation::Exact ||
      strategy.spec.exchange.approximation != scf::FockApproximation::Exact ||
      !global_hybrid_coefficients_match(strategy.spec, model.functional, model.spins,
                                        model.semilocal_exchange_scale,
                                        model.semilocal_correlation_scale) ||
      -strategy.spec.exchange.coefficient * (shape_.spins == 1 ? 2.0 : 1.0) !=
          binding->exact_exchange ||
      !state.stationarity_tolerance || !std::isfinite(*state.stationarity_tolerance) ||
      *state.stationarity_tolerance <= 0 || !fock_.matches_system(system_))
    throw std::invalid_argument("native CPU CPKS model, source or verified-state binding mismatch");
  const auto& integrals = fock_.one_electron();
  reference_ = electronic_reference(state_, integrals.overlap, integrals.hcore);
  core::validate_electronic_reference_shape(reference_);
  if (reference_.basis_functions != shape_.basis_functions ||
      reference_.spin_channels != shape_.spins)
    throw std::invalid_argument("native CPU CPKS grid/frame shape mismatch");
  if (shape_.points != grid_point_count(system_, model.grid) ||
      shape_.tile_points != model.tile_points)
    throw std::invalid_argument("native CPU CPKS grid prescription or tile identity mismatch");
  matrix_size_ = response_checked_multiply(shape_.basis_functions, shape_.basis_functions);
  for (std::size_t spin = 0; spin < shape_.spins; ++spin) {
    const auto& channel = reference_.channels[spin];
    finite(channel.coefficients);
    finite(channel.orbital_energies);
    finite(channel.density);
    finite(channel.fock);
    dimension_ = response_checked_add(
        dimension_,
        response_checked_multiply(channel.occupied, shape_.basis_functions - channel.occupied));
    for (std::size_t occupied = 0; occupied < channel.occupied; ++occupied)
      for (std::size_t virtual_index = channel.occupied; virtual_index < shape_.basis_functions;
           ++virtual_index)
        if (const auto gap =
                channel.orbital_energies[virtual_index] - channel.orbital_energies[occupied];
            !std::isfinite(gap) || !(gap > minimum_orbital_gap))
          throw std::domain_error("native CPU CPKS orbital gap is unresolved");
  }
  if (!dimension_) throw std::invalid_argument("native CPU CPKS has no occupied/virtual rotations");
  const bool restricted = shape_.spins == 1;
  auto elements =
      response_checked_multiply(3 * shape_.spins + 1 + (restricted ? 1 : 0), matrix_size_);
  elements =
      response_checked_add(elements, response_checked_multiply(restricted ? 6 : 2, dimension_));
  const auto panel_bytes = response_checked_multiply(elements, sizeof(double));
  const auto xc_bytes = HostGgaXcResponse::workspace_bytes(shape_);
  persistent_bytes_ = response_checked_add(panel_bytes, xc_bytes);
  action_array_bound_bytes_ = response_checked_multiply(10 * sizeof(double), matrix_size_);
  if (response_checked_add(persistent_bytes_, action_array_bound_bytes_) > max_numeric_bytes_)
    throw std::length_error("native CPU CPKS owned-array budget exceeded");
  validate();
  panels_ = std::make_unique<double[]>(elements);
  std::span<double> cursor(panels_.get(), elements);
  const auto take = [&cursor](std::size_t count) {
    const auto panel = cursor.first(count);
    cursor = cursor.subspan(count);
    return panel;
  };
  density_ = take(shape_.spins * matrix_size_);
  direction_ = take(shape_.spins * matrix_size_);
  xc_output_ = take(shape_.spins * matrix_size_);
  zero_ = take(matrix_size_);
  half_density_ = take(restricted ? matrix_size_ : 0);
  diagonal_ = take(dimension_);
  staging_ = take(dimension_);
  singlet_input_ = take(restricted ? 2 * dimension_ : 0);
  singlet_output_ = take(restricted ? 2 * dimension_ : 0);
  std::size_t offset = 0;
  for (std::size_t spin = 0; spin < shape_.spins; ++spin) {
    const auto& channel = reference_.channels[spin];
    std::copy(channel.density.begin(), channel.density.end(),
              density_.begin() + spin * matrix_size_);
    for (std::size_t occupied = 0; occupied < channel.occupied; ++occupied)
      for (std::size_t virtual_index = channel.occupied; virtual_index < shape_.basis_functions;
           ++virtual_index)
        diagonal_[offset++] =
            channel.orbital_energies[virtual_index] - channel.orbital_energies[occupied];
  }
  xc_ = std::make_unique<HostGgaXcResponse>(shape_, xc_bytes, density_, binding->xc_identity,
                                            binding->coefficient_response_identity,
                                            std::move(reader), [this] { validate(); });
  auto spin_reference = reference_;
  if (restricted) {
    std::transform(density_.begin(), density_.end(), half_density_.begin(),
                   [](double value) { return 0.5 * value; });
    spin_reference.spin_channels = 2;
    spin_reference.channels[0].density = half_density_;
    // The shared orbital operator needs no stationary energy-weighted density.
    // Do not attach the occupation-two RKS W to an occupation-one spin frame.
    spin_reference.channels[0].weighted_density = {};
    spin_reference.channels[1] = spin_reference.channels[0];
  }
  full_problem_.emplace(response::spin_orbital_problem(
      spin_reference, [this](const auto& direction) { return induced(direction); },
      [this] { validate(); }, [this](auto directions) { return induced_batch(directions); }));
  if (restricted) {
    problem_.emplace(
        dimension_,
        [this](std::span<const double> input, std::span<double> output) {
          std::copy(input.begin(), input.end(), singlet_input_.begin());
          std::copy(input.begin(), input.end(), singlet_input_.begin() + dimension_);
          full_problem_->apply()(singlet_input_, singlet_output_);
          for (std::size_t index = 0; index < dimension_; ++index)
            output[index] = 0.5 * (singlet_output_[index] + singlet_output_[dimension_ + index]);
        },
        response::LinearResponseSymmetry::Symmetric);
  } else {
    problem_.emplace(*full_problem_);
  }
}

void HostKsResponse::validate() const {
  validate_current_(identity_);
  if (state_.identity != identity_ || fock_.strategy() != identity_.determinant.model ||
      !fock_.matches_system(system_) ||
      !same_frame(reference_, electronic_reference(state_, fock_.one_electron().overlap,
                                                   fock_.one_electron().hcore)))
    throw std::invalid_argument("native CPU CPKS stationary owner or frame changed");
}

void HostKsResponse::validate_output(std::span<double> output) const {
  const auto overlaps = [output](std::span<const double> borrowed) {
    if (output.empty() || borrowed.empty()) return false;
    const auto first = reinterpret_cast<std::uintptr_t>(output.data());
    const auto second = reinterpret_cast<std::uintptr_t>(borrowed.data());
    return first < second ? second - first < output.size_bytes()
                          : first - second < borrowed.size_bytes();
  };
  bool aliased = overlaps(reference_.overlap) || overlaps(reference_.hcore);
  for (std::size_t spin = 0; spin < reference_.spin_channels; ++spin) {
    const auto& channel = reference_.channels[spin];
    aliased |= overlaps(channel.coefficients) || overlaps(channel.orbital_energies) ||
               overlaps(channel.density) || overlaps(channel.fock) ||
               overlaps(channel.weighted_density);
  }
  if (aliased) throw std::invalid_argument("native CPU CPKS output aliases the stationary frame");
}

response::SpinFockAction::result_type HostKsResponse::induced(
    const std::array<std::vector<double>, 2>& density) {
  validate();
  if (density[0].size() != matrix_size_ || density[1].size() != matrix_size_)
    throw std::invalid_argument("native CPU CPKS spin density shape mismatch");
  const bool restricted = shape_.spins == 1;
  for (std::size_t index = 0; index < matrix_size_; ++index) {
    direction_[index] = restricted ? density[0][index] + density[1][index] : density[0][index];
    if (!restricted) direction_[matrix_size_ + index] = density[1][index];
  }
  finite(direction_);
  return induced_current(density);
}

response::SpinFockAction::result_type HostKsResponse::induced_current(
    const std::array<std::vector<double>, 2>& density) {
  const bool restricted = shape_.spins == 1;
  fock_actions_ = response_checked_add(fock_actions_, 1);
  const auto jk = restricted
                      ? fock_.build(std::vector<double>(direction_.begin(), direction_.end()))
                      : fock_.build(density[0], density[1]);
  auto fock = scf::assemble_fock(fock_.strategy(), zero_, jk);
  xc_->apply(direction_, xc_output_);
  if (fock.alpha.size() != matrix_size_ || (!restricted && fock.beta.size() != matrix_size_))
    throw std::runtime_error("native CPU CPKS physical J/K output shape mismatch");
  for (std::size_t index = 0; index < matrix_size_; ++index) {
    fock.alpha[index] += xc_output_[index];
    if (!restricted) fock.beta[index] += xc_output_[matrix_size_ + index];
  }
  if (restricted) fock.beta = fock.alpha;
  finite(fock.alpha);
  finite(fock.beta);
  validate();
  return {std::move(fock.alpha), std::move(fock.beta)};
}

core::ElectronicReferenceView HostKsResponse::reference() const {
  validate();
  return reference_;
}

std::vector<response::SpinFockAction::result_type> HostKsResponse::induced_batch(
    std::span<const response::SpinFockAction::result_type> density) {
  validate();
  const auto count = density.size();
  const auto matrices = shape_.spins * matrix_size_;
  const auto panel = count * matrices;
  auto directions = active_batch_workspace_.first(panel);
  auto xc_output = active_batch_workspace_.subspan(panel, panel);
  auto xc_workspace = active_batch_workspace_.subspan(
      2 * panel, HostGgaXcResponse::batch_workspace_bytes(shape_, count) / sizeof(double));
  const bool restricted = shape_.spins == 1;
  std::vector<response::SpinFockAction::result_type> result(count);
  for (std::size_t request = 0; request < count; ++request) {
    validate();
    const auto& channels = density[request];
    if (channels[0].size() != matrix_size_ || channels[1].size() != matrix_size_)
      throw std::invalid_argument("native CPU CPKS batch spin density shape mismatch");
    auto direction = directions.subspan(request * matrices, matrices);
    for (std::size_t index = 0; index < matrix_size_; ++index) {
      direction[index] = restricted ? channels[0][index] + channels[1][index] : channels[0][index];
      if (!restricted) direction[matrix_size_ + index] = channels[1][index];
    }
    finite(direction);
    fock_actions_ = response_checked_add(fock_actions_, 1);
    const auto jk = restricted
                        ? fock_.build(std::vector<double>(direction.begin(), direction.end()))
                        : fock_.build(channels[0], channels[1]);
    auto fock = scf::assemble_fock(fock_.strategy(), zero_, jk);
    if (fock.alpha.size() != matrix_size_ || (!restricted && fock.beta.size() != matrix_size_))
      throw std::runtime_error("native CPU CPKS batch J/K output shape mismatch");
    result[request] = {std::move(fock.alpha), std::move(fock.beta)};
    validate();
  }
  xc_->apply_batch(directions, xc_output, xc_workspace);
  for (std::size_t request = 0; request < count; ++request) {
    auto& fock = result[request];
    const auto contribution = xc_output.subspan(request * matrices, matrices);
    for (std::size_t index = 0; index < matrix_size_; ++index) {
      fock[0][index] += contribution[index];
      if (!restricted) fock[1][index] += contribution[matrix_size_ + index];
    }
    if (restricted) fock[1] = fock[0];
    finite(fock[0]);
    finite(fock[1]);
  }
  validate();
  return result;
}

void HostKsResponse::apply_density(std::span<const double> density, std::span<double> output) {
  Invocation invocation(busy_);
  if (density.size() != direction_.size() || output.size() != direction_.size())
    throw std::invalid_argument("native CPU CPKS AO density vector shape mismatch");
  validate();
  validate_output(output);
  finite(density);
  std::copy(density.begin(), density.end(), direction_.begin());
  std::array<std::vector<double>, 2> channels;
  if (shape_.spins == 2) {
    channels[0].assign(direction_.begin(), direction_.begin() + matrix_size_);
    channels[1].assign(direction_.begin() + matrix_size_, direction_.end());
  }
  const auto result = induced_current(channels);
  validate();
  for (std::size_t spin = 0; spin < shape_.spins; ++spin)
    std::copy(result[spin].begin(), result[spin].end(), output.begin() + spin * matrix_size_);
}

std::size_t HostKsResponse::batch_workspace_bytes(std::size_t directions) const {
  const auto xc_bytes = HostGgaXcResponse::batch_workspace_bytes(shape_, directions);
  if (directions == 1) return 0;
  auto elements = response_checked_multiply(2 * directions * shape_.spins, matrix_size_);
  elements = response_checked_add(
      elements, response_checked_multiply(directions * (shape_.spins == 1 ? 5 : 1), dimension_));
  return response_checked_add(response_checked_multiply(elements, sizeof(double)), xc_bytes);
}

HostKsResponseResources HostKsResponse::resources(const response::GmresOptions& options,
                                                  std::size_t directions) const {
  const auto plan = response::prepare_response(*problem_, options);
  const auto batch_bytes = batch_workspace_bytes(directions);
  const auto action_bytes = response_checked_multiply(directions, action_array_bound_bytes_);
  const auto gmres_bytes = response_checked_multiply(directions, plan.workspace_bytes);
  const auto base =
      response_checked_add(persistent_bytes_, response_checked_add(action_bytes, batch_bytes));
  return {persistent_bytes_, action_bytes, gmres_bytes,
          plan.workspace_bytes <= options.max_workspace_bytes && base <= max_numeric_bytes_ &&
              gmres_bytes <= max_numeric_bytes_ - base,
          batch_bytes};
}

std::size_t HostKsResponse::admit_xc_cache(const response::GmresOptions& options,
                                           std::size_t maximum_extra_bytes,
                                           std::size_t directions) {
  Invocation invocation(busy_);
  validate();
  const auto current = resources(options, directions);
  if (fock_actions_)
    throw std::logic_error("native CPU XC cache must be admitted before Fock actions");
  if (!current.admitted) return 0;
  const auto available = max_numeric_bytes_ - persistent_bytes_ - current.action_array_bound_bytes -
                         current.gmres_workspace_bytes - current.batch_workspace_bytes;
  const auto before = xc_->scratch_bytes();
  if (!xc_->admit_primal_cache(std::min(available, maximum_extra_bytes))) return 0;
  const auto added = xc_->scratch_bytes() - before;
  persistent_bytes_ = response_checked_add(persistent_bytes_, added);
  return added;
}

void HostKsResponse::apply_current(std::span<const double> input, std::span<double> output) {
  if (input.size() != dimension_ || output.size() != dimension_)
    throw std::invalid_argument("native CPU CPKS orbital vector shape mismatch");
  validate();
  validate_output(output);
  finite(input);
  problem_->apply()(input, staging_);
  finite(staging_);
  validate();
  std::copy(staging_.begin(), staging_.end(), output.begin());
}

void HostKsResponse::apply(std::span<const double> input, std::span<double> output) {
  Invocation invocation(busy_);
  apply_current(input, output);
}

void HostKsResponse::apply_batch_current(std::span<const response::LinearAction> actions,
                                         std::span<double> workspace) {
  const auto count = actions.size();
  const auto bytes = batch_workspace_bytes(count);
  if (workspace.size_bytes() < bytes)
    throw std::length_error("native CPU CPKS batch workspace is too small");
  validate();
  validate_output(workspace);
  const auto action_bytes = response_checked_multiply(count, action_array_bound_bytes_);
  const auto base =
      response_checked_add(persistent_bytes_, response_checked_add(action_bytes, bytes));
  if (base > max_numeric_bytes_)
    throw std::length_error("native CPU CPKS batch action exceeds owned-array cap");
  for (std::size_t index = 0; index < count; ++index) {
    const auto& action = actions[index];
    if (action.input.size() != dimension_ || action.output.size() != dimension_)
      throw std::invalid_argument("native CPU CPKS batch orbital vector shape mismatch");
    validate_output(action.output);
    finite(action.input);
    if (overlaps(workspace, action.input) || overlaps(workspace, action.output))
      throw std::invalid_argument("native CPU CPKS batch workspace aliases an action");
    for (std::size_t prior = 0; prior < index; ++prior)
      if (overlaps(action.output, actions[prior].output))
        throw std::invalid_argument("native CPU CPKS batch outputs overlap");
  }
  if (count == 1) {
    apply_current(actions[0].input, actions[0].output);
    return;
  }
  if (!active_batch_workspace_.empty())
    throw std::logic_error("native CPU CPKS batch workspace is already borrowed");
  struct Borrow {
    std::span<double>& active;
    ~Borrow() { active = {}; }
  } borrow{active_batch_workspace_};
  active_batch_workspace_ = workspace.first(bytes / sizeof(double));
  const auto physical = count * shape_.spins * matrix_size_;
  auto cursor = active_batch_workspace_.subspan(
      2 * physical + HostGgaXcResponse::batch_workspace_bytes(shape_, count) / sizeof(double));
  const auto take = [&cursor](std::size_t elements) {
    const auto result = cursor.first(elements);
    cursor = cursor.subspan(elements);
    return result;
  };
  std::array<response::LinearAction, 4> spin_actions;
  const bool restricted = shape_.spins == 1;
  const auto inputs = take(restricted ? 2 * count * dimension_ : 0);
  const auto outputs = take(restricted ? 2 * count * dimension_ : 0);
  const auto staging = take(count * dimension_);
  for (std::size_t request = 0; request < count; ++request) {
    if (restricted) {
      auto input = inputs.subspan(2 * request * dimension_, 2 * dimension_);
      std::copy(actions[request].input.begin(), actions[request].input.end(), input.begin());
      std::copy(actions[request].input.begin(), actions[request].input.end(),
                input.begin() + dimension_);
      spin_actions[request] = {input, outputs.subspan(2 * request * dimension_, 2 * dimension_)};
    } else {
      spin_actions[request] = {actions[request].input,
                               staging.subspan(request * dimension_, dimension_)};
    }
  }
  full_problem_->apply_batch(std::span<const response::LinearAction>(spin_actions.data(), count));
  if (restricted)
    for (std::size_t request = 0; request < count; ++request)
      for (std::size_t index = 0; index < dimension_; ++index)
        staging[request * dimension_ + index] =
            0.5 * (outputs[2 * request * dimension_ + index] +
                   outputs[(2 * request + 1) * dimension_ + index]);
  finite(staging);
  validate();
  for (std::size_t request = 0; request < count; ++request)
    std::copy_n(staging.begin() + request * dimension_, dimension_,
                actions[request].output.begin());
}

void HostKsResponse::apply_batch(std::span<const response::LinearAction> actions,
                                 std::span<double> workspace) {
  Invocation invocation(busy_);
  apply_batch_current(actions, workspace);
}

HostKsResponseDiagnostic HostKsResponse::solve(std::span<const double> rhs,
                                               std::span<double> solution,
                                               const response::GmresOptions& options,
                                               std::span<const double> initial_guess) {
  Invocation invocation(busy_);
  if (rhs.size() != dimension_ || solution.size() != dimension_)
    throw std::invalid_argument("native CPU CPKS solve vector shape mismatch");
  validate();
  validate_output(solution);
  const auto available = max_numeric_bytes_ - persistent_bytes_ - action_array_bound_bytes_;
  if (!available) {
    // The action alone may fit exactly. Keep valid caller options distinct
    // from our lack of solve capacity rather than manufacturing an invalid
    // zero-workspace GMRES request.
    const auto requested = response::prepare_response(*problem_, options);
    HostKsResponseDiagnostic refused;
    refused.status = response::GmresStatus::workspace_limit;
    refused.gmres_workspace_bytes = requested.workspace_bytes;
    return refused;
  }
  auto bounded = options;
  bounded.max_workspace_bytes = std::min(options.max_workspace_bytes, available);
  const response::LinearResponseProblem checked(
      dimension_, [this](auto input, auto output) { apply_current(input, output); },
      response::LinearResponseSymmetry::Symmetric);
  const auto plan = response::prepare_response(checked, bounded);
  auto result = response::solve_response(plan, checked, rhs, initial_guess, diagonal_);
  validate();
  if (result.converged()) {
    finite(result.solution);
    std::copy(result.solution.begin(), result.solution.end(), solution.begin());
  }
  return diagnostic(result);
}

std::vector<HostKsResponseDiagnostic> HostKsResponse::solve_batch(
    std::span<const HostKsResponseRequest> requests, const response::GmresOptions& options) {
  Invocation invocation(busy_);
  const auto count = requests.size();
  const auto current = resources(options, count);
  validate();
  for (std::size_t index = 0; index < count; ++index) {
    const auto& request = requests[index];
    if (request.rhs.size() != dimension_ || request.solution.size() != dimension_)
      throw std::invalid_argument("native CPU CPKS batch solve vector shape mismatch");
    validate_output(request.solution);
    for (std::size_t prior = 0; prior < index; ++prior)
      if (overlaps(request.solution, requests[prior].solution))
        throw std::invalid_argument("native CPU CPKS batch solutions overlap");
  }
  std::vector<HostKsResponseDiagnostic> diagnostics(count);
  if (!current.admitted) {
    const auto numeric = response::prepare_gmres(dimension_, options);
    for (auto& entry : diagnostics) {
      entry.status = response::GmresStatus::workspace_limit;
      entry.gmres_workspace_bytes = numeric.workspace_bytes;
    }
    return diagnostics;
  }
  auto storage = std::make_unique<double[]>(current.batch_workspace_bytes / sizeof(double));
  std::span<double> workspace(storage.get(), current.batch_workspace_bytes / sizeof(double));
  std::array<response::ResponseSolveRequest, 4> borrowed;
  for (std::size_t index = 0; index < count; ++index)
    borrowed[index] = {requests[index].rhs, requests[index].initial_guess, diagonal_};
  const response::LinearResponseProblem checked(
      dimension_, [this](auto input, auto output) { apply_current(input, output); },
      response::LinearResponseSymmetry::Symmetric, {},
      [this, workspace](auto actions) { apply_batch_current(actions, workspace); });
  const auto plan = response::prepare_response_batch(checked, options, count,
                                                     current.gmres_workspace_bytes, count);
  const auto result = response::solve_response_batch(
      plan, checked, std::span<const response::ResponseSolveRequest>(borrowed.data(), count));
  bool converged = result.complete();
  for (std::size_t index = 0; index < count; ++index) {
    diagnostics[index] = diagnostic(result.responses[index]);
    converged &= diagnostics[index].converged();
  }
  validate();
  if (converged) {
    for (const auto& response : result.responses) finite(response.solution);
    for (std::size_t index = 0; index < count; ++index) {
      std::copy(result.responses[index].solution.begin(), result.responses[index].solution.end(),
                requests[index].solution.begin());
    }
  }
  return diagnostics;
}

}  // namespace generativeqc::dft
