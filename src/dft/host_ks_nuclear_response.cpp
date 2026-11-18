#include "dft/host_ks_nuclear_response.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "dft/host_ao_prefix.hpp"
#include "host_ks_cpks_maps.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_host_ks_maps;
using runtime::size_add;
using runtime::size_mul;

struct Invocation {
  explicit Invocation(std::atomic_flag& flag) : flag(flag) {
    if (flag.test_and_set()) throw std::logic_error("native nuclear response overlaps or reenters");
  }
  ~Invocation() { flag.clear(); }
  std::atomic_flag& flag;
};

bool overlaps(std::span<const double> output, const void* data, std::size_t bytes) {
  if (output.empty() || !bytes) return false;
  const auto first = reinterpret_cast<std::uintptr_t>(output.data());
  const auto second = reinterpret_cast<std::uintptr_t>(data);
  return first <= second ? second - first < output.size_bytes() : first - second < bytes;
}

void admit_ecp(const integrals::EcpDirectionLimits& needed,
               const integrals::EcpDirectionLimits& available) {
  if (needed.workspace_bytes > available.workspace_bytes ||
      needed.gaussian_kernel_blocks > available.gaussian_kernel_blocks ||
      needed.grid_radial_layers > available.grid_radial_layers ||
      needed.grid_ao_primitive_points > available.grid_ao_primitive_points ||
      needed.grid_contraction_terms > available.grid_contraction_terms)
    throw std::length_error("native nuclear response ECP source cap exceeded");
}
}  // namespace

class HostKsNuclearResponse::Impl {
 public:
  Impl(const VerifiedKsFinalState& state, const core::System& system,
       const scf::PreparedFockPlan& fock, HostGridGeometry& geometry, std::string_view method,
       HostKsNuclearLimits limits, HostKsResponse::ValidateCurrent validator)
      : state(state),
        system(system),
        fock(fock),
        geometry(geometry),
        ao_prefix(geometry.basis(), geometry.grid(), geometry.tile_points()),
        identity(state.identity),
        method_identity(method),
        limits(limits),
        validator(std::move(validator)) {
    if (!this->validator || !limits.max_numeric_bytes || !limits.perturbation_map_terms ||
        !limits.assembly_map_terms || geometry.grid().spec() != identity.model.grid)
      throw std::invalid_argument("native nuclear response requires exact grid/owner and limits");
    authenticate(identity);
    const auto spins = identity.model.spins;
    const auto basis = geometry.basis().nao;
    matrix = size_mul(basis, basis);
    matrices = size_mul(spins, matrix);
    shape = {basis, 0, basis};
    const auto name = spins == 1 ? "rks_nuclear_fock" : "uks_nuclear_fock";
    for (const auto& entry : maps::programs())
      if (entry.name == name) assembly = &entry;
    if (!assembly || assembly->inputs.size() != 4 || assembly->outputs.size() != 1 ||
        assembly->outputs[0] != "frozen_fock" || !assembly->workspace(shape, map_bytes) ||
        !assembly->work(shape, assembly_terms) || map_bytes % sizeof(double))
      throw std::invalid_argument("native nuclear response assembly closure is absent");
    assembly_terms = size_mul(spins, assembly_terms);
    if (assembly_terms > limits.assembly_map_terms)
      throw std::length_error("native nuclear response assembly work cap exceeded");
    const auto integral_plan = HostKsNuclearIntegrals::plan(geometry.basis(), spins);
    const auto xc_plan = HostGgaNuclearXc::plan(geometry, spins);
    const auto ecp_plan = integrals::ecp_direction_plan(system);
    admit_ecp(ecp_plan.limits, limits.ecp);
    response = std::make_unique<HostKsResponse>(
        state, system, fock,
        HostGgaResponseShape{basis, geometry.grid().point_count(), spins, geometry.tile_points()},
        method,
        [this](std::size_t begin, std::size_t count, std::span<double> ao,
               std::span<double> weights) {
          authenticate(identity);
          ao_prefix.read(begin, count, ao, weights);
          authenticate(identity);
        },
        [this](const auto& current) { authenticate(current); }, limits.max_numeric_bytes);
    perturbation = std::make_unique<HostKsPerturbation>(
        *response,
        HostKsPerturbationOptions{limits.max_numeric_bytes, limits.perturbation_map_terms});
    const auto response_plan = perturbation->resources({});
    storage_count = size_add(size_add(size_mul(6, matrices), matrix), map_bytes / sizeof(double));
    fixed_bytes = size_add(size_mul(storage_count, sizeof(double)),
                           size_add(integral_plan.numeric_bytes, xc_plan.numeric_bytes));
    fixed_bytes = size_add(fixed_bytes, ecp_plan.limits.workspace_bytes);
    fixed_bytes = size_add(fixed_bytes, size_add(response_plan.owned_numeric_bytes,
                                                 response_plan.response_array_bound_bytes));
    if (fixed_bytes > limits.max_numeric_bytes)
      throw std::length_error("native nuclear response composed array cap exceeded");
    storage = std::make_unique<double[]>(storage_count);
    std::span<double> cursor(storage.get(), storage_count);
    const auto take = [&cursor](std::size_t count) {
      const auto result = cursor.first(count);
      cursor = cursor.subspan(count);
      return result;
    };
    density = take(matrices);
    overlap = take(matrix);
    integral = take(matrices);
    xc_fock = take(matrices);
    frozen = take(matrices);
    moving_density = take(matrices);
    moving_weights = take(matrices);
    arena = take(map_bytes / sizeof(double));
    const auto frame = response->reference();
    for (std::size_t spin = 0; spin < spins; ++spin)
      std::copy(frame.channels[spin].density.begin(), frame.channels[spin].density.end(),
                density.begin() + spin * matrix);
    const auto validate = [this] { this->validate(); };
    integral_source = std::make_unique<HostKsNuclearIntegrals>(
        system, geometry.basis(), fock.strategy(), density, method, limits.integrals, validate);
    xc_source =
        std::make_unique<HostGgaNuclearXc>(geometry, spins, density, method, limits.xc, validate);
    this->validate();
  }

  void authenticate(const KsFinalStateIdentity& current) const {
    validator(current);
    geometry.validate_current();
    if (current != identity || state.identity != identity ||
        geometry.grid().spec() != identity.model.grid)
      throw std::invalid_argument("native nuclear response owner/grid identity changed");
  }

  void validate() const {
    const auto frame = response->reference();
    for (std::size_t spin = 0; spin < frame.spin_channels; ++spin)
      if (!std::equal(frame.channels[spin].density.begin(), frame.channels[spin].density.end(),
                      density.begin() + spin * matrix))
        throw std::invalid_argument("native nuclear response frozen density changed");
  }

  void outputs(std::span<double> first, std::size_t first_size, std::span<double> second,
               std::size_t second_size) const {
    if (!first.data() || first.size() != first_size || !second.data() ||
        second.size() != second_size || overlaps(first, second.data(), second.size_bytes()))
      throw std::invalid_argument("invalid or overlapping native nuclear response outputs");
    reject_output(first);
    reject_output(second);
  }

  void reject_output(std::span<double> output) const {
    const auto frame = response->reference();
    geometry.reject_borrowed_alias(output);
    if (overlaps(output, &system, sizeof(system)) || overlaps(output, &state, sizeof(state)) ||
        overlaps(output, &fock, sizeof(fock)) || overlaps(output, &geometry, sizeof(geometry)) ||
        overlaps(output, &geometry.basis(), sizeof(geometry.basis())) ||
        overlaps(output, &geometry.grid(), sizeof(geometry.grid())))
      throw std::invalid_argument("native nuclear response output aliases borrowed owner");
    const auto reject = [output](const auto& values) {
      if (overlaps(output, values.data(), size_mul(values.size(), sizeof(*values.data()))))
        throw std::invalid_argument("native nuclear response output aliases borrowed state");
    };
    reject(frame.overlap);
    reject(frame.hcore);
    reject(system.atoms);
    reject(system.shells);
    reject(system.ecp_terms);
    for (const auto& shell : system.shells) reject(shell.primitives);
    for (std::size_t spin = 0; spin < frame.spin_channels; ++spin) {
      const auto& channel = frame.channels[spin];
      reject(channel.coefficients);
      reject(channel.orbital_energies);
      reject(channel.density);
      reject(channel.fock);
      reject(channel.weighted_density);
    }
  }

  void source_current(std::span<const double> direction) {
    validate();
    if (!direction.data() || direction.size() != size_mul(3, system.atoms.size()) ||
        !maps::finite(direction.data(), direction.size()))
      throw std::invalid_argument("native nuclear response requires finite atom/xyz direction");
    census.source_attempts = size_add(census.source_attempts, 1);
    (coordinate_source ? coordinate_source : integral_source)->apply(direction, overlap, integral);
    xc_source->apply(direction, xc_fock);
    const auto ecp =
        integrals::checked_ecp_direction(system, direction, limits.ecp, [this] { validate(); });
    census.last_ecp = ecp.work;
    for (std::size_t spin = 0; spin < identity.model.spins; ++spin) {
      std::array<maps::Input, 4> inputs{};
      for (std::size_t index = 0; index < assembly->inputs.size(); ++index) {
        const auto name = assembly->inputs[index];
        const auto values =
            name == "integral"    ? std::span<const double>(integral.subspan(spin * matrix, matrix))
            : name == "xc"        ? std::span<const double>(xc_fock.subspan(spin * matrix, matrix))
            : name == "ecp_local" ? std::span<const double>(ecp.local)
            : name == "ecp_nonlocal"
                ? std::span<const double>(ecp.nonlocal)
                : throw std::invalid_argument("unknown nuclear Fock assembly input");
        inputs[index] = {values.data(), values.size()};
      }
      const std::array<maps::Output, 1> output{{{frozen.data() + spin * matrix, matrix}}};
      if (assembly->evaluate(shape, inputs, output, arena) != maps::Status::success)
        throw std::domain_error("native nuclear Fock assembly failed");
    }
    validate();
    census.sources_completed = size_add(census.sources_completed, 1);
  }

  std::size_t batch_panel_elements(std::size_t directions) const {
    return directions == 1 ? 0 : size_mul(directions, size_add(matrix, size_mul(3, matrices)));
  }

  HostKsNuclearResources resources(const response::GmresOptions& options,
                                   std::size_t directions = 1) const {
    const auto single = perturbation->resources(options);
    const auto child = perturbation->resources(options, directions);
    const auto extra =
        size_add(child.owned_numeric_bytes - single.owned_numeric_bytes,
                 child.response_array_bound_bytes - single.response_array_bound_bytes);
    const auto fixed = size_add(size_add(fixed_bytes, extra),
                                size_mul(batch_panel_elements(directions), sizeof(double)));
    return {fixed,
            child.gmres_workspace_bytes,
            assembly_terms,
            child.admitted && fixed <= limits.max_numeric_bytes &&
                child.gmres_workspace_bytes <= limits.max_numeric_bytes - fixed,
            ao_prefix.numeric_bytes(),
            ao_prefix.prefix_points()};
  }

  bool admit_coordinates(const response::GmresOptions& options) {
    validate();
    if (census.source_attempts || integral_source->work().directions_started)
      throw std::logic_error("coordinate first sources must be admitted before source work");
    if (coordinate_source) return true;
    if (!limits.coordinate_source_bytes ||
        !HostKsNuclearIntegrals::coordinate_density_supported(density))
      return false;
    const auto plan = HostKsNuclearIntegrals::plan(geometry.basis(), identity.model.spins,
                                                   limits.integrals.eri_schedule,
                                                   HostNuclearFirstSchedule::CoordinatePanel);
    const auto current = resources(options);
    if (!current.admitted || plan.numeric_bytes > limits.coordinate_source_bytes ||
        plan.numeric_bytes > limits.max_numeric_bytes - fixed_bytes - current.gmres_workspace_bytes)
      return false;
    auto panel_options = limits.integrals;
    panel_options.limits.numeric_bytes = plan.numeric_bytes;
    try {
      coordinate_source = std::make_unique<HostKsNuclearIntegrals>(
          system, geometry.basis(), fock.strategy(), density, method_identity, panel_options,
          [this] { validate(); }, HostNuclearFirstSchedule::CoordinatePanel);
    } catch (const std::bad_alloc&) {
      return false;
    }
    fixed_bytes = size_add(fixed_bytes, plan.numeric_bytes);
    return true;
  }

  const VerifiedKsFinalState& state;
  const core::System& system;
  const scf::PreparedFockPlan& fock;
  HostGridGeometry& geometry;
  HostAoPrefix ao_prefix;
  KsFinalStateIdentity identity;
  std::string method_identity;
  HostKsNuclearLimits limits;
  HostKsResponse::ValidateCurrent validator;
  std::size_t matrix{}, matrices{}, map_bytes{}, assembly_terms{}, fixed_bytes{}, storage_count{};
  maps::Extents shape;
  const maps::Program* assembly{};
  std::unique_ptr<double[]> storage;
  std::span<double> density, overlap, integral, xc_fock, frozen, moving_density, moving_weights,
      arena;
  std::unique_ptr<HostKsResponse> response;
  std::unique_ptr<HostKsPerturbation> perturbation;
  std::unique_ptr<HostKsNuclearIntegrals> integral_source;
  std::unique_ptr<HostKsNuclearIntegrals> coordinate_source;
  std::unique_ptr<HostGgaNuclearXc> xc_source;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostKsNuclearWork census;
};

HostKsNuclearResponse::HostKsNuclearResponse(const VerifiedKsFinalState& state,
                                             const core::System& system,
                                             const scf::PreparedFockPlan& fock,
                                             HostGridGeometry& geometry, std::string_view method,
                                             HostKsNuclearLimits limits,
                                             HostKsResponse::ValidateCurrent validate_current)
    : impl_(std::make_unique<Impl>(state, system, fock, geometry, method, limits,
                                   std::move(validate_current))) {}
HostKsNuclearResponse::~HostKsNuclearResponse() = default;

bool HostKsNuclearResponse::admit_coordinate_sources(const response::GmresOptions& options) {
  Invocation guard(impl_->busy);
  return impl_->admit_coordinates(options);
}

std::size_t HostKsNuclearResponse::admit_response_cache(const response::GmresOptions& options,
                                                        std::size_t directions) {
  Invocation guard(impl_->busy);
  impl_->validate();
  if (impl_->census.source_attempts || impl_->census.solve_attempts)
    throw std::logic_error("nuclear response cache must be admitted before source work");
  const auto current = impl_->resources(options, directions);
  if (!current.admitted) return 0;
  const auto available =
      impl_->limits.max_numeric_bytes - current.fixed_bound_bytes - current.gmres_workspace_bytes;
  const auto xc_added = impl_->response->admit_xc_cache(options, available, directions);
  impl_->fixed_bytes = size_add(impl_->fixed_bytes, xc_added);
  const auto ao_added = impl_->ao_prefix.admit(available - xc_added);
  impl_->fixed_bytes = size_add(impl_->fixed_bytes, ao_added);
  return size_add(xc_added, ao_added);
}

void HostKsNuclearResponse::prepare_coordinate_sources() {
  Invocation guard(impl_->busy);
  impl_->validate();
  if (!impl_->coordinate_source)
    throw std::logic_error("coordinate first sources were not admitted");
  impl_->coordinate_source->prepare_coordinates();
}
void HostKsNuclearResponse::integral_density_response(
    std::span<const double> density_right, std::span<const double> weighted_density_right,
    std::span<double> output) {
  Invocation guard(impl_->busy);
  impl_->validate();
  if (!impl_->coordinate_source)
    throw std::logic_error("integral moving-density contraction requires coordinate first sources");
  impl_->reject_output(output);
  impl_->coordinate_source->apply_density_response(density_right, weighted_density_right, output,
                                                   impl_->limits.assembly_map_terms);
}

void HostKsNuclearResponse::source(std::span<const double> direction,
                                   std::span<double> overlap_direction,
                                   std::span<double> frozen_fock) {
  Invocation guard(impl_->busy);
  impl_->outputs(overlap_direction, impl_->matrix, frozen_fock, impl_->matrices);
  impl_->source_current(direction);
  std::copy(impl_->overlap.begin(), impl_->overlap.end(), overlap_direction.begin());
  std::copy(impl_->frozen.begin(), impl_->frozen.end(), frozen_fock.begin());
}

HostKsPerturbationDiagnostic HostKsNuclearResponse::solve(
    std::span<const double> direction, std::span<double> density_direction,
    std::span<double> weighted_density_direction, const response::GmresOptions& options) {
  Invocation guard(impl_->busy);
  impl_->outputs(density_direction, impl_->matrices, weighted_density_direction, impl_->matrices);
  impl_->census.solve_attempts = size_add(impl_->census.solve_attempts, 1);
  const auto plan = impl_->resources(options);
  if (!plan.admitted) {
    HostKsPerturbationDiagnostic result;
    result.status = HostKsPerturbationStatus::workspace_limit;
    result.response.status = response::GmresStatus::workspace_limit;
    result.response.gmres_workspace_bytes = plan.gmres_workspace_bytes;
    return result;
  }
  impl_->source_current(direction);
  auto bounded = options;
  bounded.max_workspace_bytes =
      std::min(options.max_workspace_bytes, impl_->limits.max_numeric_bytes - impl_->fixed_bytes);
  auto result = impl_->perturbation->solve(impl_->overlap, impl_->frozen, impl_->moving_density,
                                           impl_->moving_weights, bounded);
  if (!result.converged()) return result;
  impl_->validate();
  std::copy(impl_->moving_density.begin(), impl_->moving_density.end(), density_direction.begin());
  std::copy(impl_->moving_weights.begin(), impl_->moving_weights.end(),
            weighted_density_direction.begin());
  impl_->census.solves_completed = size_add(impl_->census.solves_completed, 1);
  return result;
}

HostKsNuclearBatchResult HostKsNuclearResponse::solve_batch(
    std::span<const HostKsNuclearRequest> requests, const response::GmresOptions& options) {
  const auto count = requests.size();
  const auto plan = impl_->resources(options, count);
  if (count == 1) {
    const auto& request = requests.front();
    return {{solve(request.direction, request.density_direction, request.weighted_density_direction,
                   options)},
            {impl_->census.last_ecp}};
  }
  Invocation guard(impl_->busy);
  impl_->validate();
  for (std::size_t index = 0; index < count; ++index) {
    const auto& request = requests[index];
    impl_->outputs(request.density_direction, impl_->matrices, request.weighted_density_direction,
                   impl_->matrices);
    if (request.direction.size() != size_mul(3, impl_->system.atoms.size()) ||
        !maps::finite(request.direction.data(), request.direction.size()))
      throw std::invalid_argument("native nuclear batch requires finite atom/xyz directions");
    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto& earlier = requests[prior];
      for (const auto output : {request.density_direction, request.weighted_density_direction})
        if (overlaps(output, earlier.density_direction.data(),
                     earlier.density_direction.size_bytes()) ||
            overlaps(output, earlier.weighted_density_direction.data(),
                     earlier.weighted_density_direction.size_bytes()))
          throw std::invalid_argument("native nuclear response batch outputs overlap");
    }
  }
  impl_->census.solve_attempts = size_add(impl_->census.solve_attempts, count);
  HostKsNuclearBatchResult result;
  result.responses.resize(count);
  result.first_ecp_work.resize(count);
  if (!plan.admitted) {
    for (auto& entry : result.responses) {
      entry.status = HostKsPerturbationStatus::workspace_limit;
      entry.response.status = response::GmresStatus::workspace_limit;
      entry.response.gmres_workspace_bytes = plan.gmres_workspace_bytes / count;
    }
    return result;
  }
  const auto elements = impl_->batch_panel_elements(count);
  auto storage = std::make_unique<double[]>(elements);
  std::span<double> cursor(storage.get(), elements);
  const auto take = [&cursor](std::size_t amount) {
    const auto values = cursor.first(amount);
    cursor = cursor.subspan(amount);
    return values;
  };
  std::array<HostKsPerturbationRequest, 4> perturbations;
  for (std::size_t index = 0; index < count; ++index) {
    const auto overlap = take(impl_->matrix), frozen = take(impl_->matrices);
    const auto density = take(impl_->matrices), weights = take(impl_->matrices);
    impl_->source_current(requests[index].direction);
    result.first_ecp_work[index] = impl_->census.last_ecp;
    std::copy(impl_->overlap.begin(), impl_->overlap.end(), overlap.begin());
    std::copy(impl_->frozen.begin(), impl_->frozen.end(), frozen.begin());
    perturbations[index] = {overlap, frozen, density, weights};
  }
  result.responses = impl_->perturbation->solve_batch({perturbations.data(), count}, options);
  bool converged = true;
  for (const auto& entry : result.responses) converged &= entry.converged();
  impl_->validate();
  if (converged) {
    for (std::size_t index = 0; index < count; ++index) {
      std::copy(perturbations[index].density_direction.begin(),
                perturbations[index].density_direction.end(),
                requests[index].density_direction.begin());
      std::copy(perturbations[index].weighted_density_direction.begin(),
                perturbations[index].weighted_density_direction.end(),
                requests[index].weighted_density_direction.begin());
    }
    impl_->census.solves_completed = size_add(impl_->census.solves_completed, count);
  }
  return result;
}

HostKsNuclearResources HostKsNuclearResponse::resources(const response::GmresOptions& options,
                                                        std::size_t directions) const {
  return impl_->resources(options, directions);
}

void HostKsNuclearResponse::validate_current() const { impl_->validate(); }
HostKsNuclearWork HostKsNuclearResponse::work() const noexcept {
  auto result = impl_->census;
  result.integrals =
      (impl_->coordinate_source ? impl_->coordinate_source : impl_->integral_source)->work();
  result.xc = impl_->xc_source->work();
  result.induced_fock_actions = impl_->response->fock_actions();
  result.response_xc_point_visits = impl_->response->xc_work().point_visits;
  result.response_xc_evaluated_points = impl_->response->xc_work().xc_evaluated_points;
  result.response_xc_cached_points = impl_->response->xc_work().cached_xc_points;
  result.response_ao_tile_reads = impl_->ao_prefix.reads();
  result.response_ao_evaluated_points = impl_->ao_prefix.evaluated_points();
  result.response_ao_cached_points = impl_->ao_prefix.cached_points();
  return result;
}
}  // namespace generativeqc::dft
