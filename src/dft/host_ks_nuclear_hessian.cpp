#include "dft/host_ks_nuclear_hessian.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
namespace {
using runtime::size_add;
using runtime::size_mul;

struct SourceInvocation {
  explicit SourceInvocation(std::atomic_flag& flag) : flag(flag) {
    if (flag.test_and_set()) throw std::logic_error("stationary Hessian overlaps or reenters");
  }
  ~SourceInvocation() { flag.clear(); }
  std::atomic_flag& flag;
};

std::size_t pair_count(std::size_t atoms) {
  if (!atoms) throw std::invalid_argument("stationary Hessian needs atoms");
  return atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
}

std::size_t panel_elements(std::size_t atoms, std::size_t basis, std::size_t spins) {
  if (!atoms || !basis || (spins != 1 && spins != 2))
    throw std::invalid_argument("invalid stationary Hessian shape");
  const auto matrix = size_mul(basis, basis), dimension = size_mul(3, atoms);
  return size_add(size_add(size_add(size_mul(6 * spins + 4, matrix), size_mul(spins, basis)),
                           size_mul(3, size_add(2, size_mul(2, dimension)))),
                  size_mul(3, dimension));
}

void finite(std::span<const double> values) {
  if (!values.empty() && !values.data()) throw std::invalid_argument("null stationary column");
  for (const auto value : values)
    if (!std::isfinite(value)) throw std::domain_error("nonfinite stationary Hessian column");
}

bool overlaps(std::span<const double> values, const void* data, std::size_t bytes) {
  if (values.empty() || !bytes) return false;
  const auto first = reinterpret_cast<std::uintptr_t>(values.data());
  const auto second = reinterpret_cast<std::uintptr_t>(data);
  return first <= second ? second - first < values.size_bytes() : first - second < bytes;
}
}  // namespace

std::size_t HostKsNuclearHessian::assembly_bytes(std::size_t atoms, std::size_t basis,
                                                 std::size_t spins) {
  return size_add(size_mul(panel_elements(atoms, basis, spins), sizeof(double)),
                  size_mul(atoms, sizeof(std::int32_t)));
}

class HostKsNuclearHessian::Impl {
 public:
  Impl(const VerifiedKsFinalState& state, const core::System& system,
       const scf::PreparedFockPlan& fock, HostGridColumnGeometry& geometry, std::string_view method,
       std::optional<dispersion::D3Parameters> bj, HostKsHessianLimits limits,
       HostKsResponse::ValidateCurrent validator)
      : state(state),
        system(system),
        fock(fock),
        geometry(geometry),
        identity(state.identity),
        method_identity(method),
        limits(limits),
        validator(std::move(validator)),
        expected_components(state.components),
        expected_tolerance(state.stationarity_tolerance) {
    auto& base = geometry.geometry();
    atoms = system.atoms.size();
    spins = identity.model.spins;
    matrix = size_mul(base.basis().nao, base.basis().nao);
    matrices = size_mul(spins, matrix);
    dimension = size_mul(3, atoms);
    fields = size_add(2, size_mul(2, dimension));
    owned_bytes = HostKsNuclearHessian::assembly_bytes(atoms, base.basis().nao, spins);
    const auto nuclear_pairs = pair_count(atoms);
    if (!this->validator || owned_bytes > limits.max_assembly_numeric_bytes ||
        nuclear_pairs > limits.max_nuclear_pairs || !limits.response.max_numeric_bytes ||
        !std::isfinite(limits.energy_absolute_tolerance) || limits.energy_absolute_tolerance < 0 ||
        !std::isfinite(limits.energy_relative_tolerance) || limits.energy_relative_tolerance < 0 ||
        !(limits.energy_absolute_tolerance > 0 || limits.energy_relative_tolerance > 0))
      throw std::invalid_argument("stationary Hessian requires owner and finite explicit limits");
    reserved = size_add(owned_bytes, limits.response.max_numeric_bytes);
    reserved = size_add(reserved, limits.integrals.limits.numeric_bytes);
    reserved = size_add(reserved, limits.xc.numeric_bytes);
    reserved = size_add(reserved, limits.ecp.workspace_bytes);
    if (bj) reserved = size_add(reserved, limits.dispersion.max_numeric_bytes);
    if (reserved > limits.max_numeric_bytes)
      throw std::length_error("stationary Hessian aggregate numeric reservation exceeded");
    for (const auto& entry : host_hessian_aot::programs())
      if (entry.name == "nuclear_pair_column") nuclear = &entry;
    if (!nuclear || nuclear->inputs.size() != 7 || nuclear->outputs != 7 || !nuclear->evaluate)
      throw std::invalid_argument("stationary Hessian nuclear column leaf is absent");
    for (std::size_t index = 0; index < nuclear->inputs.size(); ++index) {
      const auto name = nuclear->inputs[index];
      if (name == "charge_product")
        nuclear_input[index] = 6;
      else if (name.size() == 2 && name[0] == 'r' && name[1] >= '0' && name[1] <= '2')
        nuclear_input[index] = static_cast<std::size_t>(name[1] - '0');
      else if (name.size() == 3 && name.starts_with("dr") && name[2] >= '0' && name[2] <= '2')
        nuclear_input[index] = 3 + static_cast<std::size_t>(name[2] - '0');
      else
        throw std::invalid_argument("stationary Hessian nuclear input contract mismatch");
    }
    authenticate(identity);
    storage = std::make_unique<double[]>(panel_elements(atoms, base.basis().nao, spins));
    std::span<double> cursor(storage.get(), panel_elements(atoms, base.basis().nao, spins));
    const auto take = [&cursor](std::size_t count) {
      auto values = cursor.first(count);
      cursor = cursor.subspan(count);
      return values;
    };
    density = take(matrices);
    weighted = take(matrices);
    density_right = take(matrices);
    weighted_right = take(matrices);
    frozen_fock = take(matrices);
    coefficients = take(matrices);
    energies = take(size_mul(spins, base.basis().nao));
    total_density = take(matrix);
    total_density_right = take(matrix);
    overlap_snapshot = take(matrix);
    hcore_snapshot = take(matrix);
    right = take(dimension);
    integral_column = take(fields);
    xc_column = take(fields);
    staged = take(fields);
    d3_gradient = take(dimension);
    d3_action = take(dimension);
    if (!cursor.empty()) throw std::logic_error("stationary Hessian panel layout mismatch");
    if (state.density.size() != spins || state.weighted_density.size() != spins ||
        state.fock.size() != spins || state.orbitals.size() != spins)
      throw std::invalid_argument("stationary Hessian spin panels are incomplete");
    for (std::size_t spin = 0; spin < spins; ++spin) {
      if (state.density[spin].size() != matrix || state.weighted_density[spin].size() != matrix ||
          state.fock[spin].size() != matrix || state.orbitals[spin].vectors.size() != matrix ||
          state.orbitals[spin].values.size() != base.basis().nao)
        throw std::invalid_argument("stationary Hessian AO panel shape mismatch");
      std::copy(state.density[spin].begin(), state.density[spin].end(),
                density.begin() + spin * matrix);
      std::copy(state.weighted_density[spin].begin(), state.weighted_density[spin].end(),
                weighted.begin() + spin * matrix);
      std::copy(state.fock[spin].begin(), state.fock[spin].end(),
                frozen_fock.begin() + spin * matrix);
      std::copy(state.orbitals[spin].vectors.begin(), state.orbitals[spin].vectors.end(),
                coefficients.begin() + spin * matrix);
      std::copy(state.orbitals[spin].values.begin(), state.orbitals[spin].values.end(),
                energies.begin() + spin * base.basis().nao);
    }
    if (fock.one_electron().overlap.size() != matrix || fock.one_electron().hcore.size() != matrix)
      throw std::invalid_argument("stationary Hessian core matrix shape mismatch");
    std::copy(fock.one_electron().overlap.begin(), fock.one_electron().overlap.end(),
              overlap_snapshot.begin());
    std::copy(fock.one_electron().hcore.begin(), fock.one_electron().hcore.end(),
              hcore_snapshot.begin());
    frame_ready = true;
    for (std::size_t element = 0; element < matrix; ++element) {
      total_density[element] = 0;
      for (std::size_t spin = 0; spin < spins; ++spin)
        total_density[element] += density[spin * matrix + element];
    }
    response = std::make_unique<HostKsNuclearResponse>(
        state, system, fock, base, method, limits.response,
        [this](const auto& current) { authenticate(current); });
    const auto validate = [this] { this->validate(); };
    integral = std::make_unique<HostKsNuclearIntegralCurvature>(
        system, base.basis(), fock.strategy(), density, weighted, method, limits.integrals,
        validate, HostNuclearIntegralCurvatureSchedule::Column);
    xc = std::make_unique<HostGgaNuclearColumn>(geometry, spins, density, method, limits.xc,
                                                validate);
    const auto ecp_plan = integrals::ecp_curvature_plan(system).limits;
    if (limits.ecp.workspace_bytes < ecp_plan.workspace_bytes ||
        limits.ecp.gaussian_kernel_blocks < ecp_plan.gaussian_kernel_blocks ||
        limits.ecp.grid_radial_layers < ecp_plan.grid_radial_layers ||
        limits.ecp.grid_ao_primitive_points < ecp_plan.grid_ao_primitive_points ||
        limits.ecp.grid_contraction_terms < ecp_plan.grid_contraction_terms)
      throw std::length_error("stationary Hessian ECP source reservation exceeded");
    if (bj) {
      std::vector<std::int32_t> numbers(atoms);
      for (std::size_t atom = 0; atom < atoms; ++atom) {
        numbers[atom] = system.atoms[atom].atomic_number;
        std::copy(system.atoms[atom].position.begin(), system.atoms[atom].position.end(),
                  right.begin() + 3 * atom);
      }
      dispersion = std::make_unique<dispersion::HostD3Column>(numbers, right, *bj,
                                                              limits.dispersion, validate);
    }
    validate();
  }

  void authenticate(const KsFinalStateIdentity& current) const {
    validator(current);
    geometry.validate_current();
    if (current != identity || state.identity != identity)
      throw std::invalid_argument("stationary Hessian owner identity changed");
    if (frame_ready) {
      if (state.density.size() != spins || state.weighted_density.size() != spins ||
          state.fock.size() != spins || state.orbitals.size() != spins)
        throw std::invalid_argument("stationary Hessian spin binding changed");
      const auto equal = [](const auto& current, std::span<const double> expected) {
        return current.size() == expected.size() &&
               std::equal(current.begin(), current.end(), expected.begin());
      };
      for (std::size_t spin = 0; spin < spins; ++spin)
        if (!equal(state.density[spin], density.subspan(spin * matrix, matrix)) ||
            !equal(state.weighted_density[spin], weighted.subspan(spin * matrix, matrix)) ||
            !equal(state.fock[spin], frozen_fock.subspan(spin * matrix, matrix)) ||
            !equal(state.orbitals[spin].vectors, coefficients.subspan(spin * matrix, matrix)) ||
            !equal(state.orbitals[spin].values,
                   energies.subspan(spin * geometry.geometry().basis().nao,
                                    geometry.geometry().basis().nao)))
          throw std::invalid_argument("stationary Hessian electronic frame contents changed");
      if (!equal(fock.one_electron().overlap, overlap_snapshot) ||
          !equal(fock.one_electron().hcore, hcore_snapshot) ||
          state.stationarity_tolerance != expected_tolerance ||
          state.components.nuclear != expected_components.nuclear ||
          state.components.one_electron != expected_components.one_electron ||
          state.components.hartree != expected_components.hartree ||
          state.components.xc != expected_components.xc ||
          state.components.exact_exchange != expected_components.exact_exchange)
        throw std::invalid_argument("stationary Hessian verified physical state changed");
    }
  }

  void validate() const {
    authenticate(identity);
    if (response) response->validate_current();
  }

  void output(std::span<double> values) const {
    if (values.size() != fields || !values.data())
      throw std::invalid_argument("stationary Hessian output shape mismatch");
    geometry.reject_borrowed_alias(values);
    const auto reject_owner = [values](const auto& owner) {
      if (overlaps(values, &owner, sizeof(owner)))
        throw std::invalid_argument("stationary Hessian output aliases owner");
    };
    reject_owner(state);
    reject_owner(system);
    reject_owner(fock);
    reject_owner(geometry);
    reject_owner(geometry.geometry());
    reject_owner(geometry.geometry().basis());
    reject_owner(geometry.geometry().grid());
    const auto reject = [values](const auto& data) {
      if (overlaps(values, data.data(), size_mul(data.size(), sizeof(*data.data()))))
        throw std::invalid_argument("stationary Hessian output aliases borrowed state");
    };
    reject(system.atoms);
    reject(system.shells);
    reject(system.ecp_terms);
    for (const auto& shell : system.shells) reject(shell.primitives);
    reject(fock.one_electron().overlap);
    reject(fock.one_electron().hcore);
    reject(state.density);
    reject(state.fock);
    reject(state.weighted_density);
    reject(state.orbitals);
    for (const auto* channels : {&state.density, &state.fock, &state.weighted_density})
      for (const auto& channel : *channels) reject(channel);
    for (const auto& channel : state.orbitals) {
      reject(channel.vectors);
      reject(channel.values);
    }
  }

  double nuclei() {
    double energy{};
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        std::array<double, 7> values{}, input{}, result{};
        double squared{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          values[axis] = system.atoms[first].position[axis] - system.atoms[second].position[axis];
          values[3 + axis] = right[3 * first + axis] - right[3 * second + axis];
          squared += values[axis] * values[axis];
        }
        if (!(squared > 0) || !std::isfinite(squared))
          throw std::domain_error("stationary Hessian nuclear distance invalid");
        values[6] = static_cast<double>(system.atoms[first].ionic_charge()) *
                    system.atoms[second].ionic_charge();
        for (std::size_t index = 0; index < 7; ++index) input[index] = values[nuclear_input[index]];
        if (nuclear->evaluate(input.data(), 7, result.data(), 7, 1))
          throw std::domain_error("stationary Hessian nuclear column nonfinite");
        energy += result[0];
        for (std::size_t axis = 0; axis < 3; ++axis) {
          staged[2 + 3 * first + axis] += result[1 + axis];
          staged[2 + 3 * second + axis] -= result[1 + axis];
          staged[2 + dimension + 3 * first + axis] += result[4 + axis];
          staged[2 + dimension + 3 * second + axis] -= result[4 + axis];
        }
        ++census.nuclear_pairs;
      }
    return energy;
  }

  HostKsHessianDiagnostic apply(std::span<const double> external, std::span<double> output_values,
                                const response::GmresOptions& options) {
    if (busy.test_and_set()) throw std::logic_error("stationary Hessian cannot overlap or reenter");
    struct Guard {
      std::atomic_flag& flag;
      ~Guard() { flag.clear(); }
    } guard{busy};
    batch_ready = false;
    output(output_values);
    if (external.size() != dimension)
      throw std::invalid_argument("stationary Hessian right shape mismatch");
    finite(external);
    std::copy(external.begin(), external.end(), right.begin());
    validate();
    if (integral_matrix_source && !integral_matrix_ready)
      throw std::logic_error("stationary Hessian frozen integral matrix is not prepared");
    ++census.columns_started;
    HostKsHessianDiagnostic diagnostic;
    diagnostic.response = response->solve(right, density_right, weighted_right, options);
    if (!diagnostic.response.converged()) return diagnostic;
    diagnostic.first_ecp_work = response->work().last_ecp;
    return assemble(external, output_values, diagnostic);
  }

  HostKsHessianDiagnostic assemble(std::span<const double> external,
                                   std::span<double> output_values,
                                   HostKsHessianDiagnostic diagnostic) {
    if (integral_matrix_source) {
      response->integral_density_response(density_right, weighted_right, integral_response);
      integral_column[0] = frozen_integral_matrix[0];
      integral_column[1] = 0;
      for (std::size_t row = 0; row < dimension; ++row) {
        integral_column[2 + row] = frozen_integral_matrix[1 + row];
        integral_column[1] += frozen_integral_matrix[1 + row] * right[row];
        double action = integral_response[row];
        for (std::size_t column = 0; column < dimension; ++column)
          action +=
              frozen_integral_matrix[1 + dimension + row * dimension + column] * right[column];
        integral_column[2 + dimension + row] = action;
      }
    } else
      integral->apply_column(right, density_right, weighted_right, integral_column);
    xc->apply(right, density_right, xc_column);
    for (std::size_t element = 0; element < matrix; ++element) {
      total_density_right[element] = 0;
      for (std::size_t spin = 0; spin < spins; ++spin)
        total_density_right[element] += density_right[spin * matrix + element];
    }
    const auto ecp = integrals::checked_ecp_curvature(system, total_density, total_density_right,
                                                      right, limits.ecp, [this] { validate(); });
    census.ecp = ecp.work;
    diagnostic.curvature_ecp_work = ecp.work;
    std::fill(staged.begin(), staged.end(), 0);
    for (std::size_t index = 0; index < fields; ++index) {
      staged[index] = integral_column[index] + xc_column[index];
      for (const auto& part : ecp.parts) staged[index] += part[index];
    }
    const auto nuclear_energy = nuclei();
    double metric_energy{};
    const auto& overlap = fock.one_electron().overlap;
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < geometry.geometry().basis().nao; ++row)
        for (std::size_t column = 0; column < geometry.geometry().basis().nao; ++column)
          metric_energy +=
              weighted[spin * matrix + row * geometry.geometry().basis().nao + column] *
              overlap[column * geometry.geometry().basis().nao + row];
    diagnostic.reconstructed_energy = staged[0] + nuclear_energy + metric_energy;
    diagnostic.verified_energy = state.components.total();
    diagnostic.energy_error =
        std::abs(diagnostic.reconstructed_energy - diagnostic.verified_energy);
    if (!std::isfinite(diagnostic.energy_error) ||
        diagnostic.energy_error >
            limits.energy_absolute_tolerance +
                limits.energy_relative_tolerance * std::abs(diagnostic.verified_energy))
      throw std::domain_error(
          "stationary Hessian source energy does not match verified physical state");
    staged[0] = diagnostic.verified_energy;
    if (dispersion) {
      const auto correction = dispersion->apply(right, d3_gradient, d3_action);
      diagnostic.dispersion_work = dispersion->work();
      staged[0] += correction.energy;
      for (std::size_t index = 0; index < dimension; ++index) {
        staged[2 + index] += d3_gradient[index];
        staged[2 + dimension + index] += d3_action[index];
      }
    }
    staged[1] = 0;
    for (std::size_t index = 0; index < dimension; ++index)
      staged[1] += staged[2 + index] * right[index];
    finite(staged);
    validate();
    if (!std::equal(right.begin(), right.end(), external.begin()))
      throw std::runtime_error("stationary Hessian right changed before publication");
    std::copy(staged.begin(), staged.end(), output_values.begin());
    ++census.columns_completed;
    return diagnostic;
  }

  HostKsHessianResources resources(const response::GmresOptions& options) const {
    const auto child = response->resources(options, batch_width);
    const auto assembly = size_add(owned_bytes, batch_bytes);
    auto fixed = size_add(assembly, child.fixed_bound_bytes);
    fixed = size_add(fixed, integral->resources().numeric_bytes);
    fixed = size_add(fixed, xc->resources().numeric_bytes);
    fixed = size_add(fixed, integrals::ecp_curvature_plan(system).limits.workspace_bytes);
    if (dispersion) fixed = size_add(fixed, dispersion::HostD3Column::plan(atoms).numeric_bytes);
    fixed = size_add(fixed, integral_matrix_bytes);
    return {assembly, reserved, fixed, child.gmres_workspace_bytes, child.admitted};
  }

  std::size_t admit_batch(const response::GmresOptions& options, std::size_t maximum) {
    SourceInvocation guard(busy);
    validate();
    if (!maximum || maximum > 4)
      throw std::invalid_argument("stationary Hessian batch width must be one through four");
    if (census.columns_started || response->work().source_attempts)
      throw std::logic_error("stationary Hessian batches must be admitted before source work");
    if (batch_storage) return batch_width;
    const auto per_direction = size_add(size_add(dimension, size_mul(2, matrices)), fields);
    for (std::size_t width = std::min(maximum, dimension); width > 1; --width) {
      const auto elements = size_mul(width, per_direction);
      const auto bytes = size_mul(elements, sizeof(double));
      if (bytes > limits.max_assembly_numeric_bytes - owned_bytes ||
          bytes > limits.max_numeric_bytes - reserved ||
          !response->resources(options, width).admitted)
        continue;
      std::unique_ptr<double[]> candidate;
      try {
        candidate = std::make_unique<double[]>(elements);
      } catch (const std::bad_alloc&) {
        continue;
      }
      batch_storage = std::move(candidate);
      std::span<double> cursor(batch_storage.get(), elements);
      const auto take = [&cursor](std::size_t amount) {
        const auto values = cursor.first(amount);
        cursor = cursor.subspan(amount);
        return values;
      };
      batch_right = take(width * dimension);
      batch_density = take(width * matrices);
      batch_weighted = take(width * matrices);
      batch_columns = take(width * fields);
      batch_bytes = bytes;
      batch_width = width;
      reserved = size_add(reserved, bytes);
      return width;
    }
    return 1;
  }

  std::vector<HostKsHessianDiagnostic> apply_coordinates(std::size_t first, std::size_t count,
                                                         const response::GmresOptions& options) {
    SourceInvocation guard(busy);
    batch_ready = false;
    validate();
    if (!batch_storage || !count || count > batch_width || first >= dimension ||
        count > dimension - first)
      throw std::invalid_argument("stationary Hessian coordinate batch is not admitted");
    if (integral_matrix_source && !integral_matrix_ready)
      throw std::logic_error("stationary Hessian frozen integral matrix is not prepared");
    std::array<HostKsNuclearRequest, 4> requests;
    for (std::size_t index = 0; index < count; ++index) {
      const auto direction = batch_right.subspan(index * dimension, dimension);
      std::fill(direction.begin(), direction.end(), 0.0);
      direction[first + index] = 1.0;
      requests[index] = {direction, batch_density.subspan(index * matrices, matrices),
                         batch_weighted.subspan(index * matrices, matrices)};
    }
    census.columns_started = size_add(census.columns_started, count);
    const auto solved = response->solve_batch({requests.data(), count}, options);
    std::vector<HostKsHessianDiagnostic> diagnostics(count);
    bool converged = true;
    for (std::size_t index = 0; index < count; ++index) {
      diagnostics[index].response = solved.responses[index];
      diagnostics[index].first_ecp_work = solved.first_ecp_work[index];
      converged &= diagnostics[index].converged();
    }
    if (!converged) return diagnostics;
    for (std::size_t index = 0; index < count; ++index) {
      const auto& request = requests[index];
      std::copy(request.direction.begin(), request.direction.end(), right.begin());
      std::copy(request.density_direction.begin(), request.density_direction.end(),
                density_right.begin());
      std::copy(request.weighted_density_direction.begin(),
                request.weighted_density_direction.end(), weighted_right.begin());
      diagnostics[index] = assemble(
          request.direction, batch_columns.subspan(index * fields, fields), diagnostics[index]);
    }
    validate();
    batch_count = count;
    batch_ready = true;
    return diagnostics;
  }

  const VerifiedKsFinalState& state;
  const core::System& system;
  const scf::PreparedFockPlan& fock;
  HostGridColumnGeometry& geometry;
  KsFinalStateIdentity identity;
  std::string method_identity;
  HostKsHessianLimits limits;
  HostKsResponse::ValidateCurrent validator;
  EnergyComponents expected_components;
  std::optional<double> expected_tolerance;
  bool frame_ready{};
  std::size_t atoms{}, spins{}, matrix{}, matrices{}, dimension{}, fields{}, owned_bytes{},
      reserved{};
  const host_hessian_aot::Program* nuclear{};
  std::array<std::size_t, 7> nuclear_input{};
  std::unique_ptr<double[]> storage;
  std::unique_ptr<double[]> batch_storage;
  std::span<double> batch_right, batch_density, batch_weighted, batch_columns;
  std::size_t batch_width{1}, batch_bytes{}, batch_count{};
  bool batch_ready{};
  std::span<double> density, weighted, density_right, weighted_right, frozen_fock, coefficients,
      energies, overlap_snapshot, hcore_snapshot, total_density, total_density_right, right,
      integral_column, xc_column, staged, d3_gradient, d3_action;
  std::unique_ptr<HostKsNuclearResponse> response;
  std::unique_ptr<HostKsNuclearIntegralCurvature> integral;
  std::unique_ptr<HostKsNuclearIntegralCurvature> integral_matrix_source;
  std::unique_ptr<double[]> integral_matrix_storage;
  std::span<double> frozen_integral_matrix, integral_response;
  std::size_t integral_matrix_bytes{};
  bool coordinate_first_admitted{}, integral_matrix_ready{};
  std::unique_ptr<HostGgaNuclearColumn> xc;
  std::unique_ptr<dispersion::HostD3Column> dispersion;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostKsHessianWork census;
};

HostKsNuclearHessian::HostKsNuclearHessian(
    const VerifiedKsFinalState& state, const core::System& system,
    const scf::PreparedFockPlan& fock, HostGridColumnGeometry& geometry, std::string_view method,
    std::optional<dispersion::D3Parameters> bj, HostKsHessianLimits limits,
    HostKsResponse::ValidateCurrent validate_current)
    : impl_(std::make_unique<Impl>(state, system, fock, geometry, method, bj, limits,
                                   std::move(validate_current))) {}
HostKsNuclearHessian::~HostKsNuclearHessian() = default;
bool HostKsNuclearHessian::admit_coordinate_sources(const response::GmresOptions& options) {
  SourceInvocation guard(impl_->busy);
  impl_->validate();
  impl_->coordinate_first_admitted = impl_->response->admit_coordinate_sources(options);
  return impl_->coordinate_first_admitted;
}
void HostKsNuclearHessian::prepare_coordinate_sources() {
  SourceInvocation guard(impl_->busy);
  impl_->validate();
  impl_->response->prepare_coordinate_sources();
}
bool HostKsNuclearHessian::admit_integral_matrix(HostNuclearIntegralPlan allowance,
                                                 HostNuclearIntegralCurvatureSchedule schedule) {
  SourceInvocation guard(impl_->busy);
  impl_->validate();
  if (impl_->census.columns_started)
    throw std::logic_error("frozen integral matrix must be admitted before columns");
  if (impl_->integral_matrix_source) return true;
  if (!impl_->coordinate_first_admitted) return false;
  if (schedule != HostNuclearIntegralCurvatureSchedule::Matrix &&
      schedule != HostNuclearIntegralCurvatureSchedule::ProjectedMatrix)
    throw std::invalid_argument("frozen integral admission requires a matrix schedule");
  if (schedule == HostNuclearIntegralCurvatureSchedule::ProjectedMatrix &&
      !HostKsNuclearIntegralCurvature::projected_matrix_supported(
          impl_->geometry.geometry().basis(), impl_->density))
    return false;
  const auto plan =
      HostKsNuclearIntegralCurvature::plan(impl_->geometry.geometry().basis(), impl_->spins,
                                           schedule, impl_->limits.integrals.eri_schedule);
  const auto matrix_fields =
      size_add(size_add(1, impl_->dimension), size_mul(impl_->dimension, impl_->dimension));
  const auto storage_count = size_add(matrix_fields, impl_->dimension);
  const auto bytes = size_add(plan.numeric_bytes, size_mul(storage_count, sizeof(double)));
  if (bytes > allowance.numeric_bytes ||
      bytes > impl_->limits.max_numeric_bytes - impl_->reserved ||
      plan.one_electron_primitives > allowance.one_electron_primitives ||
      plan.eri_primitives > allowance.eri_primitives || plan.map_terms > allowance.map_terms)
    return false;
  auto options = impl_->limits.integrals;
  options.limits = plan;
  std::unique_ptr<HostKsNuclearIntegralCurvature> candidate;
  std::unique_ptr<double[]> storage;
  try {
    candidate = std::make_unique<HostKsNuclearIntegralCurvature>(
        impl_->system, impl_->geometry.geometry().basis(), impl_->fock.strategy(), impl_->density,
        impl_->weighted, impl_->method_identity, options, [this] { impl_->validate(); }, schedule);
    storage = std::make_unique<double[]>(storage_count);
  } catch (const std::bad_alloc&) {
    return false;
  }
  impl_->integral_matrix_source = std::move(candidate);
  impl_->integral_matrix_storage = std::move(storage);
  impl_->frozen_integral_matrix = {impl_->integral_matrix_storage.get(), matrix_fields};
  impl_->integral_response = {impl_->integral_matrix_storage.get() + matrix_fields,
                              impl_->dimension};
  impl_->integral_matrix_bytes = bytes;
  impl_->reserved = size_add(impl_->reserved, bytes);
  return true;
}
void HostKsNuclearHessian::prepare_integral_matrix() {
  SourceInvocation guard(impl_->busy);
  impl_->validate();
  if (!impl_->integral_matrix_source)
    throw std::logic_error("frozen integral matrix was not admitted");
  impl_->integral_matrix_ready = false;
  impl_->integral_matrix_source->apply_matrix(impl_->frozen_integral_matrix);
  impl_->integral_matrix_ready = true;
}
void HostKsNuclearHessian::admit_response_cache(const response::GmresOptions& options) {
  SourceInvocation guard(impl_->busy);
  impl_->validate();
  impl_->response->admit_response_cache(options, impl_->batch_width);
}
std::size_t HostKsNuclearHessian::admit_response_batch(const response::GmresOptions& options,
                                                       std::size_t maximum_directions) {
  return impl_->admit_batch(options, maximum_directions);
}
std::vector<HostKsHessianDiagnostic> HostKsNuclearHessian::apply_coordinates(
    std::size_t first, std::size_t count, const response::GmresOptions& options) {
  return impl_->apply_coordinates(first, count, options);
}
std::span<const double> HostKsNuclearHessian::coordinate_column(std::size_t index) const {
  impl_->validate();
  if (!impl_->batch_ready || index >= impl_->batch_count)
    throw std::logic_error("stationary Hessian coordinate batch is not complete");
  return impl_->batch_columns.subspan(index * impl_->fields, impl_->fields);
}
HostKsHessianDiagnostic HostKsNuclearHessian::apply(std::span<const double> right,
                                                    std::span<double> output,
                                                    const response::GmresOptions& options) {
  return impl_->apply(right, output, options);
}
HostKsHessianResources HostKsNuclearHessian::resources(
    const response::GmresOptions& options) const {
  return impl_->resources(options);
}
HostKsHessianWork HostKsNuclearHessian::work() const noexcept {
  auto result = impl_->census;
  result.response = impl_->response->work();
  result.integrals =
      (impl_->integral_matrix_source ? impl_->integral_matrix_source : impl_->integral)->work();
  result.xc = impl_->xc->work();
  if (impl_->dispersion) result.dispersion = impl_->dispersion->work();
  return result;
}
}  // namespace generativeqc::dft
