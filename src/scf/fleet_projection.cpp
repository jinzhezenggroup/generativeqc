#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

#include "runtime/resource_usage.hpp"
#include "scf/fleet.hpp"
#include "scf/system_projection.hpp"

namespace generativeqc::scf {

struct FleetPlan::ProjectionTransaction {
  fleet_projection::Result report;
  std::vector<std::optional<HfWarmState>> candidates;
};

namespace {
struct BudgetExceeded : std::bad_alloc {
  const char* what() const noexcept override {
    return "basis projection exceeds maximum_host_bytes";
  }
};

void coordinates_for(core::System& system, std::span<const double> coordinates) {
  if (coordinates.size() != runtime::size_mul(system.atoms.size(), 3) ||
      !std::all_of(coordinates.begin(), coordinates.end(),
                   [](double x) { return std::isfinite(x); }))
    throw projection::Rejected("invalid projection geometry coordinates");
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    std::copy_n(coordinates.begin() + 3 * atom, 3, system.atoms[atom].position.begin());
}

void check_controls(const fleet_projection::Controls& controls) {
  if (!controls.maximum_host_bytes || !controls.policy.maximum_ao ||
      controls.policy.maximum_workspace_bytes)
    throw std::invalid_argument("projection requires a positive maximum_host_bytes and AO bound");
  for (const auto value : {controls.policy.relative_threshold, controls.policy.validation_tolerance,
                           controls.policy.maximum_residual})
    if (!std::isfinite(value) || value <= 0.0 || value > 1.0)
      throw std::invalid_argument("projection thresholds must be finite and in (0,1]");
}

void admit(std::size_t bytes, const fleet_projection::Controls& controls) {
  if (bytes > controls.maximum_host_bytes) throw BudgetExceeded{};
}
}  // namespace

FleetPlan::ProjectionTransaction FleetPlan::project_fleet(
    const FleetPlan& source, const fleet_projection::Coordinates& coordinates,
    const fleet_projection::Controls& controls, bool execute) const {
  check_controls(controls);
  if (this == &source || size() != source.size() ||
      (!coordinates.empty() && coordinates.size() != size()))
    throw projection::Rejected("distinct source/target fleets with matching item count required");
  if (options_.resolved_fock_build->backend != FockBackend::Cpu ||
      source.options_.resolved_fock_build->backend != FockBackend::Cpu)
    throw system_projection::Unsupported("fleet projection requires two CPU HF owners");
  if (method_ != source.method_)
    throw projection::Rejected("source/target HF spin methods must match");
  if (!warm_starts_enabled_)
    throw projection::Rejected("target initialization requires warm starts enabled");
  for (const auto& state : warm_densities_)
    if (state)
      throw projection::Rejected("target already has a seed; clear warm starts before projection");

  ProjectionTransaction transaction;
  auto& result = transaction.report;
  result.items.resize(size());
  if (execute) transaction.candidates.resize(size());
  for (std::size_t index = 0; index < size(); ++index) {
    auto& item = result.items[index];
    try {
      if (!source.projection_sources_[index])
        throw projection::Rejected("source item has no successful current SCF result");
      if (!source.warm_densities_[index])
        throw projection::Rejected("source item has no retained density");
      const auto& seed = *source.warm_densities_[index];
      auto left = systems_[index], right = source.systems_[index];
      if (!coordinates.empty() && coordinates[index]) coordinates_for(left, *coordinates[index]);
      coordinates_for(right, seed.coordinates);
      const system_projection::Controls inner{controls.policy, method_ == GENERATIVEQC_METHOD_UHF,
                                              controls.maximum_host_bytes};
      const auto required = system_projection::capacity(left, right, inner);
      const auto density_bytes = runtime::size_mul(sizeof(double), required.target_density_count);
      const auto coordinate_bytes = runtime::size_mul(sizeof(double), seed.coordinates.size());
      // The shared system owner includes metrics and detached spin results.
      // Charge the concatenated candidate while those results are still live,
      // plus explicit request/source/validation coordinate copies. Existing HF
      // arenas and nonnumeric topology metadata are not included in this bound.
      const auto workspace = runtime::size_add(
          required.numeric_workspace_bytes,
          runtime::size_add(density_bytes, runtime::size_mul(4, coordinate_bytes)));
      const auto peak = runtime::size_add(result.capacity.retained_bytes, workspace);
      result.capacity.numeric_peak_bytes = std::max(result.capacity.numeric_peak_bytes, peak);
      if (execute) {
        admit(peak, controls);
        const auto projected = system_projection::evaluate(left, right, inner, seed.density);
        HfWarmState candidate;
        candidate.density.reserve(required.target_density_count);
        candidate.coordinates = seed.coordinates;
        candidate.energy = seed.energy;
        candidate.energy_change = seed.energy_change;
        candidate.density_rms = seed.density_rms;
        candidate.iterations = seed.iterations;
        for (std::size_t spin = 0; spin < required.spins; ++spin) {
          candidate.density.insert(candidate.density.end(), projected[spin].density.begin(),
                                   projected[spin].density.end());
          item.diagnostics[spin] = projected[spin].diagnostics;
        }
        transaction.candidates[index] = std::move(candidate);
      }
      item.accepted = true;
      item.spins = required.spins;
      item.numeric_workspace_bytes = workspace;
      item.source_energy = seed.energy;
      item.source_iterations = seed.iterations;
      result.capacity.retained_bytes = runtime::size_add(
          result.capacity.retained_bytes, runtime::size_add(density_bytes, coordinate_bytes));
      result.capacity.largest_density_bytes =
          std::max(result.capacity.largest_density_bytes, density_bytes);
    } catch (const projection::Rejected& error) {
      if (controls.strict)
        throw projection::Rejected("item " + std::to_string(index) + ": " + error.what());
      item.reason = error.what();
    }
  }
  // The existing atomic restore performs independent physical validation of
  // every candidate. Its dense metric/eigen scratch is sequential, while all
  // candidates stay live. Keep that final phase in the same numeric budget.
  const auto validation = runtime::size_add(
      result.capacity.retained_bytes, runtime::size_mul(24, result.capacity.largest_density_bytes));
  result.capacity.numeric_peak_bytes = std::max(result.capacity.numeric_peak_bytes, validation);
  if (execute) admit(validation, controls);
  return transaction;
}

fleet_projection::Capacity FleetPlan::projection_capacity(
    const FleetPlan& source, const fleet_projection::Coordinates& coordinates,
    const fleet_projection::Controls& controls) const {
  return project_fleet(source, coordinates, controls, false).report.capacity;
}

fleet_projection::Result FleetPlan::initialize_from(
    const FleetPlan& source, const fleet_projection::Coordinates& coordinates,
    const fleet_projection::Controls& controls) {
  // Reject any unsupported fleet member before the first scientific projection.
  // Query bounds are conservative; actual admission below charges only accepted
  // candidates, so a rejected neighbor does not consume retained seed storage.
  (void)projection_capacity(source, coordinates, controls);
  auto transaction = project_fleet(source, coordinates, controls, true);
  if (transaction.report.capacity.retained_bytes)
    restore_warm_states(std::move(transaction.candidates));
  return std::move(transaction.report);
}

}  // namespace generativeqc::scf
