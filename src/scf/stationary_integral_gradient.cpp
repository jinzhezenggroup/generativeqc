#include "scf/stationary_integral_gradient.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "generated_stationary_integral_weights.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::scf {
bool cpu_stationary_direct_force_work_admitted(const core::System& system, bool hybrid,
                                               std::size_t grid_points,
                                               std::size_t tile_points) noexcept {
  try {
    if (!tile_points || grid_points > 1000000) return false;
    std::size_t primitive_sum = 0;
    for (const auto& shell : system.shells) {
      if (shell.angular_momentum > 2) return false;
      for (const auto& expansion :
           molecule::ao_expansions(shell.angular_momentum, system.basis_representation))
        primitive_sum = runtime::size_add(
            primitive_sum, runtime::size_mul(shell.primitives.size(), expansion.size()));
    }
    const auto atoms = system.atoms.size();
    if (!atoms) return false;
    const auto pairs = runtime::size_mul(atoms, atoms - 1) / 2;
    const auto square = runtime::size_mul(primitive_sum, primitive_sum);
    const auto records = runtime::size_add(
        pairs,
        runtime::size_add(runtime::size_mul(hybrid ? 2 : 1, runtime::size_mul(square, square)),
                          runtime::size_mul(runtime::size_add(atoms, 2), square)));
    const auto tiles = runtime::size_add(grid_points / tile_points,
                                         static_cast<std::size_t>(grid_points % tile_points != 0));
    const auto visits =
        runtime::size_mul(pairs, runtime::size_add(runtime::size_mul(2, grid_points), tiles));
    return records <= 2000000 && visits <= 100000000;
  } catch (...) {
    return false;
  }
}

CpuStationaryIntegralGradient cpu_stationary_integral_gradient(
    const PreparedFockPlan& target, std::span<const double> density,
    std::span<const double> weighted_density, CpuStationaryGga method,
    std::size_t maximum_numeric_bytes) {
  if (method != CpuStationaryGga::Pbe && method != CpuStationaryGga::Pbe0 &&
      method != CpuStationaryGga::B3lyp)
    throw std::invalid_argument("unknown stationary Direct GGA method");
  const bool hybrid = method != CpuStationaryGga::Pbe;
  const double exchange_coefficient = method == CpuStationaryGga::Pbe0 ? -0.125 : -0.1;
  const auto& strategy = target.strategy();
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || (term.approximation == FockApproximation::Exact &&
                             term.op == FockOperator::FullRange && term.omega == 0.0);
  };
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Restricted ||
      !exact(strategy.spec.coulomb) || !exact(strategy.spec.exchange) ||
      !strategy.spec.coulomb.present || strategy.spec.coulomb.coefficient != 1.0 ||
      strategy.spec.exchange.present != hybrid ||
      (hybrid && strategy.spec.exchange.coefficient != exchange_coefficient))
    throw std::invalid_argument("stationary GGA integral weights disagree with the prepared model");
  const auto& system = target.system();
  if (!cpu_stationary_direct_force_work_admitted(system, hybrid, 0, 1))
    throw std::invalid_argument("stationary Direct integral work exceeds its qualified envelope");
  if (!system.ecp_terms.empty() || std::any_of(system.atoms.begin(), system.atoms.end(),
                                               [](const auto& atom) { return atom.ecp_core != 0; }))
    throw std::invalid_argument("stationary Direct integral sources require all-electron topology");
  const auto nbf = target.one_electron().nbf;
  const auto matrix = runtime::size_mul(nbf, nbf);
  const auto ncoord = runtime::size_mul(system.atoms.size(), 3);
  if (density.size() != matrix || weighted_density.size() != matrix ||
      !std::all_of(density.begin(), density.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(weighted_density.begin(), weighted_density.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("stationary Direct D/W is inconsistent or nonfinite");
  runtime::host_trace::Region trace("cpu_direct_stationary_integrals", nbf);
  std::vector<std::size_t> offsets;
  offsets.reserve(runtime::size_add(system.shells.size(), 1));
  offsets.push_back(0);
  std::size_t largest_shell = 0;
  for (const auto& shell : system.shells) {
    const auto count =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation).size();
    offsets.push_back(runtime::size_add(offsets.back(), count));
    largest_shell = std::max(largest_shell, count);
  }
  if (offsets.back() != nbf) throw std::invalid_argument("stationary Direct AO ordering mismatch");
  const auto largest_pair = runtime::size_mul(largest_shell, largest_shell);
  const auto largest_quartet = runtime::size_mul(largest_pair, largest_pair);
  const auto elements = runtime::size_add(
      runtime::size_add(runtime::size_mul(matrix, 3), runtime::size_mul(ncoord, 6)),
      runtime::size_mul(runtime::size_add(largest_quartet, 12), hybrid ? 2 : 1));
  const auto capacity = runtime::size_add(runtime::size_mul(elements, sizeof(double)),
                                          runtime::vector_bytes(offsets));
  if (capacity > maximum_numeric_bytes) throw std::bad_alloc();
  CpuStationaryIntegralGradient result;
  result.numeric_capacity = capacity;
  result.sources.assign(runtime::size_mul(4, ncoord), 0.0);
  std::vector<double> hcore_weights(matrix), pulay_weights(matrix);
  for (std::size_t left = 0; left < nbf; ++left)
    for (std::size_t right = 0; right < nbf; ++right) {
      const std::int64_t indices[]{static_cast<std::int64_t>(left),
                                   static_cast<std::int64_t>(right)};
      const auto slot = left * nbf + right;
      hcore_weights[slot] =
          generated::pbe::one_electron(density.data(), weighted_density.data(), nbf, indices);
      pulay_weights[slot] =
          generated::pbe::overlap_pulay(density.data(), weighted_density.data(), nbf, indices);
    }
  const std::vector<double> zeros(matrix, 0.0);
  const auto hcore =
      integrals::contract_weighted_one_electron_derivative(system, zeros, hcore_weights, false);
  const auto pulay =
      integrals::contract_weighted_one_electron_derivative(system, pulay_weights, zeros, false);
  std::copy(hcore.begin(), hcore.end(), result.sources.begin());
  std::copy(pulay.begin(), pulay.end(), result.sources.begin() + ncoord);
  std::vector<double> coulomb_weights, exchange_weights;
  coulomb_weights.reserve(largest_quartet);
  if (hybrid) exchange_weights.reserve(largest_quartet);
  const auto retained_capacity =
      runtime::vector_capacities(offsets, result.sources, hcore_weights, pulay_weights, zeros,
                                 hcore, pulay, coulomb_weights, exchange_weights);
  result.numeric_capacity =
      runtime::size_add(retained_capacity, runtime::size_mul(hybrid ? 24 : 12, sizeof(double)));
  if (result.numeric_capacity > maximum_numeric_bytes) throw std::bad_alloc();
  for (std::size_t first = 0; first < system.shells.size(); ++first)
    for (std::size_t second = 0; second < system.shells.size(); ++second)
      for (std::size_t third = 0; third < system.shells.size(); ++third)
        for (std::size_t fourth = 0; fourth < system.shells.size(); ++fourth) {
          const std::array<std::size_t, 4> shells{first, second, third, fourth};
          coulomb_weights.clear();
          exchange_weights.clear();
          for (std::size_t ao_first = offsets[first]; ao_first < offsets[first + 1]; ++ao_first)
            for (std::size_t ao_second = offsets[second]; ao_second < offsets[second + 1];
                 ++ao_second)
              for (std::size_t ao_third = offsets[third]; ao_third < offsets[third + 1]; ++ao_third)
                for (std::size_t ao_fourth = offsets[fourth]; ao_fourth < offsets[fourth + 1];
                     ++ao_fourth) {
                  const std::int64_t indices[]{
                      static_cast<std::int64_t>(ao_first), static_cast<std::int64_t>(ao_second),
                      static_cast<std::int64_t>(ao_third), static_cast<std::int64_t>(ao_fourth)};
                  coulomb_weights.push_back(generated::pbe::coulomb(
                      density.data(), weighted_density.data(), nbf, indices));
                  if (method == CpuStationaryGga::Pbe0)
                    exchange_weights.push_back(generated::pbe0::exact_exchange(
                        density.data(), weighted_density.data(), nbf, indices));
                  else if (method == CpuStationaryGga::B3lyp)
                    exchange_weights.push_back(generated::b3lyp::exact_exchange(
                        density.data(), weighted_density.data(), nbf, indices));
                }
          const std::array<std::span<const double>, 2> channels{coulomb_weights, exchange_weights};
          std::uint64_t evaluations = 0;
          const auto derivatives = integrals::contract_weighted_eri_shell_derivatives(
              system, shells, std::span(channels).first(hybrid ? 2 : 1), &evaluations,
              integrals::WeightedEriDerivativeExecution::PackagedCpu);
          if (evaluations >
              std::numeric_limits<std::uint64_t>::max() - result.primitive_evaluations)
            throw std::overflow_error("stationary Direct primitive work count overflow");
          result.primitive_evaluations += evaluations;
          if (result.shell_quartets == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("stationary Direct shell work count overflow");
          ++result.shell_quartets;
          const auto live_capacity =
              runtime::size_add(retained_capacity, runtime::vector_bytes(derivatives));
          result.numeric_capacity = std::max(result.numeric_capacity, live_capacity);
          if (result.numeric_capacity > maximum_numeric_bytes) throw std::bad_alloc();
          runtime::sample_cpu_capacity(live_capacity);
          for (std::size_t channel = 0; channel < derivatives.size(); ++channel)
            for (std::size_t slot = 0; slot < shells.size(); ++slot)
              for (std::size_t axis = 0; axis < 3; ++axis)
                result.sources[(2 + channel) * ncoord + 3 * system.shells[shells[slot]].atom_index +
                               axis] += derivatives[channel][3 * slot + axis];
        }
  if (!std::all_of(result.sources.begin(), result.sources.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("stationary Direct integral sources are nonfinite");
  runtime::df_progress::number("eri_shell_submissions", result.shell_quartets);
  runtime::df_progress::number("eri_primitive_evaluations", result.primitive_evaluations);
  runtime::df_progress::label("eri_primitive_backend", "packaged_cpu_aot");
  runtime::df_progress::number("owned_numeric_capacity_bytes", result.numeric_capacity);
  return result;
}
}  // namespace generativeqc::scf
