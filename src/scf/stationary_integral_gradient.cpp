#include "scf/stationary_integral_gradient.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dft/ao_grid.hpp"
#include "generated_native_range_force.hpp"
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
    std::span<const double> weighted_density, std::size_t maximum_numeric_bytes,
    std::span<const double> beta_density, std::span<const double> beta_weighted_density) {
  const auto& strategy = target.strategy();
  const bool hybrid = strategy.spec.exchange.present;
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || (term.approximation == FockApproximation::Exact &&
                             term.op == FockOperator::FullRange && term.omega == 0.0);
  };
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  if (strategy.backend != FockBackend::Cpu ||
      (strategy.spec.spin != FockSpin::Restricted && !unrestricted) ||
      !exact(strategy.spec.coulomb) || !exact(strategy.spec.exchange) ||
      !strategy.spec.coulomb.present || strategy.spec.coulomb.coefficient != 1.0 ||
      (hybrid && (!std::isfinite(strategy.spec.exchange.coefficient) ||
                  strategy.spec.exchange.coefficient >= 0.0)))
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
  const auto valid_matrix = [matrix](std::span<const double> values) {
    return values.size() == matrix && std::all_of(values.begin(), values.end(), [](double value) {
             return std::isfinite(value);
           });
  };
  if (!valid_matrix(density) || !valid_matrix(weighted_density) ||
      (unrestricted ? (!valid_matrix(beta_density) || !valid_matrix(beta_weighted_density))
                    : (!beta_density.empty() || !beta_weighted_density.empty())))
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
  const auto elements =
      runtime::size_add(runtime::size_add(runtime::size_mul(matrix, unrestricted ? 7 : 3),
                                          runtime::size_mul(ncoord, 6)),
                        runtime::size_mul(runtime::size_add(largest_quartet, 12), hybrid ? 2 : 1));
  const auto capacity = runtime::size_add(runtime::size_mul(elements, sizeof(double)),
                                          runtime::vector_bytes(offsets));
  if (capacity > maximum_numeric_bytes) throw std::bad_alloc();
  CpuStationaryIntegralGradient result;
  result.numeric_capacity = capacity;
  result.sources.assign(runtime::size_mul(4, ncoord), 0.0);
  std::vector<double> flattened_density, flattened_weighted_density;
  std::span<const double> source_density = density, source_weighted_density = weighted_density;
  if (unrestricted) {
    flattened_density.resize(runtime::size_mul(2, matrix));
    flattened_weighted_density.resize(flattened_density.size());
    std::copy(density.begin(), density.end(), flattened_density.begin());
    std::copy(beta_density.begin(), beta_density.end(), flattened_density.begin() + matrix);
    std::copy(weighted_density.begin(), weighted_density.end(), flattened_weighted_density.begin());
    std::copy(beta_weighted_density.begin(), beta_weighted_density.end(),
              flattened_weighted_density.begin() + matrix);
    source_density = flattened_density;
    source_weighted_density = flattened_weighted_density;
  }
  std::vector<double> hcore_weights(matrix), pulay_weights(matrix);
  for (std::size_t left = 0; left < nbf; ++left)
    for (std::size_t right = 0; right < nbf; ++right) {
      const std::int64_t indices[]{static_cast<std::int64_t>(left),
                                   static_cast<std::int64_t>(right)};
      const auto slot = left * nbf + right;
      hcore_weights[slot] =
          unrestricted ? generated::pbe_uks::one_electron(
                             source_density.data(), source_weighted_density.data(), nbf, indices)
                       : generated::pbe::one_electron(source_density.data(),
                                                      source_weighted_density.data(), nbf, indices);
      pulay_weights[slot] =
          unrestricted ? generated::pbe_uks::overlap_pulay(
                             source_density.data(), source_weighted_density.data(), nbf, indices)
                       : generated::pbe::overlap_pulay(
                             source_density.data(), source_weighted_density.data(), nbf, indices);
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
  const auto retained_capacity = runtime::vector_capacities(
      offsets, result.sources, hcore_weights, pulay_weights, zeros, hcore, pulay, coulomb_weights,
      exchange_weights, flattened_density, flattened_weighted_density);
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
                  coulomb_weights.push_back(
                      unrestricted
                          ? generated::pbe_uks::coulomb(
                                source_density.data(), source_weighted_density.data(), nbf, indices)
                          : generated::pbe::coulomb(source_density.data(),
                                                    source_weighted_density.data(), nbf, indices));
                  if (hybrid) {
                    // TensorIR cotangent for ordered (ab|cd): Fock K coefficient
                    // / 2 * sum_spin D[a,c] D[b,d]. No named hybrid fractions.
                    const auto ac = ao_first * nbf + ao_third;
                    const auto bd = ao_second * nbf + ao_fourth;
                    double pairing = source_density[ac] * source_density[bd];
                    if (unrestricted)
                      pairing += source_density[matrix + ac] * source_density[matrix + bd];
                    exchange_weights.push_back(0.5 * strategy.spec.exchange.coefficient * pairing);
                  }
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
std::vector<double> cpu_stationary_range_correction(const PreparedFockPlan& target,
                                                    const ResolvedFockBuild& correction,
                                                    std::span<const double> density,
                                                    std::span<const double> beta_density,
                                                    std::size_t maximum_bytes) {
  const auto& term = correction.spec.exchange;
  const auto& primary = target.strategy();
  const dft::AoBasis basis(target.system());
  const auto n = basis.nao, matrix = n * n, nc = 3 * basis.natom;
  const bool uks = primary.spec.spin == FockSpin::Unrestricted;
  if (!native_range_force_available || term.omega != native_range_force_omega ||
      correction.backend != FockBackend::Cpu || correction.spec.coulomb.present ||
      correction.spec.spin != primary.spec.spin || !term.present ||
      term.op != FockOperator::LongRange || term.approximation != FockApproximation::Exact ||
      !std::isfinite(term.coefficient) || density.size() != matrix ||
      (uks ? beta_density.size() != matrix : !beta_density.empty()) ||
      !cpu_stationary_direct_force_work_admitted(target.system(), true, 0, 1))
    throw std::invalid_argument("native CPU range force exceeds its operator/work domain");
  constexpr std::size_t tile = 256;
  std::vector<std::int64_t> indices;
  std::vector<double> weights, result(nc), partial(nc);
  indices.reserve(4 * tile);
  weights.reserve(tile);
  // The generated range derivative holds another O(natom) transactional vector.
  const auto bytes =
      runtime::size_add(runtime::vector_capacities(basis.packed, indices, weights, result, partial),
                        runtime::size_mul(nc, sizeof(double)));
  if (bytes > maximum_bytes) throw std::bad_alloc();
  runtime::sample_cpu_capacity(bytes);
  auto submit = [&] {
    if (weights.empty()) return;
    std::uint64_t visits[2]{};
    if (generativeqc_native_range_force_cpu(basis.packed.data(), basis.natom, basis.nprimitive, n,
                                            indices.data(), weights.data(), weights.size(), false,
                                            partial.data(), visits))
      throw std::runtime_error("native CPU range derivative failed");
    for (std::size_t c = 0; c < nc; ++c) result[c] += partial[c];
    indices.clear();
    weights.clear();
  };
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          // Same ordered exchange cotangent as the full-range provider.
          double pairing = density[a * n + c] * density[b * n + d];
          if (uks) pairing += beta_density[a * n + c] * beta_density[b * n + d];
          const double weight = 0.5 * term.coefficient * pairing;
          if (weight == 0) continue;
          indices.insert(indices.end(),
                         {static_cast<std::int64_t>(a), static_cast<std::int64_t>(b),
                          static_cast<std::int64_t>(c), static_cast<std::int64_t>(d)});
          weights.push_back(weight);
          if (weights.size() == tile) submit();
        }
  submit();
  return result;
}
}  // namespace generativeqc::scf
