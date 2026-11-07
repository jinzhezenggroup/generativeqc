#include "posthf/mp2_derivative_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "hf/reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/df_progress_trace.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::mp2::detail {
namespace {
bool finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::size_t square(std::size_t value) { return posthf::checked_mul(value, value); }
std::size_t fourth(std::size_t value) { return square(square(value)); }

std::vector<std::size_t> shell_offsets(const core::System& system) {
  std::vector<std::size_t> offsets(system.shells.size() + 1, 0);
  for (std::size_t shell = 0; shell < system.shells.size(); ++shell) {
    const auto count = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * system.shells[shell].angular_momentum + 1
                           : molecule::cartesian_count(system.shells[shell].angular_momentum);
    offsets[shell + 1] = posthf::checked_add(offsets[shell], count);
  }
  return offsets;
}

std::vector<double> pullback_matrix(std::span<const double> coefficients,
                                    std::span<const double> mo, std::size_t n) {
  const auto n2 = square(n);
  if (coefficients.size() != n2 || mo.size() != n2)
    throw std::invalid_argument("rank-2 AO pullback shape mismatch");
  std::vector<double> ao(n2), workspace(n2);
  tensor::cpu_congruence('N', n, coefficients.data(), mo.data(), ao.data(), workspace.data());
  return ao;
}

double symmetric_eri_weight(std::span<const double> weights, std::size_t n, std::size_t p,
                            std::size_t q, std::size_t r, std::size_t s) {
  const auto at = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return weights[((a * n + b) * n + c) * n + d];
  };
  // Project arbitrary adjoint weights onto the eight exact ERI permutations.
  // The same coefficient matrix acts on all four slots, so this projection
  // commutes with the MO-to-AO pullback. No additional rank-four buffer is live.
  return 0.125 * at(p, q, r, s) + 0.125 * at(q, p, r, s) + 0.125 * at(p, q, s, r) +
         0.125 * at(q, p, s, r) + 0.125 * at(r, s, p, q) + 0.125 * at(s, r, p, q) +
         0.125 * at(r, s, q, p) + 0.125 * at(s, r, q, p);
}

bool stationary_two_electron_weights(const FactorizedTwoElectronWeights& factors) {
  if (!std::all_of(factors.correlation_iajb.begin(), factors.correlation_iajb.end(),
                   [](double value) { return value == 0.0; }))
    return false;
  for (std::size_t row = 0; row < factors.orbitals; ++row)
    for (std::size_t column = 0; column < factors.orbitals; ++column)
      if (factors.fock[row * factors.orbitals + column] !=
          (row == column && row < factors.occupied ? 1.0 : 0.0))
        return false;
  return true;
}

/** Exact RHF two-electron cotangent, without a staged four-index MO pullback.
 * P is the one-spin occupied projector in this same coefficient frame. The
 * eightfold ERI projection is 2 P_uv P_wx - (P_uw P_vx + P_ux P_vw)/2.
 * Its shell-orbit multiplicity is valid even for repeated shells/atoms. */
void stationary_two_electron_derivative(const core::System& system,
                                        const hf::PhysicalReference& reference,
                                        const std::vector<std::size_t>& offsets,
                                        const EriShellDerivativeAccumulate& eri_shell,
                                        std::vector<double>& derivative) {
  const auto orbitals = reference.nbf;
  std::vector<double> projector(square(orbitals));
  for (std::size_t row = 0; row < orbitals; ++row)
    for (std::size_t column = 0; column <= row; ++column) {
      double value = 0;
      for (std::size_t occupied = 0; occupied < reference.nocc; ++occupied)
        value += reference.coefficients[row * orbitals + occupied] *
                 reference.coefficients[column * orbitals + occupied];
      projector[row * orbitals + column] = projector[column * orbitals + row] = value;
    }
  const auto shell_count = system.shells.size();
  for (std::size_t first = 0; first < shell_count; ++first)
    for (std::size_t second = 0; second <= first; ++second)
      for (std::size_t third = 0; third <= first; ++third)
        for (std::size_t fourth = 0; fourth <= third; ++fourth) {
          if (first == third && fourth > second) continue;
          const auto orbit = (first == second ? 1 : 2) * (third == fourth ? 1 : 2) *
                             (first == third && second == fourth ? 1 : 2);
          const std::array<std::size_t, 4> shells{first, second, third, fourth};
          std::size_t elements = 1;
          for (const auto shell : shells)
            elements = posthf::checked_mul(elements, offsets[shell + 1] - offsets[shell]);
          std::vector<double> local(elements);
          std::size_t element = 0;
          for (auto first_ao = offsets[first]; first_ao < offsets[first + 1]; ++first_ao)
            for (auto second_ao = offsets[second]; second_ao < offsets[second + 1]; ++second_ao)
              for (auto third_ao = offsets[third]; third_ao < offsets[third + 1]; ++third_ao)
                for (auto fourth_ao = offsets[fourth]; fourth_ao < offsets[fourth + 1];
                     ++fourth_ao) {
                  local[element++] = orbit * (2.0 * projector[first_ao * orbitals + second_ao] *
                                                  projector[third_ao * orbitals + fourth_ao] -
                                              0.5 * projector[first_ao * orbitals + third_ao] *
                                                  projector[second_ao * orbitals + fourth_ao] -
                                              0.5 * projector[first_ao * orbitals + fourth_ao] *
                                                  projector[second_ao * orbitals + third_ao]);
                }
          eri_shell(shells, local, derivative);
        }
}

void transform_remaining_shells(const core::System& system, const hf::PhysicalReference& reference,
                                const std::vector<std::size_t>& offsets, std::size_t si,
                                std::span<const double> first,
                                const EriShellDerivativeAccumulate& eri_shell,
                                std::vector<double>& derivative, bool canonical_shells,
                                bool compact_first) {
  const auto n = reference.nbf;
  const auto di = offsets[si + 1] - offsets[si];
  // Factorized weights have an occupied third MO slot. Preserve that exact
  // support through the second stage instead of storing and reducing virtual
  // zeros. Dense inputs and the explicit staged route retain the full domain.
  const auto second_rows = compact_first ? reference.nocc : n;
  for (std::size_t sj = 0; sj < system.shells.size(); ++sj) {
    if (canonical_shells && sj > si) continue;
    const auto dj = offsets[sj + 1] - offsets[sj];
    std::vector<double> second(
        posthf::checked_mul(posthf::checked_mul(di, dj), posthf::checked_mul(second_rows, n)), 0.0);
    if (compact_first) {
      const auto occupied = reference.nocc, virtuals = n - occupied;
      const auto exchange_offset = di * n * occupied;
      const auto correlation_offset = 2 * exchange_offset;
      for (std::size_t iu = 0; iu < di; ++iu)
        for (std::size_t jv = 0; jv < dj; ++jv)
          for (std::size_t r = 0; r < occupied; ++r)
            for (std::size_t s = 0; s < n; ++s) {
              auto& target = second[((iu * dj + jv) * second_rows + r) * n + s];
              const auto nu = offsets[sj] + jv;
              if (s == r) {
                for (std::size_t q = 0; q < n; ++q)
                  target += reference.coefficients[nu * n + q] * first[(iu * n + q) * occupied + r];
              } else {
                target += reference.coefficients[nu * n + r] *
                          first[exchange_offset + (iu * occupied + r) * n + s];
              }
              // The occupied exchange slot precedes all virtual correlation
              // slots in the original q reduction. Keep that order, including
              // its overlapping diagonal in the Coulomb region above.
              if (s >= occupied)
                for (std::size_t a = 0; a < virtuals; ++a)
                  target += reference.coefficients[nu * n + occupied + a] *
                            first[correlation_offset +
                                  ((iu * virtuals + a) * occupied + r) * virtuals + s - occupied];
            }
    } else {
      for (std::size_t iu = 0; iu < di; ++iu)
        for (std::size_t jv = 0; jv < dj; ++jv)
          for (std::size_t r = 0; r < n; ++r)
            for (std::size_t s = 0; s < n; ++s)
              for (std::size_t q = 0; q < n; ++q)
                second[((iu * dj + jv) * n + r) * n + s] +=
                    reference.coefficients[(offsets[sj] + jv) * n + q] *
                    first[((iu * n + q) * n + r) * n + s];
    }
    for (std::size_t sk = 0; sk < system.shells.size(); ++sk) {
      if (canonical_shells && sk > si) continue;
      const auto dk = offsets[sk + 1] - offsets[sk];
      std::vector<double> third(
          posthf::checked_mul(posthf::checked_mul(posthf::checked_mul(di, dj), dk), n), 0.0);
      for (std::size_t iu = 0; iu < di; ++iu)
        for (std::size_t jv = 0; jv < dj; ++jv)
          for (std::size_t kw = 0; kw < dk; ++kw)
            for (std::size_t s = 0; s < n; ++s)
              for (std::size_t r = 0; r < second_rows; ++r)
                third[((iu * dj + jv) * dk + kw) * n + s] +=
                    reference.coefficients[(offsets[sk] + kw) * n + r] *
                    second[((iu * dj + jv) * second_rows + r) * n + s];
      for (std::size_t sl = 0; sl < system.shells.size(); ++sl) {
        if (canonical_shells && (sl > sk || (si == sk && sl > sj))) continue;
        const auto dl = offsets[sl + 1] - offsets[sl];
        std::vector<double> local(
            posthf::checked_mul(posthf::checked_mul(posthf::checked_mul(di, dj), dk), dl), 0.0);
        for (std::size_t iu = 0; iu < di; ++iu)
          for (std::size_t jv = 0; jv < dj; ++jv)
            for (std::size_t kw = 0; kw < dk; ++kw)
              for (std::size_t lx = 0; lx < dl; ++lx)
                for (std::size_t s = 0; s < n; ++s)
                  local[((iu * dj + jv) * dk + kw) * dl + lx] +=
                      reference.coefficients[(offsets[sl] + lx) * n + s] *
                      third[((iu * dj + jv) * dk + kw) * n + s];
        if (canonical_shells) {
          // All components within repeated shells remain present. Multiply by
          // the distinct shell-quartet orbit, not a fixed factor of eight;
          // center derivatives are still differentiated separately and then
          // scattered to atoms, including when several slots share an atom.
          const auto orbit =
              (si == sj ? 1 : 2) * (sk == sl ? 1 : 2) * (si == sk && sj == sl ? 1 : 2);
          for (double& value : local) value *= orbit;
        }
        const std::array<std::size_t, 4> shells{si, sj, sk, sl};
        eri_shell(shells, local, derivative);
      }
    }
  }
}
}  // namespace

std::vector<double> conventional_derivative_accumulate(
    const core::System& system, const hf::PhysicalReference& reference,
    const LagrangianWeights& weights, const OneElectronDerivativeContract& one_electron,
    const EriShellDerivativeAccumulate& eri_shell,
    const std::function<void(std::span<double>)>& finalize, FactorizedFirstLayout first_layout) {
  const auto n = reference.nbf;
  bool dense_two = false;
  if (!weights.two_electron.empty())
    dense_two = weights.two_electron.size() == fourth(n) && finite(weights.two_electron);
  const bool factorized_two = weights.two_electron.empty() &&
                              weights.two_electron_factors.orbitals == n &&
                              weights.two_electron_factors.occupied == reference.nocc &&
                              valid_factorized_two_electron_weights(weights.two_electron_factors);
  if (!n || molecule::ao_count(system) != n || reference.coefficients.size() != square(n) ||
      weights.orbitals != n || weights.occupied != reference.nocc ||
      weights.one_electron.size() != square(n) || weights.overlap.size() != square(n) ||
      (!dense_two && !factorized_two) || !finite(reference.coefficients) ||
      !finite(weights.one_electron) || !finite(weights.overlap) || !one_electron || !eri_shell)
    throw std::invalid_argument("conventional derivative reference/weight mismatch");

  runtime::df_progress::Scope trace("conventional_derivative");
  using Clock = std::chrono::steady_clock;
  const auto now = [&] { return trace.enabled() ? Clock::now() : Clock::time_point{}; };
  const auto started = now();
  const auto one_ao = pullback_matrix(reference.coefficients, weights.one_electron, n);
  const auto overlap_ao = pullback_matrix(reference.coefficients, weights.overlap, n);
  const auto one_started = now();
  auto derivative = one_electron(overlap_ao, one_ao);
  const auto two_started = now();
  if (derivative.size() != posthf::checked_mul(system.atoms.size(), std::size_t{3}) ||
      !finite(derivative))
    throw std::runtime_error("conventional one-electron derivative has the wrong shape");
  const auto offsets = shell_offsets(system);
  if (offsets.back() != n)
    throw std::runtime_error("conventional derivative shell offsets disagree with the reference");

  std::uint64_t shell_ns = 0, shell_calls = 0;
  EriShellDerivativeAccumulate measured_shell;
  if (trace.enabled()) {
    measured_shell = [&](const auto& shells, auto local, auto gradient) {
      const auto shell_started = Clock::now();
      eri_shell(shells, local, gradient);
      shell_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - shell_started)
                      .count();
      ++shell_calls;
    };
  }
  const auto& shell_contract = trace.enabled() ? measured_shell : eri_shell;
  // Recognize only the exact stationary algebra, never a tolerance-based
  // approximation to a small correlation contribution. The explicit staged
  // layout remains an independent numerical/resource fallback for validation.
  const bool stationary_two = factorized_two && first_layout == FactorizedFirstLayout::automatic &&
                              stationary_two_electron_weights(weights.two_electron_factors);
  if (stationary_two) {
    stationary_two_electron_derivative(system, reference, offsets, shell_contract, derivative);
  } else
    for (std::size_t si = 0; si < system.shells.size(); ++si) {
      const auto di = offsets[si + 1] - offsets[si];
      const auto first_elements = posthf::checked_mul(di, posthf::checked_mul(n, square(n)));
      const auto occupied = reference.nocc, virtuals = n - occupied;
      const auto compact_elements =
          factorized_two
              ? posthf::checked_mul(
                    di, posthf::checked_mul(occupied, posthf::checked_add(posthf::checked_mul(2, n),
                                                                          square(virtuals))))
              : first_elements;
      // Never enlarge the admitted legacy staging footprint. The explicit staged
      // route remains available to bounded consumers and independent validation.
      const bool compact_first = factorized_two &&
                                 first_layout == FactorizedFirstLayout::automatic &&
                                 compact_elements < first_elements;
      std::vector<double> first(compact_first ? compact_elements : first_elements, 0.0);
      if (dense_two) {
        for (std::size_t iu = 0; iu < di; ++iu)
          for (std::size_t q = 0; q < n; ++q)
            for (std::size_t r = 0; r < n; ++r)
              for (std::size_t s = 0; s < n; ++s)
                for (std::size_t p = 0; p < n; ++p)
                  first[((iu * n + q) * n + r) * n + s] +=
                      reference.coefficients[(offsets[si] + iu) * n + p] *
                      symmetric_eri_weight(weights.two_electron, n, p, q, r, s);
      } else {
        const auto& factors = weights.two_electron_factors;
        const auto exchange_offset = di * n * occupied;
        const auto correlation_offset = 2 * exchange_offset;
        auto g_index = [occupied, virtuals](std::size_t i, std::size_t j, std::size_t a,
                                            std::size_t b) {
          return ((i * occupied + j) * virtuals + a) * virtuals + b;
        };
        for (std::size_t iu = 0; iu < di; ++iu) {
          const auto mu = offsets[si] + iu;
          for (std::size_t p = 0; p < n; ++p) {
            const double coefficient = reference.coefficients[mu * n + p];
            if (coefficient == 0.0) continue;
            for (std::size_t q = 0; q < n; ++q) {
              const double weight = factors.fock[p * n + q];
              if (weight == 0.0) continue;
              for (std::size_t i = 0; i < occupied; ++i) {
                const auto coulomb =
                    compact_first ? (iu * n + q) * occupied + i : ((iu * n + q) * n + i) * n + i;
                const auto exchange = !compact_first ? ((iu * n + i) * n + i) * n + q
                                      : q == i ? (iu * n + i) * occupied + i
                                               : exchange_offset + (iu * occupied + i) * n + q;
                first[coulomb] += 2.0 * coefficient * weight;
                first[exchange] -= coefficient * weight;
              }
            }
          }
          for (std::size_t i = 0; i < occupied; ++i) {
            const double coefficient = reference.coefficients[mu * n + i];
            if (coefficient == 0.0) continue;
            for (std::size_t j = 0; j < occupied; ++j)
              for (std::size_t a = 0; a < virtuals; ++a)
                for (std::size_t b = 0; b < virtuals; ++b) {
                  const auto correlation =
                      compact_first
                          ? correlation_offset + ((iu * virtuals + a) * occupied + j) * virtuals + b
                          : ((iu * n + occupied + a) * n + j) * n + occupied + b;
                  first[correlation] += coefficient * factors.correlation_iajb[g_index(i, j, a, b)];
                }
          }
        }
      }
      transform_remaining_shells(system, reference, offsets, si, first, shell_contract, derivative,
                                 dense_two, compact_first);
    }
  const auto finalize_started = now();
  if (finalize) finalize(derivative);
  const auto finalize_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now() - finalize_started).count();
  if (trace.enabled()) shell_ns += finalize_ns;
  if (!finite(derivative)) throw std::runtime_error("conventional derivative is nonfinite");
  if (trace.enabled()) {
    using runtime::df_progress::Scope;
    const auto ns = [](auto interval) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count();
    };
    const auto two_ns = ns(now() - two_started);
    // Disjoint intervals. The shell callback may contain nested GPU consumer
    // work; the remainder measures host pullback/scatter and bookkeeping.
    Scope::number("ao_functions", n);
    Scope::number("canonical_shells", dense_two || stationary_two);
    Scope::number("stationary_density_products", stationary_two);
    Scope::number("rank_two_pullback_ns", ns(one_started - started));
    Scope::number("one_electron_ns", ns(two_started - one_started));
    Scope::number("two_electron_shell_calls", shell_calls);
    Scope::number("two_electron_shell_ns", shell_ns);
    Scope::number("two_electron_finalize_ns", finalize_ns);
    Scope::number("two_electron_pullback_scatter_ns", two_ns - shell_ns);
  }
  return derivative;
}

std::vector<double> conventional_derivative(const core::System& system,
                                            const hf::PhysicalReference& reference,
                                            const LagrangianWeights& weights,
                                            const OneElectronDerivativeContract& one_electron,
                                            const EriShellDerivativeContract& eri_shell,
                                            FactorizedFirstLayout first_layout) {
  if (!eri_shell) throw std::invalid_argument("missing conventional ERI derivative consumer");
  return conventional_derivative_accumulate(
      system, reference, weights, one_electron,
      [&](const auto& shells, auto local, auto gradient) {
        const auto center = eri_shell(shells, local);
        for (std::size_t slot = 0; slot < 4; ++slot)
          for (std::size_t axis = 0; axis < 3; ++axis)
            gradient[3 * system.shells[shells[slot]].atom_index + axis] += center[3 * slot + axis];
      },
      {}, first_layout);
}

}  // namespace generativeqc::mp2::detail
