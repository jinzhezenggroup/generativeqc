#include "posthf/ump2_force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "response/solve.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::mp2 {
namespace {
using Matrix = std::vector<double>;
using SpinMatrices = std::array<Matrix, 2>;
constexpr std::array<std::array<unsigned, 2>, 3> channels{{{0, 0}, {1, 1}, {0, 1}}};
std::size_t mul(std::size_t a, std::size_t b) { return posthf::checked_mul(a, b); }
std::size_t add(std::size_t a, std::size_t b) { return posthf::checked_add(a, b); }
bool finite(std::span<const double> x) {
  return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
}
struct Weights {
  SpinMatrices one, fock, h;
  std::array<Matrix, 3> correlation;
};
std::size_t ci(std::size_t i, std::size_t j, std::size_t a, std::size_t b, std::size_t no_right,
               std::size_t nv_left, std::size_t nv_right) {
  return ((i * no_right + j) * nv_left + a) * nv_right + b;
}

// Enumerate structural support only. Opposite-spin pair reversal is mapped
// into the one alpha-alpha|beta-beta frame before any orbital/AO pullback.
template <class Consume>
void terms(const hf::UnrestrictedPhysicalReference& ref, const Weights& w, unsigned channel,
           Consume consume) {
  const auto n = ref.nbf;
  const auto [left, right] = channels[channel];
  for (unsigned spin = 0; spin < 2; ++spin)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        const auto weight = w.fock[spin][p * n + q];
        if (weight == 0) continue;
        for (unsigned other = 0; other < 2; ++other)
          for (std::size_t i = 0; i < ref.nocc[other]; ++i) {
            if (spin == left && other == right)
              consume(p, q, i, i, weight);
            else if (left != right && spin == right && other == left)
              consume(i, i, p, q, weight);
          }
        if (left == right && spin == left)
          for (std::size_t i = 0; i < ref.nocc[spin]; ++i) consume(p, i, i, q, -weight);
      }
  const auto nl = ref.nocc[left], nr = ref.nocc[right], vl = n - nl, vr = n - nr;
  for (std::size_t i = 0; i < nl; ++i)
    for (std::size_t j = 0; j < nr; ++j)
      for (std::size_t a = 0; a < vl; ++a)
        for (std::size_t b = 0; b < vr; ++b) {
          const auto weight = w.correlation[channel][ci(i, j, a, b, nr, vl, vr)];
          if (weight != 0) consume(i, nl + a, j, nr + b, weight);
        }
}
SpinMatrices rotation_gradient(const hf::UnrestrictedPhysicalReference& ref,
                               const posthf::NativeBlockProvider& provider, const Weights& w) {
  const auto n = ref.nbf, n2 = mul(n, n);
  SpinMatrices gradient{Matrix(n2), Matrix(n2)};
  for (unsigned spin = 0; spin < 2; ++spin)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t t = 0; t < n; ++t) {
          gradient[spin][t * n + p] += w.one[spin][p * n + q] * w.h[spin][t * n + q];
          gradient[spin][t * n + q] += w.one[spin][p * n + q] * w.h[spin][p * n + t];
        }
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), 0);
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto [left, right] = channels[channel];
    for (std::size_t t = 0; t < n; ++t) {
      std::array<Matrix, 4> block;
      for (unsigned slot = 0; slot < 4; ++slot) {
        posthf::SpinMOSlots request{{all, all, all, all}, {left, left, right, right}};
        request.slots[slot] = {t};
        block[slot] = provider.get_spin(request);
      }
      terms(ref, w, channel, [&](auto p, auto q, auto r, auto s, double weight) {
        gradient[left][t * n + p] += weight * block[0][(q * n + r) * n + s];
        gradient[left][t * n + q] += weight * block[1][(p * n + r) * n + s];
        gradient[right][t * n + r] += weight * block[2][(p * n + q) * n + s];
        gradient[right][t * n + s] += weight * block[3][(p * n + q) * n + r];
      });
    }
  }
  for (const auto& g : gradient)
    if (!finite(g)) throw std::runtime_error("nonfinite UMP2 orbital gradient");
  return gradient;
}
void subtract_fock(Weights& w, unsigned spin, std::size_t n, std::size_t p, std::size_t q,
                   double v) {
  w.one[spin][p * n + q] -= v;
  w.fock[spin][p * n + q] -= v;
}

// All three correlation energies and their epsilon/integral cotangents use
// the existing spin-labelled TensorIR and the common generated native VJP.
Weights initial_weights(const hf::UnrestrictedPhysicalReference& ref,
                        const posthf::NativeBlockProvider& provider, double threshold) {
  const auto n = ref.nbf, n2 = mul(n, n);
  Weights w;
  SpinMatrices eps{Matrix(n), Matrix(n)};
  Matrix scratch(n2);
  for (unsigned s = 0; s < 2; ++s) {
    w.one[s].resize(n2);
    w.fock[s].resize(n2);
    w.h[s].resize(n2);
    tensor::cpu_congruence('T', n, ref.coefficients[s].data(), ref.hcore.data(), w.h[s].data(),
                           scratch.data());
  }
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto [left, right] = channels[channel];
    const auto nl = ref.nocc[left], nr = ref.nocc[right], vl = n - nl, vr = n - nr;
    w.correlation[channel].resize(mul(mul(nl, nr), mul(vl, vr)));
    const auto program = generated::ump2_adjoint_cpu_plan(channel);
    for (std::size_t i = 0; i < nl; ++i)
      for (std::size_t j = 0; j < nr; ++j)
        for (std::size_t a = 0; a < vl; ++a)
          for (std::size_t b = 0; b < vr; ++b) {
            const auto ei = ref.orbital_energies[left][i], ej = ref.orbital_energies[right][j];
            const auto ea = ref.orbital_energies[left][nl + a],
                       eb = ref.orbital_energies[right][nr + b];
            const double denominator = ei + ej - ea - eb;
            if (!std::isfinite(denominator) || denominator >= -threshold)
              throw std::invalid_argument("invalid UMP2 force denominator");
            posthf::SpinMOSlots request{
                posthf::MOSlots{std::vector<std::size_t>{i}, std::vector<std::size_t>{nl + a},
                                std::vector<std::size_t>{j}, std::vector<std::size_t>{nr + b}},
                {left, left, right, right}};
            const auto g = provider.get_spin(request).at(0);
            double x = 0;
            if (left == right) {
              request.slots = {std::vector<std::size_t>{i}, std::vector<std::size_t>{nr + b},
                               std::vector<std::size_t>{j}, std::vector<std::size_t>{nl + a}};
              x = provider.get_spin(request).at(0);
            }
            std::array<double, 6> bars{};
            program.run(g, x, ei, ej, ea, eb, bars.data());
            if (!finite(bars)) throw std::runtime_error("nonfinite UMP2 correlation adjoint");
            w.correlation[channel][ci(i, j, a, b, nr, vl, vr)] += bars[0];
            if (left == right) w.correlation[channel][ci(i, j, b, a, nr, vl, vr)] += bars[1];
            eps[left][i] += bars[2];
            eps[right][j] += bars[3];
            eps[left][nl + a] += bars[4];
            eps[right][nr + b] += bars[5];
          }
  }
  for (unsigned s = 0; s < 2; ++s)
    for (std::size_t p = 0; p < n; ++p) w.one[s][p * n + p] = w.fock[s][p * n + p] = eps[s][p];
  return w;
}

// Rectangular per-shell pullback. Each spin frame acts on its own slots;
// no molecular AO-rank-four cotangent or coordinate derivative tensor exists.
Matrix derivative(const hf::UnrestrictedPhysicalReference& ref, const core::System& system,
                  const Weights& w, const SpinMatrices& overlap) {
  const auto n = ref.nbf, n2 = mul(n, n);
  Matrix one(n2), ov(n2), tmp(n2), scratch(n2);
  for (unsigned spin = 0; spin < 2; ++spin) {
    tensor::cpu_congruence('N', n, ref.coefficients[spin].data(), w.one[spin].data(), tmp.data(),
                           scratch.data());
    for (std::size_t k = 0; k < n2; ++k) one[k] += tmp[k];
    tensor::cpu_congruence('N', n, ref.coefficients[spin].data(), overlap[spin].data(), tmp.data(),
                           scratch.data());
    for (std::size_t k = 0; k < n2; ++k) ov[k] += tmp[k];
  }
  auto result = integrals::contract_weighted_one_electron_derivative(system, ov, one, true);
  std::vector<std::size_t> offsets{0};
  for (const auto& shell : system.shells) {
    const auto count = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : molecule::cartesian_count(shell.angular_momentum);
    offsets.push_back(add(offsets.back(), count));
  }
  if (offsets.back() != n) throw std::invalid_argument("UMP2 derivative shell/reference mismatch");
  const auto ns = system.shells.size();
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto [left, right] = channels[channel];
    const auto& ca = ref.coefficients[left];
    const auto& cb = ref.coefficients[right];
    for (std::size_t si = 0; si < ns; ++si) {
      const auto di = offsets[si + 1] - offsets[si];
      Matrix first(mul(di, mul(n, n2)));
      terms(ref, w, channel, [&](auto p, auto q, auto r, auto s, double weight) {
        for (std::size_t u = 0; u < di; ++u)
          first[((u * n + q) * n + r) * n + s] += ca[(offsets[si] + u) * n + p] * weight;
      });
      for (std::size_t sj = 0; sj < ns; ++sj) {
        const auto dj = offsets[sj + 1] - offsets[sj];
        Matrix second(mul(mul(di, dj), n2));
        for (std::size_t u = 0; u < di; ++u)
          for (std::size_t v = 0; v < dj; ++v)
            for (std::size_t r = 0; r < n; ++r)
              for (std::size_t s = 0; s < n; ++s)
                for (std::size_t q = 0; q < n; ++q)
                  second[((u * dj + v) * n + r) * n + s] +=
                      ca[(offsets[sj] + v) * n + q] * first[((u * n + q) * n + r) * n + s];
        for (std::size_t sk = 0; sk < ns; ++sk) {
          const auto dk = offsets[sk + 1] - offsets[sk];
          Matrix third(mul(mul(mul(di, dj), dk), n));
          for (std::size_t u = 0; u < di; ++u)
            for (std::size_t v = 0; v < dj; ++v)
              for (std::size_t z = 0; z < dk; ++z)
                for (std::size_t s = 0; s < n; ++s)
                  for (std::size_t r = 0; r < n; ++r)
                    third[((u * dj + v) * dk + z) * n + s] +=
                        cb[(offsets[sk] + z) * n + r] * second[((u * dj + v) * n + r) * n + s];
          for (std::size_t sl = 0; sl < ns; ++sl) {
            const auto dl = offsets[sl + 1] - offsets[sl];
            Matrix local(mul(mul(mul(di, dj), dk), dl));
            for (std::size_t u = 0; u < di; ++u)
              for (std::size_t v = 0; v < dj; ++v)
                for (std::size_t z = 0; z < dk; ++z)
                  for (std::size_t t = 0; t < dl; ++t)
                    for (std::size_t s = 0; s < n; ++s)
                      local[((u * dj + v) * dk + z) * dl + t] +=
                          cb[(offsets[sl] + t) * n + s] * third[((u * dj + v) * dk + z) * n + s];
            const std::array<std::size_t, 4> shells{si, sj, sk, sl};
            const auto values =
                integrals::contract_weighted_eri_shell_derivative(system, shells, local);
            for (unsigned slot = 0; slot < 4; ++slot)
              for (unsigned xyz = 0; xyz < 3; ++xyz)
                result[3 * system.shells[shells[slot]].atom_index + xyz] += values[3 * slot + xyz];
          }
        }
      }
    }
  }
  if (!finite(result)) throw std::runtime_error("nonfinite UMP2 force contraction");
  for (auto& x : result) x = -x;
  return result;
}
}  // namespace

ConventionalForceResult unrestricted_force_cpu(const hf::UnrestrictedPhysicalReference& ref,
                                               const response::SpinResponseProblem* problem,
                                               const integrals::ElectronInteractionSource& source,
                                               std::size_t budget, double threshold) {
  const auto n = ref.nbf, n2 = mul(n, n);
  if (!n || source.nbf() != n || !ref.source_identity ||
      !ref.source_identity->matches(source.orbital()) || !std::isfinite(threshold) ||
      threshold <= 0 || !budget)
    throw std::invalid_argument("invalid UMP2 force owner/request");
  (void)validate_unrestricted_energy_input(ref, source, threshold);
  const auto dimension = add(mul(ref.nocc[0], n - ref.nocc[0]), mul(ref.nocc[1], n - ref.nocc[1]));
  if ((dimension && (!problem || problem->dimension() != dimension ||
                     !problem->matches(ref.electronic_reference()))) ||
      (!dimension && problem) || !std::isfinite(ref.commutator_residual) ||
      ref.commutator_residual > 1e-8)
    throw std::invalid_argument("UMP2 response owner/dimension mismatch");
  response::GmresOptions options;
  options.absolute_tolerance = 1e-12;
  options.relative_tolerance = 1e-11;
  response::ResponsePlan plan;
  if (problem) plan = response::prepare_response(*problem, options);
  posthf::NativeBlockProvider provider(source, ref, budget, 2);
  std::size_t correlation = 0, max_shell = 1;
  for (const auto [left, right] : channels)
    correlation = add(correlation, mul(mul(ref.nocc[left], ref.nocc[right]),
                                       mul(n - ref.nocc[left], n - ref.nocc[right])));
  for (const auto& shell : source.orbital().shells)
    max_shell =
        std::max(max_shell, source.orbital().basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                                ? std::size_t(2 * shell.angular_momentum + 1)
                                : molecule::cartesian_count(shell.angular_momentum));
  // Conservative composed numeric capacity. Sum coexisting response, all MO
  // weights/frames, four O(N^3) provider blocks, and the four shell-transform
  // stages even though these phases do not all overlap. Object/allocator and
  // compiler stack overhead are outside this numeric-payload contract.
  const auto provider_peak = provider.batch_bytes({1, n, n, n}, 1, false);
  const auto rank3 = mul(n, n2), shell3 = mul(max_shell, max_shell);
  std::size_t elements = add(correlation, add(mul(48, n2), mul(8, rank3)));
  const auto staging_elements =
      add(mul(max_shell, rank3),
          add(mul(shell3, n2), add(mul(mul(shell3, max_shell), n), mul(shell3, shell3))));
  elements = add(elements, staging_elements);
  elements = add(elements, mul(8, mul(source.orbital().atoms.size(), 3)));
  auto peak = add(provider_peak, add(mul(elements, sizeof(double)), plan.workspace_bytes));
  for (unsigned c = 0; c < 3; ++c)
    peak = add(peak, generated::ump2_adjoint_cpu_plan(c).numeric_bytes);
  // The native weighted derivative consumer has bounded recurrence scratch.
  peak = add(peak, posthf::source_scratch_bytes);
  if (peak > budget)
    throw std::length_error("complete UMP2 force endpoint exceeds numeric memory budget");
  auto w = initial_weights(ref, provider, threshold);
  auto gradient = rotation_gradient(ref, provider, w);
  for (unsigned spin = 0; spin < 2; ++spin) {
    const auto no = ref.nocc[spin];
    for (auto block : std::array<std::array<std::size_t, 2>, 2>{{{0, no}, {no, n}}})
      for (std::size_t p = block[0]; p < block[1]; ++p)
        for (std::size_t q = p + 1; q < block[1]; ++q) {
          const auto gap = ref.orbital_energies[spin][p] - ref.orbital_energies[spin][q];
          const auto g = gradient[spin][p * n + q] - gradient[spin][q * n + p];
          if (std::abs(gap) <= 1e-9) {
            if (std::abs(g) > 1e-9)
              throw std::runtime_error("nonstationary degenerate UMP2 orbital subspace");
          } else
            subtract_fock(w, spin, n, q, p, g / gap);
        }
  }
  gradient = rotation_gradient(ref, provider, w);
  Matrix rhs(dimension), diagonal(dimension);
  std::size_t offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    const auto no = ref.nocc[spin], nv = n - no;
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t a = 0; a < nv; ++a) {
        rhs[offset + i * nv + a] =
            gradient[spin][(no + a) * n + i] - gradient[spin][i * n + no + a];
        diagonal[offset + i * nv + a] =
            ref.orbital_energies[spin][no + a] - ref.orbital_energies[spin][i];
      }
    offset += no * nv;
  }
  response::GmresResult solved;
  solved.status = response::GmresStatus::initial_residual;
  if (problem) solved = response::solve_response(plan, *problem, rhs, {}, diagonal);
  if (!solved.converged()) throw std::runtime_error("UMP2 orbital response did not converge");
  offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    const auto no = ref.nocc[spin], nv = n - no;
    for (std::size_t i = 0; i < no; ++i) {
      w.one[spin][i * n + i] += 1.0;
      w.fock[spin][i * n + i] += 0.5;
      for (std::size_t a = 0; a < nv; ++a)
        subtract_fock(w, spin, n, no + a, i, solved.solution[offset + i * nv + a]);
    }
    offset += no * nv;
  }
  gradient = rotation_gradient(ref, provider, w);
  SpinMatrices overlap{Matrix(n2), Matrix(n2)};
  double residual = 0;
  for (unsigned spin = 0; spin < 2; ++spin)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        overlap[spin][p * n + q] = -0.25 * (gradient[spin][p * n + q] + gradient[spin][q * n + p]);
        residual =
            std::max(residual, std::abs(gradient[spin][p * n + q] - gradient[spin][q * n + p]));
      }
  if (!std::isfinite(residual) || residual > 1e-7)
    throw std::runtime_error("UMP2 relaxed Lagrangian is not stationary");
  ConventionalForceResult result;
  result.forces = derivative(ref, source.orbital(), w, overlap);
  result.response = std::move(solved);
  result.stationarity_residual = residual;
  result.planned_endpoint_peak_bytes = peak;
  result.derivative_workspace_bytes =
      add(posthf::source_scratch_bytes, mul(add(staging_elements, mul(4, n2)), sizeof(double)));
  result.weighted_eri_shell_tiles =
      mul(3, mul(mul(source.orbital().shells.size(), source.orbital().shells.size()),
                 mul(source.orbital().shells.size(), source.orbital().shells.size())));
  return result;
}
}  // namespace generativeqc::mp2
