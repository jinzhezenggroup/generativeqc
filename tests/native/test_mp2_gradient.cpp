#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_derivative_common.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "response/native_gmres.hpp"
#include "scf/density_fitting.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t eri_index(std::size_t n, std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

std::size_t g_index(std::size_t no, std::size_t nv, std::size_t i, std::size_t j, std::size_t a,
                    std::size_t b) {
  return ((i * no + j) * nv + a) * nv + b;
}

bool factorized_matches_dense(const generativeqc::mp2::FactorizedTwoElectronWeights& factorized,
                              std::span<const double> dense, double tolerance) {
  const auto n = factorized.orbitals;
  if (!generativeqc::mp2::valid_factorized_two_electron_weights(factorized) ||
      dense.size() != n * n * n * n || !std::isfinite(tolerance) || tolerance < 0.0)
    return false;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const double actual =
              generativeqc::mp2::factorized_two_electron_weight(factorized, p, q, r, s);
          const double expected = dense[eri_index(n, p, q, r, s)];
          if (!std::isfinite(actual) || !std::isfinite(expected) ||
              !(std::abs(actual - expected) <= tolerance))
            return false;
        }
  return true;
}

void factorized_comparison_rejects_nonfinite_oracles() {
  generativeqc::mp2::FactorizedTwoElectronWeights factors;
  factors.orbitals = 2;
  factors.occupied = 1;
  factors.fock.assign(4, 0.0);
  factors.correlation_iajb.assign(1, 0.0);
  std::vector<double> dense(16, 0.0);
  require(factorized_matches_dense(factors, dense, 0.0), "finite zero oracle did not match");
  for (const double invalid :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    for (std::size_t index = 0; index < dense.size(); ++index) {
      dense[index] = invalid;
      require(!factorized_matches_dense(factors, dense, 2e-11),
              "factorized comparison accepted a nonfinite dense oracle");
      dense[index] = 0.0;
    }
    require(!factorized_matches_dense(factors, dense, invalid),
            "factorized comparison accepted a nonfinite tolerance");
    factors.fock[0] = invalid;
    require(!factorized_matches_dense(factors, dense, 2e-11),
            "factorized comparison accepted a nonfinite compact input");
    factors.fock[0] = 0.0;
  }
  require(!factorized_matches_dense(factors, dense, -1.0),
          "factorized comparison accepted a negative tolerance");
  dense[0] = 1e-6;
  require(!factorized_matches_dense(factors, dense, 2e-11),
          "factorized comparison accepted a finite mismatch");
  require(factorized_matches_dense(factors, dense, 1e-6),
          "factorized comparison changed the inclusive tolerance boundary");
}

double mp2_energy(std::span<const double> g, std::span<const double> eps, std::size_t no) {
  const auto nv = eps.size() - no;
  double energy = 0.0;
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t j = 0; j < no; ++j)
      for (std::size_t a = 0; a < nv; ++a)
        for (std::size_t b = 0; b < nv; ++b) {
          const double direct = g[g_index(no, nv, i, j, a, b)];
          const double exchange = g[g_index(no, nv, i, j, b, a)];
          const double denominator = eps[i] + eps[j] - eps[no + a] - eps[no + b];
          energy += (2.0 * direct * direct - direct * exchange) / denominator;
        }
  return energy;
}

std::vector<double> symmetric_eri(std::size_t n) {
  std::vector<double> values(n * n * n * n);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const auto first = std::min(p, q) * n + std::max(p, q);
          const auto second = std::min(r, s) * n + std::max(r, s);
          const auto lo = std::min(first, second), hi = std::max(first, second);
          values[eri_index(n, p, q, r, s)] = 0.004 * (1 + lo + 3 * hi);
        }
  return values;
}

std::vector<double> fock(std::span<const double> h, std::span<const double> eri, std::size_t n,
                         std::size_t no) {
  std::vector<double> result(h.begin(), h.end());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t i = 0; i < no; ++i)
        result[p * n + q] += 2.0 * eri[eri_index(n, p, q, i, i)] - eri[eri_index(n, p, i, i, q)];
  return result;
}

std::vector<double> multiply(std::span<const double> left, std::span<const double> right,
                             std::size_t n) {
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k) result[i * n + j] += left[i * n + k] * right[k * n + j];
  return result;
}

std::vector<double> transpose(std::span<const double> matrix, std::size_t n) {
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) result[i * n + j] = matrix[j * n + i];
  return result;
}

std::vector<double> inverse(std::span<const double> matrix, std::size_t n) {
  std::vector<double> augmented(n * 2 * n);
  const auto stride = 2 * n;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) augmented[i * stride + j] = matrix[i * n + j];
    augmented[i * stride + n + i] = 1.0;
  }
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row)
      if (std::abs(augmented[row * stride + column]) > std::abs(augmented[pivot * stride + column]))
        pivot = row;
    require(std::abs(augmented[pivot * stride + column]) > 1e-14, "singular test matrix");
    for (std::size_t j = 0; j < stride; ++j)
      std::swap(augmented[column * stride + j], augmented[pivot * stride + j]);
    const double scale = augmented[column * stride + column];
    for (std::size_t j = 0; j < stride; ++j) augmented[column * stride + j] /= scale;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == column) continue;
      const double factor = augmented[row * stride + column];
      for (std::size_t j = 0; j < stride; ++j)
        augmented[row * stride + j] -= factor * augmented[column * stride + j];
    }
  }
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) result[i * n + j] = augmented[i * stride + n + j];
  return result;
}

std::vector<double> rotated_eri(std::span<const double> eri, std::span<const double> rotation,
                                std::size_t n) {
  std::vector<double> result(n * n * n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q)
              for (std::size_t r = 0; r < n; ++r)
                for (std::size_t s = 0; s < n; ++s)
                  result[eri_index(n, i, j, k, l)] += eri[eri_index(n, p, q, r, s)] *
                                                      rotation[p * n + i] * rotation[q * n + j] *
                                                      rotation[r * n + k] * rotation[s * n + l];
  return result;
}

void energy_adjoint_matches_independent_finite_difference() {
  constexpr std::size_t no = 1, nv = 2;
  const std::vector<double> g{0.12, -0.04, 0.07, 0.09};
  const std::vector<double> eps{-0.8, 0.2, 0.55};
  const std::vector<double> dg{0.3, -0.2, 0.5, 0.1};
  const std::vector<double> de{-0.4, 0.2, 0.6};
  const auto adjoint = generativeqc::mp2::canonical_energy_adjoint(g, eps, no, 1e-10);
  double reverse = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i) reverse += adjoint.integrals_iajb[i] * dg[i];
  for (std::size_t i = 0; i < eps.size(); ++i) reverse += adjoint.orbital_energies[i] * de[i];
  double previous = std::numeric_limits<double>::infinity();
  for (double step : {1e-3, 1e-4, 1e-5}) {
    auto plus_g = g, minus_g = g, plus_e = eps, minus_e = eps;
    for (std::size_t i = 0; i < g.size(); ++i) {
      plus_g[i] += step * dg[i];
      minus_g[i] -= step * dg[i];
    }
    for (std::size_t i = 0; i < eps.size(); ++i) {
      plus_e[i] += step * de[i];
      minus_e[i] -= step * de[i];
    }
    const double error = std::abs(
        (mp2_energy(plus_g, plus_e, no) - mp2_energy(minus_g, minus_e, no)) / (2.0 * step) -
        reverse);
    require(error < previous, "energy-adjoint finite difference did not improve");
    previous = error;
  }
  require(previous < 1e-9, "energy adjoint failed the independent directional derivative");
  const auto a01 = g_index(no, nv, 0, 0, 0, 1);
  const double d01 = eps[0] + eps[0] - eps[1] - eps[2];
  const double expected =
      (4.0 * g[a01] - g[g_index(no, nv, 0, 0, 1, 0)]) / d01 - g[g_index(no, nv, 0, 0, 1, 0)] / d01;
  require(std::abs(adjoint.integrals_iajb[a01] - expected) < 1e-14,
          "exchange cotangent was not transposed back and accumulated");
}

struct Fixture {
  std::size_t n{3}, no{1};
  std::vector<double> eps{-0.9, 0.25, 0.7};
  std::vector<double> eri{symmetric_eri(n)};
  std::vector<double> h;
  std::vector<double> g;

  Fixture() : h(n * n), g(no * no * (n - no) * (n - no)) {
    auto mean_field = fock(h, eri, n, no);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        h[p * n + q] = (p == q ? eps[p] : 0.0) - mean_field[p * n + q];
    for (std::size_t a = 0; a < n - no; ++a)
      for (std::size_t b = 0; b < n - no; ++b)
        g[g_index(no, n - no, 0, 0, a, b)] = eri[eri_index(n, 0, no + a, 0, no + b)];
  }
};

generativeqc::core::System h2() {
  generativeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<generativeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "H2 setup failed");
  return system;
}

void streamed_provider_matches_dense_oracle() {
  const auto system = h2();
  generativeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0.0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = generativeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "H2 reference did not converge");
  const auto& reference = *hf.reference;
  generativeqc::posthf::RawSource source(system);
  generativeqc::posthf::NativeBlockProvider provider(source, reference, 256ULL << 20, 1);
  std::vector<std::size_t> orbitals(reference.nbf);
  for (std::size_t p = 0; p < reference.nbf; ++p) orbitals[p] = p;
  const auto eri = provider.get({orbitals, orbitals, orbitals, orbitals});
  std::vector<double> hcore_mo(reference.nbf * reference.nbf);
  for (std::size_t p = 0; p < reference.nbf; ++p)
    for (std::size_t q = 0; q < reference.nbf; ++q)
      for (std::size_t mu = 0; mu < reference.nbf; ++mu)
        for (std::size_t nu = 0; nu < reference.nbf; ++nu)
          hcore_mo[p * reference.nbf + q] += reference.coefficients[mu * reference.nbf + p] *
                                             reference.hcore[mu * reference.nbf + nu] *
                                             reference.coefficients[nu * reference.nbf + q];
  const auto virtuals = reference.nbf - reference.nocc;
  std::vector<double> g(reference.nocc * reference.nocc * virtuals * virtuals);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t j = 0; j < reference.nocc; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          g[g_index(reference.nocc, virtuals, i, j, a, b)] =
              eri[eri_index(reference.nbf, i, reference.nocc + a, j, reference.nocc + b)];
  const auto adjoint = generativeqc::mp2::canonical_energy_adjoint(g, reference.orbital_energies,
                                                                   reference.nocc, 1e-10);
  const auto dense = generativeqc::mp2::canonical_orbital_rhs(hcore_mo, eri, adjoint, 1e-10);
  auto streamed = generativeqc::mp2::canonical_orbital_rhs_streamed(reference, hcore_mo, provider,
                                                                    adjoint, 1e-10);
  auto close = [](std::span<const double> first, std::span<const double> second) {
    if (first.size() != second.size()) return false;
    for (std::size_t i = 0; i < first.size(); ++i)
      if (!std::isfinite(first[i]) || !std::isfinite(second[i]) ||
          !(std::abs(first[i] - second[i]) <= 2e-11))
        return false;
    return true;
  };
  generativeqc::mp2::FactorizedTwoElectronWeights orbital_factors;
  orbital_factors.orbitals = reference.nbf;
  orbital_factors.occupied = reference.nocc;
  orbital_factors.fock = streamed.fock_weights;
  orbital_factors.correlation_iajb = adjoint.integrals_iajb;
  require(close(streamed.energy_gradient, dense.energy_gradient) &&
              close(streamed.response_rhs, dense.response_rhs) &&
              close(streamed.one_electron, dense.one_electron) && streamed.two_electron.empty() &&
              factorized_matches_dense(orbital_factors, dense.two_electron, 2e-11),
          "streamed native provider factorized weights differ from the dense oracle");
  const double response_denominator =
      reference.orbital_energies[reference.nocc] - reference.orbital_energies[0] +
      4.0 * eri[eri_index(reference.nbf, reference.nocc, 0, reference.nocc, 0)] -
      eri[eri_index(reference.nbf, reference.nocc, reference.nocc, 0, 0)] -
      eri[eri_index(reference.nbf, reference.nocc, 0, 0, reference.nocc)];
  const std::array<double, 1> response{streamed.response_rhs[0] / response_denominator};
  const auto dense_weights =
      generativeqc::mp2::canonical_lagrangian_weights(hcore_mo, eri, adjoint, response, 1e-10);
  const auto streamed_weights = generativeqc::mp2::canonical_lagrangian_weights_streamed(
      reference, hcore_mo, provider, adjoint, std::move(streamed), response, 1e-10);
  require(close(streamed_weights.one_electron, dense_weights.one_electron) &&
              streamed_weights.two_electron.empty() &&
              factorized_matches_dense(streamed_weights.two_electron_factors,
                                       dense_weights.two_electron, 2e-11) &&
              close(streamed_weights.overlap, dense_weights.overlap) &&
              std::abs(streamed_weights.stationarity_residual -
                       dense_weights.stationarity_residual) < 2e-11,
          "streamed factorized relaxed weights differ from the dense oracle");
  auto stale = reference;
  bool rejected = false;
  try {
    (void)generativeqc::mp2::canonical_orbital_rhs_streamed(stale, hcore_mo, provider, adjoint,
                                                            1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "streamed provider accepted a copied/stale reference owner");
#if !GENERATIVEQC_HAS_CUDA
  rejected = false;
  try {
    (void)generativeqc::mp2::canonical_orbital_rhs_streamed(reference, hcore_mo, provider, adjoint,
                                                            1e-10, true, 7);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "streamed orbital response ignored the requested CUDA backend");
#endif
}

void density_fitted_provider_matches_dense_ri_oracle() {
  const auto system = h2();
  const auto auxiliary = system;
  constexpr double metric_threshold = 1.0e-10;
  generativeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0.0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.density_fitting_relative_threshold = metric_threshold;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = generativeqc::scf::run_rhf_density_fitting(system, auxiliary, options);
  require(hf.converged && hf.reference, "H2 RI reference did not converge");
  const auto& reference = *hf.reference;
  generativeqc::posthf::RawSource source(system, &auxiliary);
  generativeqc::posthf::DensityFittedBlockProvider provider(source, reference, 256ULL << 20,
                                                            metric_threshold);
  const auto n = reference.nbf, na = provider.auxiliary_count();
  std::vector<std::size_t> orbitals(n);
  for (std::size_t p = 0; p < n; ++p) orbitals[p] = p;
  const auto actual_eri = provider.get({orbitals, orbitals, orbitals, orbitals});

  // Independent value-side RI reconstruction from public AO A/M tensors.
  const auto raw =
      generativeqc::integrals::build_density_fitting_integrals(system, auxiliary, false);
  const auto factor =
      generativeqc::scf::factor_density_fitting_metric(raw.metric, na, metric_threshold);
  std::vector<double> transformed(n * n * na), whitened(n * n * na), expected_eri(n * n * n * n);
  auto three = [n, na](std::size_t p, std::size_t q, std::size_t P) {
    return (p * n + q) * na + P;
  };
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          for (std::size_t P = 0; P < na; ++P)
            transformed[three(p, q, P)] += reference.coefficients[mu * n + p] *
                                           reference.coefficients[nu * n + q] *
                                           raw.three_center[(mu * n + nu) * na + P];
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t Q = 0; Q < na; ++Q)
        for (std::size_t P = 0; P < na; ++P)
          whitened[three(p, q, Q)] +=
              transformed[three(p, q, P)] * factor.inverse_square_root[P * na + Q];
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t t = 0; t < n; ++t)
          for (std::size_t Q = 0; Q < na; ++Q)
            expected_eri[eri_index(n, p, q, r, t)] +=
                whitened[three(p, q, Q)] * whitened[three(r, t, Q)];
  auto close = [](std::span<const double> first, std::span<const double> second, double tolerance) {
    if (first.size() != second.size() || !std::isfinite(tolerance) || tolerance < 0.0) return false;
    for (std::size_t i = 0; i < first.size(); ++i)
      if (!std::isfinite(first[i]) || !std::isfinite(second[i]) ||
          !(std::abs(first[i] - second[i]) <= tolerance))
        return false;
    return true;
  };
  require(close(actual_eri, expected_eri, 2e-12),
          "density-fitted MO provider differs from independent RI reconstruction");
  require(close(provider.metric(), raw.metric, 0.0) &&
              close(provider.inverse_square_root(), factor.inverse_square_root, 2e-14),
          "density-fitted MO provider lost its metric branch identity");

  std::vector<double> hcore_mo(n * n);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          hcore_mo[p * n + q] += reference.coefficients[mu * n + p] * reference.hcore[mu * n + nu] *
                                 reference.coefficients[nu * n + q];
  const auto nv = n - reference.nocc;
  std::vector<double> g(reference.nocc * reference.nocc * nv * nv);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t j = 0; j < reference.nocc; ++j)
      for (std::size_t a = 0; a < nv; ++a)
        for (std::size_t b = 0; b < nv; ++b)
          g[g_index(reference.nocc, nv, i, j, a, b)] =
              expected_eri[eri_index(n, i, reference.nocc + a, j, reference.nocc + b)];
  const auto adjoint = generativeqc::mp2::canonical_energy_adjoint(g, reference.orbital_energies,
                                                                   reference.nocc, 1e-10);
  const auto dense =
      generativeqc::mp2::canonical_orbital_rhs(hcore_mo, expected_eri, adjoint, 1e-10);
  auto streamed = generativeqc::mp2::canonical_orbital_rhs_streamed(reference, hcore_mo, provider,
                                                                    adjoint, 1e-10);
  generativeqc::mp2::FactorizedTwoElectronWeights ri_orbital_factors;
  ri_orbital_factors.orbitals = n;
  ri_orbital_factors.occupied = reference.nocc;
  ri_orbital_factors.fock = streamed.fock_weights;
  ri_orbital_factors.correlation_iajb = adjoint.integrals_iajb;
  require(close(streamed.energy_gradient, dense.energy_gradient, 2e-10) &&
              close(streamed.response_rhs, dense.response_rhs, 2e-10) &&
              close(streamed.one_electron, dense.one_electron, 2e-10) &&
              streamed.two_electron.empty() &&
              factorized_matches_dense(ri_orbital_factors, dense.two_electron, 2e-10),
          "RI provider cannot drive the factorized streamed MP2 orbital response");

  // H2/STO-3G has one occupied and one virtual MO, so solve the response scalar
  // independently and compare the complete relaxed-weight construction.
  const double denominator =
      reference.orbital_energies[reference.nocc] - reference.orbital_energies[0] +
      4.0 * expected_eri[eri_index(n, reference.nocc, 0, reference.nocc, 0)] -
      expected_eri[eri_index(n, reference.nocc, reference.nocc, 0, 0)] -
      expected_eri[eri_index(n, reference.nocc, 0, 0, reference.nocc)];
  const std::array<double, 1> response{streamed.response_rhs[0] / denominator};
  const auto dense_weights = generativeqc::mp2::canonical_lagrangian_weights(
      hcore_mo, expected_eri, adjoint, response, 1e-10);
  const auto streamed_weights = generativeqc::mp2::canonical_lagrangian_weights_streamed(
      reference, hcore_mo, provider, adjoint, std::move(streamed), response, 1e-10);
  require(close(streamed_weights.one_electron, dense_weights.one_electron, 2e-10) &&
              streamed_weights.two_electron.empty() &&
              factorized_matches_dense(streamed_weights.two_electron_factors,
                                       dense_weights.two_electron, 2e-10) &&
              close(streamed_weights.overlap, dense_weights.overlap, 2e-10) &&
              std::abs(streamed_weights.stationarity_residual -
                       dense_weights.stationarity_residual) < 2e-10,
          "RI provider cannot drive the factorized relaxed MP2 Lagrangian");

  const auto raw_weights = generativeqc::mp2::density_fitted_lagrangian_weights(
      reference, provider, streamed_weights, 256ULL << 20);
  require(raw_weights.orbitals == n && raw_weights.auxiliary == na &&
              raw_weights.three_center.size() == n * n * na && raw_weights.metric.size() == na * na,
          "RI reverse returned inconsistent raw-weight dimensions");

  // Independent directional derivative of the relaxed two-electron functional
  // with respect to raw A and M. This simultaneously exercises A->B reverse
  // composition and the shared #466 M^(-1/2) pullback.
  auto ri_two_functional = [&](std::span<const double> raw_a, std::span<const double> metric) {
    const auto local_factor = generativeqc::scf::factor_density_fitting_metric(
        std::vector<double>(metric.begin(), metric.end()), na, metric_threshold);
    std::vector<double> local_transformed(n * n * na), local_whitened(n * n * na);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t mu = 0; mu < n; ++mu)
          for (std::size_t nu = 0; nu < n; ++nu)
            for (std::size_t P = 0; P < na; ++P)
              local_transformed[three(p, q, P)] += reference.coefficients[mu * n + p] *
                                                   reference.coefficients[nu * n + q] *
                                                   raw_a[(mu * n + nu) * na + P];
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t Q = 0; Q < na; ++Q)
          for (std::size_t P = 0; P < na; ++P)
            local_whitened[three(p, q, Q)] +=
                local_transformed[three(p, q, P)] * local_factor.inverse_square_root[P * na + Q];
    double value = 0.0;
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t t = 0; t < n; ++t) {
            double eri = 0.0;
            for (std::size_t Q = 0; Q < na; ++Q)
              eri += local_whitened[three(p, q, Q)] * local_whitened[three(r, t, Q)];
            value += generativeqc::mp2::two_electron_weight(streamed_weights, p, q, r, t) * eri;
          }
    return value;
  };
  std::vector<double> d_a(raw.three_center.size()), d_m(raw.metric.size());
  for (std::size_t i = 0; i < d_a.size(); ++i) d_a[i] = 0.013 * (static_cast<double>(i % 7) - 2.5);
  for (std::size_t P = 0; P < na; ++P)
    for (std::size_t Q = P; Q < na; ++Q) {
      const double value = 0.009 * (1.0 + P + 2.0 * Q);
      d_m[P * na + Q] = d_m[Q * na + P] = value;
    }
  double reverse_dot = 0.0;
  for (std::size_t i = 0; i < d_a.size(); ++i) reverse_dot += raw_weights.three_center[i] * d_a[i];
  for (std::size_t i = 0; i < d_m.size(); ++i) reverse_dot += raw_weights.metric[i] * d_m[i];
  double previous_error = std::numeric_limits<double>::infinity();
  for (const double step : {1e-3, 2e-4, 4e-5}) {
    auto plus_a = raw.three_center, minus_a = raw.three_center;
    auto plus_m = raw.metric, minus_m = raw.metric;
    for (std::size_t i = 0; i < d_a.size(); ++i) {
      plus_a[i] += step * d_a[i];
      minus_a[i] -= step * d_a[i];
    }
    for (std::size_t i = 0; i < d_m.size(); ++i) {
      plus_m[i] += step * d_m[i];
      minus_m[i] -= step * d_m[i];
    }
    const double plus_value = ri_two_functional(plus_a, plus_m);
    const double minus_value = ri_two_functional(minus_a, minus_m);
    const double finite = (plus_value - minus_value) / (2 * step);
    const double error = std::abs(finite - reverse_dot);
    // Below the subtraction roundoff floor, a smaller step can increase the
    // error. Keep the independent absolute acceptance gate below unchanged.
    const double roundoff_floor = 64 * std::numeric_limits<double>::epsilon() *
                                  (std::abs(plus_value) + std::abs(minus_value)) / (2 * step);
    require(error <= previous_error + roundoff_floor,
            "RI raw-weight finite difference did not improve above its roundoff floor");
    previous_error = error;
  }
  require(previous_error < 2e-8,
          "RI raw A/M weights do not match the independent finite-difference functional");

  const auto direct_df = generativeqc::integrals::contract_weighted_density_fitting_derivative(
      system, auxiliary, raw_weights.three_center, raw_weights.metric, 256ULL << 20);
  const auto dense_df =
      generativeqc::integrals::build_density_fitting_integrals(system, auxiliary, true);
  std::vector<double> dense_df_contraction(dense_df.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < dense_df.ncoord; ++coordinate) {
    for (std::size_t i = 0; i < raw_weights.three_center.size(); ++i)
      dense_df_contraction[coordinate] +=
          dense_df.three_center_derivative[coordinate * raw_weights.three_center.size() + i] *
          raw_weights.three_center[i];
    for (std::size_t i = 0; i < raw_weights.metric.size(); ++i)
      dense_df_contraction[coordinate] +=
          dense_df.metric_derivative[coordinate * raw_weights.metric.size() + i] *
          raw_weights.metric[i];
  }
  require(close(direct_df, dense_df_contraction, 2e-11),
          "direct weighted DF derivative contraction differs from the dense derivative oracle");
  bool derivative_budget_rejected = false;
  try {
    (void)generativeqc::integrals::contract_weighted_density_fitting_derivative(
        system, auxiliary, raw_weights.three_center, raw_weights.metric, 1);
  } catch (const std::length_error&) {
    derivative_budget_rejected = true;
  }
  require(derivative_budget_rejected, "direct weighted DF derivative ignored its memory budget");

  bool budget_rejected = false;
  try {
    (void)generativeqc::mp2::density_fitted_lagrangian_weights(reference, provider,
                                                               streamed_weights, 1);
  } catch (const std::length_error&) {
    budget_rejected = true;
  }
  require(budget_rejected, "RI Lagrangian reverse ignored its memory budget");

  bool rejected = false;
  try {
    (void)provider.get({orbitals, orbitals, orbitals, orbitals}, true, 0);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "CPU RI provider silently accepted a CUDA request");
}

double rotated_mp2_energy(const Fixture& fixture, std::span<const double> direction, double step) {
  const auto n = fixture.n, no = fixture.no;
  std::vector<double> generator(n * n), left(n * n), right(n * n);
  for (std::size_t p = 0; p < n; ++p) left[p * n + p] = right[p * n + p] = 1.0;
  for (std::size_t a = 0; a < n - no; ++a) {
    generator[a + no] = direction[a];
    generator[(a + no) * n] = -direction[a];
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    left[i] -= 0.5 * step * generator[i];
    right[i] += 0.5 * step * generator[i];
  }
  const auto rotation = multiply(inverse(left, n), right, n);
  const auto transformed_h = multiply(multiply(transpose(rotation, n), fixture.h, n), rotation, n);
  const auto transformed_eri = rotated_eri(fixture.eri, rotation, n);
  const auto transformed_fock = fock(transformed_h, transformed_eri, n, no);
  std::vector<double> eps(n), g((n - no) * (n - no));
  for (std::size_t p = 0; p < n; ++p) eps[p] = transformed_fock[p * n + p];
  for (std::size_t a = 0; a < n - no; ++a)
    for (std::size_t b = 0; b < n - no; ++b)
      g[g_index(no, n - no, 0, 0, a, b)] = transformed_eri[eri_index(n, 0, no + a, 0, no + b)];
  return mp2_energy(g, eps, no);
}

void orbital_rhs_and_relaxed_weights_match_independent_oracles() {
  const Fixture fixture;
  const auto adjoint =
      generativeqc::mp2::canonical_energy_adjoint(fixture.g, fixture.eps, fixture.no, 1e-10);
  const auto orbital =
      generativeqc::mp2::canonical_orbital_rhs(fixture.h, fixture.eri, adjoint, 1e-10);
  const std::array<double, 2> direction{0.31, -0.27};
  double reverse = 0.0;
  for (std::size_t i = 0; i < direction.size(); ++i)
    reverse += orbital.energy_gradient[i] * direction[i];
  std::array<double, 3> errors{};
  std::size_t error_index = 0;
  for (double step : {1e-3, 1e-4, 1e-5}) {
    const double finite = (rotated_mp2_energy(fixture, direction, step) -
                           rotated_mp2_energy(fixture, direction, -step)) /
                          (2.0 * step);
    const double error = std::abs(finite - reverse);
    errors[error_index++] = error;
  }
  require(errors[1] < errors[0] && errors[1] < 1e-10 && errors[2] < 1e-10,
          "orbital gradient has the wrong sign or missing terms");

  const auto nv = fixture.n - fixture.no;
  std::vector<double> response_matrix(nv * nv);
  for (std::size_t a = 0; a < nv; ++a)
    for (std::size_t b = 0; b < nv; ++b)
      response_matrix[a * nv + b] =
          (fixture.eps[fixture.no + a] - fixture.eps[0]) * (a == b) +
          4.0 * fixture.eri[eri_index(fixture.n, fixture.no + a, 0, fixture.no + b, 0)] -
          fixture.eri[eri_index(fixture.n, fixture.no + a, fixture.no + b, 0, 0)] -
          fixture.eri[eri_index(fixture.n, fixture.no + a, 0, 0, fixture.no + b)];
  const auto response_inverse = inverse(response_matrix, nv);
  std::vector<double> z(nv);
  for (std::size_t a = 0; a < nv; ++a)
    for (std::size_t b = 0; b < nv; ++b)
      z[a] += response_inverse[a * nv + b] * orbital.response_rhs[b];
  const auto weights =
      generativeqc::mp2::canonical_lagrangian_weights(fixture.h, fixture.eri, adjoint, z, 1e-10);
  require(weights.stationarity_residual < 1e-10,
          "relaxed weights are not stationary with the independent Z-vector");
  for (std::size_t p = 0; p < fixture.n; ++p)
    for (std::size_t q = 0; q < fixture.n; ++q)
      require(
          std::abs(weights.overlap[p * fixture.n + q] - weights.overlap[q * fixture.n + p]) < 1e-14,
          "overlap weight is not symmetric");
}

void factorized_consumer_population_guards() {
  auto system = h2();
  system.shells.push_back({0, 0, {{0.13, 1.0}}});
  system.shells.push_back({1, 0, {{0.17, 1.0}}});
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "factorized population fixture is invalid");
  generativeqc::hf::PhysicalReference reference;
  reference.nbf = generativeqc::molecule::ao_count(system);
  reference.nocc = 1;
  const auto n = reference.nbf;
  reference.coefficients.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) reference.coefficients[i * n + i] = 1.0;
  generativeqc::mp2::LagrangianWeights weights;
  weights.orbitals = n;
  weights.occupied = 1;
  weights.one_electron.assign(n * n, 0.0);
  weights.overlap.assign(n * n, 0.0);
  for (unsigned mismatch = 0; mismatch < 3; ++mismatch) {
    auto& factors = weights.two_electron_factors;
    factors.orbitals = n + (mismatch == 1);
    factors.occupied = mismatch == 2 ? 2 : 1;
    const auto fn = factors.orbitals, fo = factors.occupied, fv = fn - fo;
    factors.fock.assign(fn * fn, 0.0);
    factors.correlation_iajb.assign(fo * fo * fv * fv, 0.0);
    require(generativeqc::mp2::valid_factorized_two_electron_weights(factors),
            "factor fixture must be internally valid");
    bool rejected = false, called = false;
    try {
      (void)generativeqc::mp2::detail::conventional_derivative(
          system, reference, weights,
          [&](std::span<const double>, std::span<const double>) {
            called = true;
            return std::vector<double>(3 * system.atoms.size(), 0.0);
          },
          [&](const std::array<std::size_t, 4>&, std::span<const double>) {
            called = true;
            return std::array<double, 12>{};
          });
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(mismatch ? rejected && !called : !rejected && called,
            "conventional factor population mismatch was not rejected before contraction");
    rejected = false;
    try {
      (void)generativeqc::mp2::two_electron_weight(weights, 0, 0, 0, 0);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected == (mismatch != 0), "factor lookup accepted a different population");
  }
}

void invalid_inputs_and_resource_boundaries() {
  bool rejected = false;
  const std::array<double, 1> unit_integral{1.0};
  const std::array<double, 2> zero_gap{0.0, 0.0};
  try {
    (void)generativeqc::mp2::canonical_energy_adjoint(unit_integral, zero_gap, 1, 1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "near-zero MP2 denominator was accepted");
  rejected = false;
  const std::array<double, 1> nonfinite_integral{std::numeric_limits<double>::quiet_NaN()};
  const std::array<double, 2> separated_energies{-1.0, 1.0};
  try {
    (void)generativeqc::mp2::canonical_energy_adjoint(nonfinite_integral, separated_energies, 1,
                                                      1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nonfinite MP2 integral was accepted");

  auto options = generativeqc::response::GmresOptions{};
  options.restart = 3;
  options.max_iterations = 10;
  const auto response = generativeqc::response::prepare_gmres(4, options);
  const auto probe = generativeqc::mp2::conventional_gradient_plan(
      4, 2, 4096, response, 3, 9, 9 * sizeof(double),
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
  require(probe.peak_bytes > probe.response_bytes && probe.shell_cotangent_bytes > 0,
          "gradient resource plan omitted a simultaneous owner");
  const auto expected_relaxed = (4 * 4 * 4 + 2 * 2 * 2) * sizeof(double);
  require(probe.relaxed_weight_bytes == expected_relaxed,
          "gradient resource plan retained dense N^4 relaxed-weight storage");
  require(probe.relaxed_weight_bytes < 4 * 4 * 4 * 4 * sizeof(double),
          "factorized relaxed-weight storage is not below one dense N^4 tensor");
  require(probe.shell_cotangent_bytes == 81 * sizeof(double),
          "shell-quartet cotangent ownership is not isolated");
  require(probe.derivative_staging_bytes == 485 * sizeof(double),
          "derivative staging double-counts the shell-quartet cotangent");
  const auto cuda_probe = generativeqc::mp2::conventional_gradient_plan(
      4, 2, 4096, response, 3, 9, 9 * sizeof(double),
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()), 12345);
  require(cuda_probe.derivative_backend_staging_bytes == 12345,
          "gradient resource plan omitted CUDA consumer staging");
  require(cuda_probe.peak_bytes == probe.peak_bytes + 12345,
          "CUDA consumer staging was not charged exactly once");
  const auto exact = generativeqc::mp2::conventional_gradient_plan(
      4, 2, 4096, response, 3, 9, 9 * sizeof(double), probe.peak_bytes);
  require(exact.peak_bytes == probe.peak_bytes, "exact resource budget changed the plan");
  rejected = false;
  try {
    (void)generativeqc::mp2::conventional_gradient_plan(4, 2, 4096, response, 3, 9,
                                                        9 * sizeof(double), probe.peak_bytes - 1);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "one-byte-short gradient budget was accepted");
  rejected = false;
  try {
    (void)generativeqc::mp2::conventional_gradient_plan(
        4, 2, 1, response, std::numeric_limits<std::size_t>::max(), 3, 24,
        std::numeric_limits<std::size_t>::max());
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "gradient resource arithmetic overflow was accepted");
}
}  // namespace

int main() {
  try {
    factorized_comparison_rejects_nonfinite_oracles();
    energy_adjoint_matches_independent_finite_difference();
    orbital_rhs_and_relaxed_weights_match_independent_oracles();
    streamed_provider_matches_dense_oracle();
    density_fitted_provider_matches_dense_ri_oracle();
    factorized_consumer_population_guards();
    invalid_inputs_and_resource_boundaries();
    std::cout << "MP2 native gradient contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
