#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "d4_eeq_oracle_fixtures.hpp"
#include "dft/dispersion/d4_eeq_response.hpp"
#include "dft/dispersion/d4_second_order.hpp"

using namespace d4_eeq_tests;

namespace {
bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

int oracle_cases() {
  for (const auto& f : kEEQOracleFixtures) {
    std::vector<double> workspace(complete_d4_eeq_workspace_elements(f.atoms));
    std::vector<double> gradient(3 * f.atoms), charges(f.atoms);
    double energy[2]{};
    const auto status = evaluate_complete_d4_eeq(
        f.atoms, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, f.profile,
        workspace.data(), workspace.size(), energy, gradient.data(), charges.data());
    if (status != D4Status::success) {
      std::fprintf(stderr, "%s: status=%d\n", f.name, static_cast<int>(status));
      return 1;
    }
    if (!near(energy[0] + energy[1], f.energy, 2e-13)) {
      std::fprintf(stderr, "%s: energy %.17g oracle %.17g\n", f.name, energy[0] + energy[1],
                   f.energy);
      return 2;
    }
    double charge_sum = 0.0;
    for (int i = 0; i < f.atoms; ++i) {
      charge_sum += charges[i];
      if (!near(charges[i], f.charges[i], 1e-10)) {
        std::fprintf(stderr, "%s: charge[%d] %.17g oracle %.17g\n", f.name, i, charges[i],
                     f.charges[i]);
        return 3;
      }
    }
    if (!near(charge_sum, f.total_charge, 2e-13)) return 4;
    for (int i = 0; i < 3 * f.atoms; ++i)
      if (!near(gradient[i], f.gradient[i], 2e-12)) {
        std::fprintf(stderr, "%s: gradient[%d] %.17g oracle %.17g\n", f.name, i, gradient[i],
                     f.gradient[i]);
        return 5;
      }
  }
  return 0;
}

int response_finite_differences() {
  const auto& f = kEEQOracleFixtures.front();
  const int n = f.atoms;
  std::vector<double> workspace(eeq2019_workspace_elements(n));
  std::vector<double> charges(n), dqdr(3 * n * n);
  if (evaluate_eeq2019(n, f.z.data(), f.xyz.data(), f.total_charge, workspace.data(),
                       workspace.size(), charges.data(), dqdr.data()) != D4Status::success)
    return 10;
  for (int c = 0; c < 3 * n; ++c) {
    double response_sum = 0.0;
    for (int i = 0; i < n; ++i) response_sum += dqdr[static_cast<std::size_t>(c) * n + i];
    if (std::fabs(response_sum) > 2e-13) return 11;
  }

  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int c = 0; c < 3 * n; ++c) {
      auto plus = f.xyz;
      auto minus = f.xyz;
      plus[c] += h;
      minus[c] -= h;
      std::vector<double> wp(workspace.size()), wm(workspace.size());
      std::vector<double> qp(n), qm(n), dp(dqdr.size()), dm(dqdr.size());
      if (evaluate_eeq2019(n, f.z.data(), plus.data(), f.total_charge, wp.data(), wp.size(),
                           qp.data(), dp.data()) != D4Status::success ||
          evaluate_eeq2019(n, f.z.data(), minus.data(), f.total_charge, wm.data(), wm.size(),
                           qm.data(), dm.data()) != D4Status::success)
        return 12;
      for (int i = 0; i < n; ++i) {
        const double numerical = (qp[i] - qm[i]) / (2 * h);
        maximum =
            std::max(maximum, std::fabs(numerical - dqdr[static_cast<std::size_t>(c) * n + i]));
      }
    }
    if (maximum > 2e-8) {
      std::fprintf(stderr, "dq/dR finite-difference error at h=%g: %.3e\n", h, maximum);
      return 13;
    }
  }
  return 0;
}

int complete_gradient_finite_difference() {
  const auto& f = kEEQOracleFixtures.front();
  const int n = f.atoms;
  std::vector<double> workspace(complete_d4_eeq_workspace_elements(n));
  std::vector<double> gradient(3 * n), charges(n);
  double energy[2]{};
  if (evaluate_complete_d4_eeq(n, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, f.profile,
                               workspace.data(), workspace.size(), energy, gradient.data(),
                               charges.data()) != D4Status::success)
    return 20;
  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int c = 0; c < 3 * n; ++c) {
      auto plus = f.xyz;
      auto minus = f.xyz;
      plus[c] += h;
      minus[c] -= h;
      std::vector<double> wp(workspace.size()), wm(workspace.size());
      std::vector<double> gp(3 * n), gm(3 * n), qp(n), qm(n);
      double ep[2]{}, em[2]{};
      if (evaluate_complete_d4_eeq(n, f.z.data(), plus.data(), f.total_charge, f.parameters,
                                   f.profile, wp.data(), wp.size(), ep, gp.data(),
                                   qp.data()) != D4Status::success ||
          evaluate_complete_d4_eeq(n, f.z.data(), minus.data(), f.total_charge, f.parameters,
                                   f.profile, wm.data(), wm.size(), em, gm.data(),
                                   qm.data()) != D4Status::success)
        return 21;
      const double numerical = ((ep[0] + ep[1]) - (em[0] + em[1])) / (2 * h);
      maximum = std::max(maximum, std::fabs(numerical - gradient[c]));
    }
    if (maximum > 2e-9) {
      std::fprintf(stderr, "complete D4 gradient finite-difference error at h=%g: %.3e\n", h,
                   maximum);
      return 22;
    }
  }
  return 0;
}

int directional_charge_pullback() {
  // E = 1/2 sum_i (1 + i/4) q_i(R)^2. This exercises both the geometric
  // change of the charge Jacobian and the moving charge cotangent.
  double largest_error = 0;
  for (const auto& fixture : kEEQOracleFixtures) {
    const int atoms = fixture.atoms;
    const int coordinates = 3 * atoms;
    std::vector<double> workspace(eeq_direction_workspace_elements(atoms));
    std::vector<double> direction(coordinates), gradient(coordinates, 19), action(coordinates, 23);
    for (int coordinate = 0; coordinate < coordinates; ++coordinate)
      direction[coordinate] = 0.3 * std::sin(0.23 + 0.65 * coordinate);
    const auto seed = [=](const double* charges, const double* response, double* cotangent,
                          double* tangent) {
      double charge_sum = 0, response_sum = 0;
      for (int atom = 0; atom < atoms; ++atom) {
        charge_sum += charges[atom];
        response_sum += response[atom];
        cotangent[atom] = (1 + 0.25 * atom) * charges[atom];
        tangent[atom] = (1 + 0.25 * atom) * response[atom];
      }
      return near(charge_sum, fixture.total_charge, 2e-13) && near(response_sum, 0, 2e-13)
                 ? D4Status::success
                 : D4Status::numerical_failure;
    };
    const auto evaluate = [&](std::size_t capacity) {
      return eeq_directional_pullback(
          atoms, fixture.z.data(), fixture.xyz.data(), fixture.total_charge, eeq2019_host_tables(),
          direction.data(), workspace.data(), capacity, seed, gradient.data(), action.data());
    };
    if (evaluate(workspace.size() - 1) != D4Status::invalid_argument ||
        gradient != std::vector<double>(coordinates, 19) ||
        action != std::vector<double>(coordinates, 23) ||
        evaluate(workspace.size()) != D4Status::success)
      return 40;
    // Use the independent full-Jacobian transport, not the adjoint, at displaced
    // geometries. Its first-order charge/gradient provenance is gated above.
    const auto reference = [&](double step, std::vector<double>& result) {
      auto xyz = fixture.xyz;
      for (int coordinate = 0; coordinate < coordinates; ++coordinate)
        xyz[coordinate] += step * direction[coordinate];
      std::vector<double> scratch(eeq2019_workspace_elements(atoms));
      std::vector<double> charges(atoms), jacobian(coordinates * atoms);
      if (evaluate_eeq2019(atoms, fixture.z.data(), xyz.data(), fixture.total_charge,
                           scratch.data(), scratch.size(), charges.data(),
                           jacobian.data()) != D4Status::success)
        return false;
      std::fill(result.begin(), result.end(), 0);
      for (int coordinate = 0; coordinate < coordinates; ++coordinate)
        for (int atom = 0; atom < atoms; ++atom)
          result[coordinate] +=
              jacobian[coordinate * atoms + atom] * (1 + 0.25 * atom) * charges[atom];
      return true;
    };
    std::vector<double> plus(coordinates), minus(coordinates);
    if (!reference(0, plus)) return 41;
    for (int coordinate = 0; coordinate < coordinates; ++coordinate)
      if (!near(gradient[coordinate], plus[coordinate], 2e-13)) return 42;
    for (const double step : {1e-4, 5e-5}) {
      if (!reference(step, plus) || !reference(-step, minus)) return 43;
      for (int coordinate = 0; coordinate < coordinates; ++coordinate) {
        const double expected = (plus[coordinate] - minus[coordinate]) / (2 * step);
        if (!near(action[coordinate], expected, 1e-8)) return 44;
        largest_error = std::max(largest_error, std::fabs(action[coordinate] - expected));
      }
    }
    std::fill(direction.begin(), direction.end(), 0.125);
    if (evaluate(workspace.size()) != D4Status::success ||
        !std::all_of(action.begin(), action.end(), [](double value) { return value == 0; }))
      return 45;
  }
  std::printf("EEQ charge pullback: max directional error %.3e\n", largest_error);
  return 0;
}

int complete_hessian_action() {
  double largest_error = 0;
  for (const auto& fixture : kEEQOracleFixtures) {
    const auto atoms = fixture.atoms;
    const auto coordinates = 3 * atoms;
    std::vector<double> scratch(d4_eeq_direction_workspace_elements(atoms));
    std::vector<double> direction(coordinates), gradient(coordinates), action(coordinates);
    const auto run = [&] {
      return d4_eeq_direction(atoms, fixture.z.data(), fixture.xyz.data(), direction.data(),
                              fixture.total_charge, fixture.parameters,
                              eeq_d4_host_tables(fixture.profile), eeq2019_host_tables(),
                              scratch.data(), scratch.size(), gradient.data(), action.data());
    };
    for (int coordinate = 0; coordinate < coordinates; ++coordinate)
      direction[coordinate] = 0.37 * std::sin(0.13 + 0.71 * coordinate);
    if (run() != D4Status::success) {
      std::fprintf(stderr, "%s: complete D4 Hessian action failed\n", fixture.name);
      return 50;
    }
    for (int coordinate = 0; coordinate < coordinates; ++coordinate)
      if (!near(gradient[coordinate], fixture.gradient[coordinate], 2e-12)) return 51;
    std::vector<double> reference_workspace(complete_d4_eeq_workspace_elements(atoms));
    std::vector<double> charges(atoms), plus(coordinates), minus(coordinates);
    const auto reference = [&](double step, std::vector<double>& output) {
      auto xyz = fixture.xyz;
      for (int coordinate = 0; coordinate < coordinates; ++coordinate)
        xyz[coordinate] += step * direction[coordinate];
      double energy[2];
      return evaluate_complete_d4_eeq(atoms, fixture.z.data(), xyz.data(), fixture.total_charge,
                                      fixture.parameters, fixture.profile,
                                      reference_workspace.data(), reference_workspace.size(),
                                      energy, output.data(), charges.data());
    };
    for (const double step : {1e-4, 5e-5}) {
      if (reference(step, plus) != D4Status::success ||
          reference(-step, minus) != D4Status::success)
        return 52;
      for (int coordinate = 0; coordinate < coordinates; ++coordinate) {
        const double expected = (plus[coordinate] - minus[coordinate]) / (2 * step);
        if (!near(action[coordinate], expected, 2e-9)) {
          std::fprintf(stderr, "%s: D4 HVP[%d] %.17g expected %.17g\n", fixture.name, coordinate,
                       action[coordinate], expected);
          return 53;
        }
        largest_error = std::max(largest_error, std::abs(action[coordinate] - expected));
      }
    }
    std::fill(direction.begin(), direction.end(), -0.625);
    if (run() != D4Status::success ||
        !std::all_of(action.begin(), action.end(), [](double value) { return value == 0; }))
      return 54;
  }
  std::printf("Complete D4 HVP: max reconverged-gradient error %.3e\n", largest_error);
  return 0;
}

int negative_cases() {
  const auto& f = kEEQOracleFixtures.front();
  std::vector<double> workspace(complete_d4_eeq_workspace_elements(f.atoms));
  std::vector<double> gradient(3 * f.atoms, 17.0), charges(f.atoms, 19.0);
  double energy[2]{23.0, 29.0};
  auto wrong = f.parameters;
  wrong.ga = 3.0;
  wrong.gc = 2.0;
  const auto status = evaluate_complete_d4_eeq(
      f.atoms, f.z.data(), f.xyz.data(), f.total_charge, wrong, D4EEQProfile::r2scan3c,
      workspace.data(), workspace.size(), energy, gradient.data(), charges.data());
  if (status != D4Status::unsupported) return 30;
  if (energy[0] != 23.0 || energy[1] != 29.0 ||
      !std::all_of(gradient.begin(), gradient.end(), [](double x) { return x == 17.0; }) ||
      !std::all_of(charges.begin(), charges.end(), [](double x) { return x == 19.0; }))
    return 31;

  const auto table_status = evaluate_complete_d4_eeq_with_tables(
      f.atoms, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, D4EEQProfile::r2scan3c,
      eeq_d4_host_tables(D4EEQProfile::standard), eeq2019_host_tables(), workspace.data(),
      workspace.size(), energy, gradient.data(), charges.data());
  if (table_status != D4Status::unsupported) {
    std::fprintf(stderr, "table-profile mismatch status=%d\n", static_cast<int>(table_status));
    return 32;
  }
  if (energy[0] != 23.0 || energy[1] != 29.0 ||
      !std::all_of(gradient.begin(), gradient.end(), [](double x) { return x == 17.0; }) ||
      !std::all_of(charges.begin(), charges.end(), [](double x) { return x == 19.0; }))
    return 33;
  return 0;
}
}  // namespace

int main() {
  if (const int rc = oracle_cases()) return rc;
  if (const int rc = response_finite_differences()) return rc;
  if (const int rc = complete_gradient_finite_difference()) return rc;
  if (const int rc = directional_charge_pullback()) return rc;
  if (const int rc = complete_hessian_action()) return rc;
  if (const int rc = negative_cases()) return rc;
  std::puts("D4 EEQ oracle/response/complete-gradient tests passed");
  return 0;
}
