#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace generativeqc::core {

/** A copied, endpoint-local convergence allowance; zero preserves strict callers.
 * This is a budget-scaled stopping policy, not a residual-to-observable
 * error estimator. Arithmetic, denominators and canonical eigensystem checks
 * are independent of this policy. Never infer it from an observed residual. */
struct CorrelationConvergence {
  double scale{};

  static CorrelationConvergence from_budget(const std::array<double, 3>& budget) {
    for (double value : budget)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("invalid correlated convergence budget");
    if (budget == std::array<double, 3>{}) return {};
    if (budget[0] <= 0 || budget[1] <= 0)
      throw std::invalid_argument("correlated convergence requires energy and force allowances");
    const double selected =
        std::min({1.0, budget[0] / 1e-5, budget[1] / 1e-4, budget[2] > 0 ? budget[2] / 3e-5 : 1.0});
    // Below this supported bracket retain strict convergence. Do not claim that a
    // tighter requested observable budget is certified by solver convergence.
    return {selected >= 1e-3 ? selected : 0.0};
  }

  void validate() const {
    if (!std::isfinite(scale) || scale < 0 || scale > 1)
      throw std::invalid_argument("invalid correlated convergence policy");
  }
  bool relaxed() const noexcept { return scale > 0; }
  double limit(double strict, double fast) const noexcept {
    return relaxed() ? std::max(strict, scale * fast) : strict;
  }
  double scf_energy() const noexcept { return limit(1e-11, 1e-8); }
  // Retain the physical RHF export/canonical-frame contract. Even Fast uses a
  // tighter reference density than DFT: post-HF energies depend on its orbitals.
  double scf_density() const noexcept { return limit(1e-11, 1e-9); }
  double cc_energy() const noexcept { return 1e-8; }
  double cc_residual() const noexcept { return limit(1e-9, 1e-6); }
  double lambda_residual() const noexcept { return limit(1e-9, 1e-7); }
  double response_solve() const noexcept { return limit(1e-12, 1e-8); }
  double orbital_residual() const noexcept { return limit(1e-10, 1e-7); }
  double stationarity(double strict = 1e-8) const noexcept { return limit(strict, 1e-6); }
  bool operator==(const CorrelationConvergence&) const = default;
};

}  // namespace generativeqc::core
