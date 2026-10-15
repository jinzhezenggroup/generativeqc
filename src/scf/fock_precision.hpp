#pragma once

#include <cmath>

#include "scf/fock_build.hpp"

namespace generativeqc::scf {

/** Arithmetic capabilities of the shared Direct value consumer, not a method
 * qualification. Range exchange and fitted providers retain their own strict
 * consumers. Basis/resource admission and final physical validation belong to
 * the enclosing solver; neither XC identity nor exchange fraction selects J/K
 * recurrence precision. */
struct CudaFockPrecisionCapabilities {
  bool coulomb{}, exchange{};

  bool any_lower_precision() const noexcept { return coulomb || exchange; }
};

inline CudaFockPrecisionCapabilities cuda_fock_precision_capabilities(
    const ResolvedFockBuild& strategy) noexcept {
  const auto& spec = strategy.spec;
  const bool direct_schedule = strategy.schedule == FockSchedule::CudaFused ||
                               strategy.schedule == FockSchedule::CudaIndependent;
  if (strategy.backend != FockBackend::Cuda || !direct_schedule || spec.version != 1 ||
      spec.derivative_order != 0 || spec.spin != FockSpin::Restricted)
    return {};
  const auto exact_full_range = [](const FockTermSpec& term) {
    return term.present && term.approximation == FockApproximation::Exact &&
           term.op == FockOperator::FullRange && term.omega == 0.0 &&
           std::isfinite(term.coefficient);
  };
  const bool coulomb = exact_full_range(spec.coulomb);
  return {coulomb, coulomb && exact_full_range(spec.exchange)};
}

}  // namespace generativeqc::scf
