#pragma once

#include <cmath>

#include "scf/fock_build.hpp"

namespace generativeqc::scf {

/** Arithmetic capabilities of the shared Direct value consumer, not a method
 * qualification. Fitted providers retain their own strict consumers.
 * Basis/resource admission and final physical validation belong to
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

/** Shared SR/LR values retain FP64 radial moments and output accumulation.
 * Whole-shell contraction stays FP64; bounded AO contraction follows the same
 * mixed scalar policy as full-range values. Derivatives and fitted sources
 * are deliberately outside this capability. */
inline bool cuda_range_exchange_precision_capability(const ResolvedFockBuild& strategy) noexcept {
  const auto& spec = strategy.spec;
  const auto& term = spec.exchange;
  const bool direct_schedule = strategy.schedule == FockSchedule::CudaFused ||
                               strategy.schedule == FockSchedule::CudaIndependent;
  return strategy.backend == FockBackend::Cuda && direct_schedule && spec.version == 1 &&
         spec.derivative_order == 0 && spec.spin == FockSpin::Restricted && !spec.coulomb.present &&
         term.present && term.approximation == FockApproximation::Exact &&
         (term.op == FockOperator::ShortRange || term.op == FockOperator::LongRange) &&
         term.omega > 0.0 && std::isfinite(term.omega) && std::isfinite(term.coefficient);
}

}  // namespace generativeqc::scf
