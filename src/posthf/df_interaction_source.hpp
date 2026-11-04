#pragma once

#include <span>

#include "integrals/correlation_backend.hpp"
#include "integrals/density_fitting_metric.hpp"
#include "integrals/electron_interaction_source.hpp"
#include "integrals/s_integrals.hpp"

namespace generativeqc::posthf {

/** Explicit dense AO fitted interaction for independent unrestricted frames.
 * Raw integral generation and Gram/VJP arithmetic follow the selected backend.
 * Native host FP64 metric conditioning and detached value/cotangent transport
 * are part of this hybrid contract, not an automatic backend fallback.
 * Checked endpoint storage/work admission replaces molecular AO-count caps. */
class DensityFittedInteractionSource final : public integrals::ElectronInteractionSource {
 public:
  DensityFittedInteractionSource(const integrals::ElectronInteractionSource& reference,
                                 core::System auxiliary, double threshold, std::size_t budget,
                                 bool cuda, int device, bool forces,
                                 integrals::CorrelationIntegralBackend backend = {});
  const core::System& orbital() const override { return reference_.orbital(); }
  const core::System& auxiliary() const noexcept { return auxiliary_; }
  std::size_t nbf() const override { return raw_.nbf; }
  std::size_t naux() const override { return raw_.naux; }
  std::size_t retained_numeric_bytes() const override;
  bool supports(Operator) const noexcept override { return true; }
  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* output,
            std::size_t elements) const override;
  std::size_t numeric_capacity_bytes() const noexcept { return capacity_; }
  /** Positive two-electron nuclear derivative only. No reference, hcore,
   * overlap or nuclear-repulsion contribution is implicitly added here. */
  std::vector<double> nuclear_derivative(std::span<const double> eri_cotangent) const;

 private:
  const integrals::ElectronInteractionSource& reference_;
  core::System auxiliary_;
  double threshold_{};
  std::size_t budget_{}, capacity_{};
  bool cuda_{}, forces_{};
  int device_{};
  integrals::CorrelationIntegralBackend backend_;
  integrals::DensityFittingIntegralData raw_;
  integrals::DensityFittingMetricFactor metric_;
  std::vector<double> eri_;
};
}  // namespace generativeqc::posthf
