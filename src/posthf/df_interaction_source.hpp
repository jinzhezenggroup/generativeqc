#pragma once

#include <array>
#include <memory>
#include <span>

#include "hf/reference.hpp"
#include "integrals/correlation_backend.hpp"
#include "integrals/density_fitting_metric.hpp"
#include "integrals/electron_interaction_source.hpp"
#include "integrals/s_integrals.hpp"
#include "posthf/df_ao_source.hpp"

namespace generativeqc::posthf {

struct SpinMOSlots;
struct ProviderWork;
class DFFactorCudaBlocks;

/** Immutable three-index fitted interaction shared by unrestricted consumers.
 * Raw integral generation, orbital projections and VJP use the selected backend.
 * Native host FP64 metric conditioning and detached value/cotangent transport
 * are part of this hybrid contract, not an automatic backend fallback.
 * Checked endpoint storage/work admission replaces molecular AO-count caps. */
class DensityFittedInteractionSource final : public integrals::ElectronInteractionSource {
 public:
  DensityFittedInteractionSource(const integrals::ElectronInteractionSource& reference,
                                 const hf::UnrestrictedPhysicalReference& frame,
                                 core::System auxiliary, double threshold, std::size_t budget,
                                 bool cuda, int device, bool forces,
                                 integrals::CorrelationIntegralBackend backend = {});
  ~DensityFittedInteractionSource();
  DensityFittedInteractionSource(const DensityFittedInteractionSource&) = delete;
  DensityFittedInteractionSource& operator=(const DensityFittedInteractionSource&) = delete;
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
  /** Reject stale spin frames before replaying cached scientific intermediates. */
  void validate_frame(const core::ElectronicReferenceView& frame) const;
  const std::vector<double>& spin_factors(unsigned spin) const;
  /** Accumulate both Gram legs of one ordered MO integral cotangent. */
  void add_pair_cotangent(std::array<std::vector<double>, 2>& seeds, unsigned left_spin,
                          unsigned right_spin, const std::array<std::size_t, 4>& orbitals,
                          double weight) const;
  std::size_t block_capacity(const std::array<std::size_t, 4>& shape, bool cuda) const;
  std::vector<double> factor_block(const SpinMOSlots& request, bool cuda, int device,
                                   std::size_t budget, ProviderWork* work) const;
  /** Complete raw/root/orbital reverse of one spin's cached three-index map. */
  DFFactorSourceResult pullback_factors(unsigned spin, std::span<const double> seed) const;
  /** Positive fitted two-electron nuclear derivative, holding orbitals fixed.
   * Reference, hcore, overlap and nuclear-repulsion terms remain caller-owned. */
  std::vector<double> nuclear_derivative(const std::vector<double>& raw_cotangent,
                                         const std::vector<double>& root_cotangent) const;

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
  std::array<std::vector<double>, 2> coefficients_;
  std::array<std::size_t, 2> occupied_{};
  std::vector<double> ao_factors_;
  std::array<std::vector<double>, 2> factors_;
  std::unique_ptr<DFFactorCudaBlocks> cuda_blocks_;
};
}  // namespace generativeqc::posthf
