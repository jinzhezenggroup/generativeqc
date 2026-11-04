#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "posthf/native_provider.hpp"

namespace generativeqc::posthf {
/** Synchronous borrowed-factor execution owner. Optional device residency may
 * fall back to bounded panel uploads, never to host contraction or a new model. */
class DFFactorCudaBlocks {
 public:
  virtual ~DFFactorCudaBlocks() = default;
  virtual std::size_t retained_numeric_bytes() const noexcept = 0;
  virtual std::vector<double> get(const SpinMOSlots& request, ProviderWork* work) const = 0;
};

#if GENERATIVEQC_HAS_CUDA
std::unique_ptr<DFFactorCudaBlocks> make_df_factor_cuda_blocks(
    std::size_t nbf, std::size_t naux, std::array<std::span<const double>, 2> factors, int device,
    std::size_t resident_budget);
#endif
}  // namespace generativeqc::posthf
