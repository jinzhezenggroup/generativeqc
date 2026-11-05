#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace generativeqc::posthf {
/** A spin-frame three-index value or its complete raw/orbital/root pullback. */
struct DFFactorSourceResult {
  std::vector<double> factors, three_center, inverse_root, coefficients;
  std::size_t numeric_capacity_bytes{};
};

#if GENERATIVEQC_HAS_CUDA
DFFactorSourceResult evaluate_df_factors_cuda(std::size_t nbf, std::size_t naux,
                                              std::span<const double> three_center,
                                              std::span<const double> inverse_root,
                                              std::span<const double> coefficients,
                                              std::span<const double> factor_cotangent, int device,
                                              std::size_t budget);
/** Enqueue the generated rectangular Gram contraction in caller-owned storage. */
void evaluate_df_factor_block_device(std::size_t left_pairs, std::size_t right_pairs,
                                     std::size_t naux, const double* left, const double* right,
                                     double* output, int* error, void* stream);
#endif
}  // namespace generativeqc::posthf
