#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace generativeqc::posthf {
/** Detached generated value or raw-source cotangents. Metric conditioning and
 * orbital/nuclear response are composed by the containing physical owner. */
struct DFAOSourceResult {
  std::vector<double> eri, three_center, inverse_root;
  std::size_t numeric_capacity_bytes{};
};

#if GENERATIVEQC_HAS_CUDA
DFAOSourceResult evaluate_df_ao_cuda(std::size_t nbf, std::size_t naux,
                                     std::span<const double> three_center,
                                     std::span<const double> inverse_root,
                                     std::span<const double> eri_cotangent, int device,
                                     std::size_t budget);
#endif
}  // namespace generativeqc::posthf
