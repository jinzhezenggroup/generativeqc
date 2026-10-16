#include "model/gfn2/mulliken_kernels.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

namespace generativeqc::xtb::detail::gfn2 {

#if !defined(GENERATIVEQC_XTB_HAS_AVX2_FMA_KERNELS)
const MullikenKernelTable& mulliken_avx2_fma_kernels() noexcept {
  /* Keep baseline-only builds linkable for generic diagnostics and tests that
   * name an unavailable AVX2 table. Production keeps its baseline binding;
   * this compatibility accessor does not perform ISA admission. */
  return mulliken_baseline_kernels();
}
#endif

const MullikenKernelTable& mulliken_kernels_for_cpu_isa(
    generativeqc::runtime::CpuIsa isa) noexcept {
#if defined(GENERATIVEQC_XTB_HAS_AVX2_FMA_KERNELS)
  if (isa == generativeqc::runtime::CpuIsa::kAvx2Fma) {
    return mulliken_avx2_fma_kernels();
  }
#else
  (void)isa;
#endif
  return mulliken_baseline_kernels();
}

}  // namespace generativeqc::xtb::detail::gfn2
