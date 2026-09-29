#include "runtime/cuda_provider.hpp"

#ifndef GENERATIVEQC_HAS_CUDA
#define GENERATIVEQC_HAS_CUDA 0
#endif
#ifndef GENERATIVEQC_CUDA_PROVIDER_CUMETAL
#define GENERATIVEQC_CUDA_PROVIDER_CUMETAL 0
#endif

namespace generativeqc::runtime {

const CudaProviderCapabilities& active_cuda_provider() noexcept {
#if !GENERATIVEQC_HAS_CUDA
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::None);
#elif GENERATIVEQC_CUDA_PROVIDER_CUMETAL
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::CuMetal);
#else
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::Nvidia);
#endif
  return provider;
}

}  // namespace generativeqc::runtime
