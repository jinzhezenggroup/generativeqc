#include "tensor/cuda_matrix_product.hpp"

#include <limits>
#include <stdexcept>

#include "generated_matrix_product_lowering.hpp"

namespace generativeqc::tensor {

static_assert(sizeof(CudaMatrixProductBinding) +
                  2 * sizeof(std::array<runtime::NativeLoweringCandidate, 5>) + 4096 <=
              CudaMatrixProductBinding::kPreparationHostBytes);

CudaMatrixProductBinding::CudaMatrixProductBinding(cudaStream_t stream, cublasHandle_t handle,
                                                   int dimension, int systems, int spins)
    : stream_(stream), handle_(handle), dimension_(dimension), systems_(systems), spins_(spins) {
  if (!stream || dimension <= 0 || systems <= 0 || spins <= 0)
    throw std::invalid_argument("invalid prepared matrix domain");
  std::size_t elements = static_cast<std::size_t>(dimension);
  const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
  for (int extent : {dimension, systems, spins}) {
    if (elements > limit / static_cast<std::size_t>(extent))
      throw std::length_error("prepared matrix domain overflow");
    elements *= static_cast<std::size_t>(extent);
  }
  scratch_bytes_ = elements * sizeof(double);
  auto status = cudaGetDevice(&context_.device);
  if (status == cudaSuccess) status = cudaRuntimeGetVersion(&context_.runtime_version);
  if (status != cudaSuccess)
    throw PreparationFailure(status == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                                 : GENERATIVEQC_STATUS_CUDA_ERROR,
                             "matrix binding CUDA context query failed");
  if (handle) {
    const auto provider_status = cublasGetVersion(handle, &context_.provider_version);
    if (provider_status != CUBLAS_STATUS_SUCCESS)
      throw PreparationFailure(provider_status == CUBLAS_STATUS_ALLOC_FAILED
                                   ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                   : GENERATIVEQC_STATUS_CUDA_ERROR,
                               "matrix binding provider version query failed");
  }
  for (std::size_t slot = 0; slot < sites_.size(); ++slot) {
    const auto portfolio = matrix_product_lowering::portfolio(slot);
    auto offers = portfolio.candidates;
    std::size_t incumbent{};
    for (std::size_t index = 0; index < offers.size(); ++index) {
      auto& offer = offers[index];
      offer.host_bytes = kPreparationHostBytes;
      if (offer.provider == "generated.cuda") {
        if (!handle) incumbent = index;
        offer.capture_safe = true;
      } else if (offer.provider == "cublas") {
        offer.provider_bytes = 96ULL << 20;
        offer.temporary_bytes = scratch_bytes_;
        offer.capture_safe = true;
        if (handle)
          incumbent = index;
        else
          offer.rejection = "borrowed optional provider context is unavailable";
      }
    }
    const auto decision = runtime::select_native_lowering(
        portfolio.request, offers, portfolio.target, portfolio.compilation, 1, incumbent);
    auto& site = sites_[slot];
    const auto& selected = offers[decision.selected];
    const auto recipe = selected.provider == "cublas" ? Recipe::Library : Recipe::Generated;
    if (slot == 0)
      recipe_ = recipe;
    else if (recipe != recipe_)
      throw std::invalid_argument("matrix family requires independently executable recipes");
    site.request = &portfolio.request;
    site.selected = &portfolio.candidates[decision.selected];
    site.provider_bytes = selected.provider_bytes;
    site.temporary_bytes = selected.temporary_bytes;
    site.host_bytes = selected.host_bytes;
    site.retained_incumbent = decision.retained_incumbent;
    if (!decision.rejections.empty())
      site.alternative_rejection = decision.rejections.front().reason;
  }
}

runtime::NativeLoweringCandidate CudaMatrixProductBinding::Site::candidate() const {
  auto result = *selected;
  result.provider_bytes = provider_bytes;
  result.temporary_bytes = temporary_bytes;
  result.host_bytes = host_bytes;
  result.capture_safe = true;
  return result;
}

bool CudaMatrixProductBinding::compatible(cudaStream_t stream, cublasHandle_t handle, int dimension,
                                          int systems, int spins) const noexcept {
  return stream == stream_ && handle == handle_ && dimension == dimension_ && systems > 0 &&
         systems <= systems_ && spins > 0 && spins <= spins_;
}

const CudaMatrixProductBinding::Site& CudaMatrixProductBinding::site(bool spin, bool left_spin,
                                                                     bool right_spin,
                                                                     bool transpose) const {
  if (!spin && (left_spin || right_spin))
    throw std::invalid_argument("plain products cannot carry spin views");
  return sites_[spin ? 2U + 4U * left_spin + 2U * right_spin + transpose : transpose];
}

namespace cuda {

generativeqc_status matrix_product(cublasHandle_t handle, int systems, int spins, int dimension,
                                   const double* left, bool left_spin, bool transpose_left,
                                   const double* right, bool right_spin, double* output,
                                   double scale) {
  const auto matrix_size = static_cast<long long>(dimension) * dimension;
  const auto spin_stride = matrix_size * spins;
  const double zero = 0;
  for (int spin = 0; spin < spins; ++spin) {
    const auto offset = static_cast<std::size_t>(spin) * matrix_size;
    const auto* spin_left = left + (left_spin ? offset : 0);
    const auto* spin_right = right + (right_spin ? offset : 0);
    const auto operation = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto status =
        systems == 1 ? cublasDgemm(handle, operation, CUBLAS_OP_N, dimension, dimension, dimension,
                                   &scale, spin_left, dimension, spin_right, dimension, &zero,
                                   output + offset, dimension)
                     : cublasDgemmStridedBatched(
                           handle, operation, CUBLAS_OP_N, dimension, dimension, dimension, &scale,
                           spin_left, dimension, left_spin ? spin_stride : matrix_size, spin_right,
                           dimension, right_spin ? spin_stride : matrix_size, &zero,
                           output + offset, dimension, spin_stride, systems);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                  : GENERATIVEQC_STATUS_CUDA_ERROR;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace cuda
}  // namespace generativeqc::tensor
