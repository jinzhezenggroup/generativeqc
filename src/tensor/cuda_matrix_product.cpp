#include "tensor/cuda_matrix_product.hpp"

#include <limits>
#include <stdexcept>
#if defined(GENERATIVEQC_TEST_HOOKS)
#include <iomanip>
#include <sstream>
#endif

#include "generated_matrix_product_lowering.hpp"
#include "tensor/cuda_structured_product.hpp"

namespace generativeqc::tensor {

namespace {
#if defined(GENERATIVEQC_TEST_HOOKS)
thread_local int structured_qualification_mask = -1;
thread_local std::string structured_preparation_diagnostic;
#endif
unsigned structured_preparation_mask() noexcept {
#if defined(GENERATIVEQC_TEST_HOOKS)
  if (structured_qualification_mask >= 0)
    return static_cast<unsigned>(structured_qualification_mask);
#endif
  return CudaStructuredProductBinding::production_library_qualified ? 127U : 0U;
}
bool structured_diagnostic_preparation() noexcept {
#if defined(GENERATIVEQC_TEST_HOOKS)
  return structured_qualification_mask >= 0;
#else
  return false;
#endif
}
}  // namespace

#if defined(GENERATIVEQC_TEST_HOOKS)
/** Select diagnostic regions before preparing an owner, never during execution.
 * -1 restores production policy; zero is a matched all-generated ablation. */
extern "C" bool tensor_structured_scf_qualification_for_test_v1(int region_mask) {
  if (region_mask < -1 || region_mask > 127) return false;
  structured_qualification_mask = region_mask;
  return true;
}

/** Diagnostic data belongs to the last preparation on this thread. No handle
 * or scientific state escapes, and normal builds have no telemetry storage. */
extern "C" const char* tensor_structured_scf_preparation_diagnostic_for_test_v1() {
  return structured_preparation_diagnostic.c_str();
}
#endif

bool CudaStructuredProductBinding::basis_workspace_required() noexcept {
  return (structured_preparation_mask() & 120U) != 0;
}

CudaStructuredProductBinding::CudaStructuredProductBinding(cudaStream_t stream,
                                                           cublasHandle_t handle, int dimension,
                                                           int direct_dimension, int systems,
                                                           int spins)
    : CudaStructuredProductBinding(stream, handle, dimension, direct_dimension, systems, spins,
                                   false) {}

CudaStructuredProductBinding::CudaStructuredProductBinding(cudaStream_t stream,
                                                           cublasHandle_t handle, int dimension,
                                                           int direct_dimension, int systems,
                                                           int spins, bool primitive_qualification)
    : stream_(stream),
      handle_(handle),
      dimension_(dimension),
      direct_dimension_(direct_dimension ? direct_dimension : dimension),
      systems_(systems),
      spins_(spins),
      qualification_mask_(primitive_qualification ? 127U : structured_preparation_mask()),
      diagnostic_(!primitive_qualification && structured_diagnostic_preparation()) {
  static_assert(sizeof(CudaStructuredProductBinding) +
                    2 * sizeof(std::array<runtime::NativeLoweringCandidate, 7>) + 8192 <=
                kPreparationHostBytes);
  if (!stream || dimension <= 0 || direct_dimension_ <= 0 || systems <= 0 || spins <= 0)
    throw std::invalid_argument("invalid structured matrix domain");
  if (systems > std::numeric_limits<int>::max() / spins)
    throw std::length_error("structured matrix state count overflow");
  std::size_t elements = static_cast<std::size_t>(std::max(dimension, direct_dimension_));
  const auto limit = std::numeric_limits<std::size_t>::max() / (3 * sizeof(double));
  for (int extent : {std::max(dimension, direct_dimension_), systems, spins}) {
    if (elements > limit / static_cast<std::size_t>(extent))
      throw std::length_error("structured matrix domain overflow");
    elements *= static_cast<std::size_t>(extent);
  }
  for (std::size_t slot = 0; slot < sites_.size(); ++slot) {
    const auto portfolio = structured_lowering::portfolio(slot);
    auto offers = candidates(static_cast<StructuredOperation>(slot));
    const std::size_t incumbent = !handle || !(qualification_mask_ & (1U << slot)) ? 0
                                  : slot == 0                                      ? 2
                                                                                   : 1;
    const auto decision = runtime::select_native_lowering(
        portfolio.request, offers, portfolio.target, portfolio.compilation, 1, incumbent);
    sites_[slot] = offers[decision.selected];
  }
#if defined(GENERATIVEQC_TEST_HOOKS)
  if (structured_diagnostic_preparation()) {
    std::ostringstream diagnostic;
    diagnostic << "{\"dimension\":" << dimension_ << ",\"direct_dimension\":" << direct_dimension_
               << ",\"systems\":" << systems_ << ",\"spins\":" << spins_ << ",\"regions\":[";
    for (std::size_t slot = 0; slot < sites_.size(); ++slot) {
      if (slot) diagnostic << ',';
      diagnostic << "{\"operation\":" << slot
                 << ",\"selected\":" << std::quoted(std::string(sites_[slot].provider))
                 << ",\"offers\":[";
      const auto offers = candidates(static_cast<StructuredOperation>(slot));
      for (std::size_t offer = 0; offer < offers.size(); ++offer) {
        if (offer) diagnostic << ',';
        diagnostic << "{\"provider\":" << std::quoted(std::string(offers[offer].provider))
                   << ",\"rejection\":" << std::quoted(std::string(offers[offer].rejection))
                   << ",\"temporary_bytes\":" << offers[offer].temporary_bytes
                   << ",\"host_bytes\":" << offers[offer].host_bytes
                   << ",\"provider_bytes\":" << offers[offer].provider_bytes << '}';
      }
      diagnostic << "]}";
    }
    diagnostic << "]}";
    structured_preparation_diagnostic = diagnostic.str();
  }
#endif
}

std::array<runtime::NativeLoweringCandidate, 7> CudaStructuredProductBinding::candidates(
    StructuredOperation operation) const {
  const auto slot = static_cast<std::size_t>(operation);
  auto offers = structured_lowering::portfolio(slot).candidates;
  const auto square_elements =
      static_cast<std::size_t>(dimension_) * dimension_ * systems_ * spins_;
  const auto maximum_dimension = std::max(dimension_, direct_dimension_);
  const auto basis_elements =
      static_cast<std::size_t>(maximum_dimension) * maximum_dimension * systems_ * spins_;
  for (auto& offer : offers) {
    offer.host_bytes = kPreparationHostBytes;
    if (offer.provider != "cublas") continue;
    offer.provider_bytes = 96ULL << 20;
    offer.temporary_bytes =
        (slot < 3 ? (slot == 1 ? 3 : 2) * square_elements : basis_elements) * sizeof(double);
    if (!offer.rejection.empty()) continue;
    if (!handle_)
      offer.rejection = "borrowed provider context unavailable";
    else if (!(qualification_mask_ & (1U << slot)))
      offer.rejection = "structured scientific endpoint qualification remains open";
  }
  return offers;
}

bool CudaStructuredProductBinding::compatible(cudaStream_t stream, cublasHandle_t handle,
                                              int dimension, int systems,
                                              int spins) const noexcept {
  return stream == stream_ && handle == handle_ && dimension == dimension_ && systems > 0 &&
         systems <= systems_ && spins > 0 && spins <= spins_;
}

const runtime::NativeLoweringCandidate& CudaStructuredProductBinding::site(
    StructuredOperation operation) const {
  return sites_.at(static_cast<std::size_t>(operation));
}

bool CudaStructuredProductBinding::library(StructuredOperation operation) const {
  return site(operation).provider == "cublas";
}

bool CudaStructuredProductBinding::basis_workspace_enabled() const {
  for (std::size_t slot = 3; slot < sites_.size(); ++slot)
    if (library(static_cast<StructuredOperation>(slot))) return true;
  return false;
}

bool CudaStructuredProductBinding::diagnostic_dense_basis(StructuredOperation operation) const {
  return diagnostic_ && operation >= StructuredOperation::DensityRight && library(operation);
}

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

generativeqc_status structured_gram(cublasHandle_t handle, int dimension, int states,
                                    const double* coefficients, const double* scaled,
                                    double* scratch, double occupancy, bool unsigned_gram) {
  const auto matrix = static_cast<std::size_t>(dimension) * dimension;
  const double zero = 0;
  for (int state = 0; state < states; ++state) {
    const auto offset = static_cast<std::size_t>(state) * matrix;
    const auto status =
        unsigned_gram
            ? cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dimension, dimension,
                          &occupancy, scaled + offset, dimension, &zero, scratch + offset,
                          dimension)
            : cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dimension, dimension, dimension,
                          &occupancy, scaled + offset, dimension, coefficients + offset, dimension,
                          &zero, scratch + offset, dimension);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                  : GENERATIVEQC_STATUS_CUDA_ERROR;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status basis_stage(cublasHandle_t handle, StructuredOperation operation, int dimension,
                                int direct_dimension, int systems, int spins,
                                const double* transform, const double* input, double* scratch) {
  const auto rectangle = static_cast<std::size_t>(dimension) * direct_dimension;
  const auto square = static_cast<std::size_t>(dimension) * dimension;
  const auto direct_square = static_cast<std::size_t>(direct_dimension) * direct_dimension;
  const double one = 1, zero = 0;
  for (int system = 0; system < systems; ++system) {
    const auto* coefficients = transform + static_cast<std::size_t>(system) * rectangle;
    for (int spin = 0; spin < spins; ++spin) {
      const auto state = static_cast<std::size_t>(system) * spins + spin;
      cublasStatus_t status{};
      if (operation == StructuredOperation::DensityRight)
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension, direct_dimension,
                             dimension, &one, input + state * square, dimension, coefficients,
                             dimension, &zero, scratch + state * rectangle, dimension);
      else if (operation == StructuredOperation::DensityLeft)
        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, direct_dimension, direct_dimension,
                             dimension, &one, coefficients, dimension, input + state * rectangle,
                             dimension, &zero, scratch + state * direct_square, direct_dimension);
      else if (operation == StructuredOperation::FockLeft)
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension, direct_dimension,
                             direct_dimension, &one, coefficients, dimension,
                             input + state * direct_square, direct_dimension, &zero,
                             scratch + state * rectangle, dimension);
      else if (operation == StructuredOperation::FockRight)
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dimension, dimension,
                             direct_dimension, &one, input + state * rectangle, dimension,
                             coefficients, dimension, &zero, scratch + state * square, dimension);
      else
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      if (status != CUBLAS_STATUS_SUCCESS)
        return status == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                    : GENERATIVEQC_STATUS_CUDA_ERROR;
    }
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

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
