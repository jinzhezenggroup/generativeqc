#include "scf/cuda/matrix_library.hpp"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <mutex>

#include "runtime/allocation_measurement.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"

namespace generativeqc::scf::cuda_execution {

MatrixLibraryOwner::~MatrixLibraryOwner() { reset(); }

generativeqc_status MatrixLibraryOwner::prepare(cudaStream_t stream, int nbf) {
  if (prepared_ || stream == nullptr || nbf <= 0) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;

  cudaStreamCaptureStatus capture{};
  auto cuda_error = cudaStreamIsCapturing(stream, &capture);
  if (cuda_error != cudaSuccess) return cuda_status(cuda_error);
  if (capture != cudaStreamCaptureStatusNone) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;

  cuda_error = cudaGetDevice(&device_);
  if (cuda_error != cudaSuccess) return cuda_status(cuda_error);
  stream_ = stream;
  prepared_ = true;

  if (!provider_allowance(nbf)) return GENERATIVEQC_STATUS_SUCCESS;

  std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
  std::size_t before{}, after{}, total{};
  cuda_error = cudaMemGetInfo(&before, &total);
  if (cuda_error != cudaSuccess) return cuda_status(cuda_error);

  auto blas_error = cublasCreate(&blas_);
  if (blas_error == CUBLAS_STATUS_ALLOC_FAILED) {
    blas_ = nullptr;
    (void)cudaGetLastError();
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  if (blas_error != CUBLAS_STATUS_SUCCESS) {
    blas_ = nullptr;
    return blas_status(blas_error);
  }

  const auto reject = [&](generativeqc_status status) {
    const auto cleanup_status = cublasDestroy(blas_);
    if (cleanup_status != CUBLAS_STATUS_SUCCESS) return blas_status(cleanup_status);
    blas_ = nullptr;
    retained_bytes_ = 0;
    return status;
  };
  blas_error = cublasSetStream(blas_, stream_);
  if (blas_error != CUBLAS_STATUS_SUCCESS) return reject(blas_status(blas_error));
  blas_error = cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST);
  if (blas_error != CUBLAS_STATUS_SUCCESS) return reject(blas_status(blas_error));
  blas_error = cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH);
  if (blas_error != CUBLAS_STATUS_SUCCESS) return reject(blas_status(blas_error));
  blas_error = cublasSetWorkspace(blas_, nullptr, 0);
  if (blas_error != CUBLAS_STATUS_SUCCESS) return reject(blas_status(blas_error));

  cuda_error = cudaMemGetInfo(&after, &total);
  if (cuda_error != cudaSuccess) return reject(cuda_status(cuda_error));
  retained_bytes_ = before > after ? before - after : 0;
  if (retained_bytes_ > kProviderAllowance) {
    // An unexpectedly large provider footprint is a resource miss, not a
    // scientific failure. Retain the generated implementation for this owner.
    // Live fallback must not hide a driver failure or lose a retained handle.
    // On failure the caller aborts preparation and reset() retries cleanup.
    blas_error = cublasDestroy(blas_);
    if (blas_error != CUBLAS_STATUS_SUCCESS) return blas_status(blas_error);
    blas_ = nullptr;
    retained_bytes_ = 0;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

void MatrixLibraryOwner::reset() noexcept {
  if (!prepared_) return;
  std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
  int previous = device_;
  (void)cudaGetDevice(&previous);
  if (device_ >= 0) (void)cudaSetDevice(device_);
  if (blas_ != nullptr) {
    if (stream_ != nullptr) (void)cudaStreamSynchronize(stream_);
    (void)cublasDestroy(blas_);
  }
  if (previous >= 0) (void)cudaSetDevice(previous);
  device_ = -1;
  stream_ = nullptr;
  blas_ = nullptr;
  retained_bytes_ = 0;
  prepared_ = false;
}

namespace {

bool matrix_shape(int batch_size, int spin_count, int nbf, std::size_t& matrix_size,
                  std::size_t& elements) {
  if (batch_size <= 0 || spin_count <= 0 || nbf <= 0) return false;
  const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
  matrix_size = static_cast<std::size_t>(nbf);
  for (const auto extent : {nbf, batch_size, spin_count}) {
    if (matrix_size > limit / static_cast<std::size_t>(extent)) return false;
    matrix_size *= static_cast<std::size_t>(extent);
  }
  elements = matrix_size;
  matrix_size = static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  return (elements - 1) / kCaptureSafeKernelThreads <
         static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool overlaps(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  const auto a = reinterpret_cast<std::uintptr_t>(first);
  const auto b = reinterpret_cast<std::uintptr_t>(second);
  const auto limit = std::numeric_limits<std::uintptr_t>::max();
  if (a > limit - first_bytes || b > limit - second_bytes) return true;
  return a < b + second_bytes && b < a + first_bytes;
}

bool masked_output_admitted(MatrixLibraryResources resources, std::size_t elements, int batch_size,
                            const double* left, std::size_t left_elements, const double* right,
                            std::size_t right_elements, const std::uint8_t* active,
                            const double* output) {
  if (!resources.masked_output_ || resources.masked_output_elements_ < elements) return false;
  const auto bytes = elements * sizeof(double);
  return !overlaps(resources.masked_output_, bytes, left, left_elements * sizeof(double)) &&
         !overlaps(resources.masked_output_, bytes, right, right_elements * sizeof(double)) &&
         !overlaps(resources.masked_output_, bytes, output, bytes) &&
         !overlaps(resources.masked_output_, bytes, active, static_cast<std::size_t>(batch_size));
}

generativeqc_status commit_masked_output(MatrixLibraryResources resources, int batch_size,
                                         int spin_count, int nbf, std::size_t elements,
                                         const std::uint8_t* active, double* output) {
  if (!active) return GENERATIVEQC_STATUS_SUCCESS;
  const auto blocks = static_cast<unsigned>((elements - 1) / kCaptureSafeKernelThreads + 1);
  launch_copy_selected_matrices_kernel(blocks, kCaptureSafeKernelThreads, 0, resources.stream_,
                                       batch_size, spin_count, nbf, active,
                                       resources.masked_output_, output);
  return cuda_status(cudaPeekAtLastError());
}

}  // namespace

generativeqc_status launch_matrix_product(MatrixLibraryResources resources, int batch_size, int nbf,
                                          const double* left, bool transpose_left,
                                          const double* right, const std::uint8_t* active,
                                          double* output, bool use_cublas, double scale) {
  std::size_t matrix_size{}, elements{};
  if (!matrix_shape(batch_size, 1, nbf, matrix_size, elements))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!use_cublas) {
    const unsigned blocks = static_cast<unsigned>((elements + kCaptureSafeKernelThreads - 1) /
                                                  kCaptureSafeKernelThreads);
    launch_matrix_product_kernel(blocks, kCaptureSafeKernelThreads, 0, resources.stream_,
                                 batch_size, nbf, left, transpose_left, right, active, output,
                                 scale);
    return cuda_status(cudaPeekAtLastError());
  }

  if (active && !masked_output_admitted(resources, elements, batch_size, left, elements, right,
                                        elements, active, output))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  double* destination = active ? resources.masked_output_ : output;

  const double alpha = scale;
  const double beta = 0.0;
  const cublasOperation_t operation = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto status =
      batch_size == 1
          ? cublasDgemm(resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha, left, nbf,
                        right, nbf, &beta, destination, nbf)
          : cublasDgemmStridedBatched(resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf,
                                      &alpha, left, nbf, static_cast<long long>(matrix_size), right,
                                      nbf, static_cast<long long>(matrix_size), &beta, destination,
                                      nbf, static_cast<long long>(matrix_size), batch_size);
  if (status != CUBLAS_STATUS_SUCCESS) return blas_status(status);
  return commit_masked_output(resources, batch_size, 1, nbf, elements, active, output);
}

/**
 * Multiply system-major spin matrices while broadcasting physical operands.
 *
 * A physical matrix repeats for alpha and beta, which is not one constant
 * stride over the interleaved state array. One library submission per spin
 * preserves the existing [system][spin][matrix] storage without pointer lists.
 * A single physical system uses ordinary GEMM; only a true multi-system batch
 * uses strided-batched GEMM.
 */
generativeqc_status launch_spin_matrix_product(MatrixLibraryResources resources, int batch_size,
                                               int spin_count, int nbf, const double* left,
                                               bool left_is_spin, bool transpose_left,
                                               const double* right, bool right_is_spin,
                                               const std::uint8_t* active, double* output,
                                               bool use_cublas) {
  std::size_t matrix_size{}, elements{};
  if (!matrix_shape(batch_size, spin_count, nbf, matrix_size, elements))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!use_cublas) {
    const unsigned blocks = static_cast<unsigned>((elements + kCaptureSafeKernelThreads - 1) /
                                                  kCaptureSafeKernelThreads);
    launch_spin_matrix_product_kernel(blocks, kCaptureSafeKernelThreads, 0, resources.stream_,
                                      batch_size, spin_count, nbf, left, left_is_spin,
                                      transpose_left, right, right_is_spin, active, output);
    return cuda_status(cudaPeekAtLastError());
  }

  const auto physical_elements = elements / static_cast<std::size_t>(spin_count);
  if (active &&
      !masked_output_admitted(resources, elements, batch_size, left,
                              left_is_spin ? elements : physical_elements, right,
                              right_is_spin ? elements : physical_elements, active, output))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  double* destination = active ? resources.masked_output_ : output;

  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasOperation_t operation = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  const long long physical_stride = static_cast<long long>(matrix_size);
  const long long spin_stride =
      static_cast<long long>(matrix_size * static_cast<std::size_t>(spin_count));
  for (int spin = 0; spin < spin_count; ++spin) {
    const std::size_t spin_offset = static_cast<std::size_t>(spin) * matrix_size;
    const double* spin_left = left + (left_is_spin ? spin_offset : 0);
    const double* spin_right = right + (right_is_spin ? spin_offset : 0);
    const cublasStatus_t status =
        batch_size == 1
            ? cublasDgemm(resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha, spin_left,
                          nbf, spin_right, nbf, &beta, destination + spin_offset, nbf)
            : cublasDgemmStridedBatched(resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf,
                                        &alpha, spin_left, nbf,
                                        left_is_spin ? spin_stride : physical_stride, spin_right,
                                        nbf, right_is_spin ? spin_stride : physical_stride, &beta,
                                        destination + spin_offset, nbf, spin_stride, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_status(status);
  }
  return commit_masked_output(resources, batch_size, spin_count, nbf, elements, active, output);
}

}  // namespace generativeqc::scf::cuda_execution
