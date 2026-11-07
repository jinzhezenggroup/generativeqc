#include <limits>
#include <utility>

#include "tensor/cuda_dense_operations.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::tensor {

cublasStatus_t CudaDenseOperations::create() {
  if (handle_) return CUBLAS_STATUS_INVALID_VALUE;
  return cublasCreate(&handle_);
}

cublasStatus_t CudaDenseOperations::bind(cudaStream_t stream) {
  auto status = cublasSetStream(handle_, stream);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
  return status;
}

cublasStatus_t CudaDenseOperations::close() noexcept {
  const auto handle = std::exchange(handle_, nullptr);
  return handle ? cublasDestroy(handle) : CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CudaDenseOperations::copy(int count, const double* source, int source_stride,
                                         double* destination, int destination_stride) const {
  return cublasDcopy(handle_, count, source, source_stride, destination, destination_stride);
}

cublasStatus_t CudaDenseOperations::scale(int count, const double* coefficient, double* values,
                                          int stride) const {
  return cublasDscal(handle_, count, coefficient, values, stride);
}

cublasStatus_t CudaDenseOperations::add(int count, const double* coefficient, const double* source,
                                        int source_stride, double* destination,
                                        int destination_stride) const {
  return cublasDaxpy(handle_, count, coefficient, source, source_stride, destination,
                     destination_stride);
}

cublasStatus_t CudaDenseOperations::dot(int count, const double* left, int left_stride,
                                        const double* right, int right_stride,
                                        double* result) const {
  return cublasDdot(handle_, count, left, left_stride, right, right_stride, result);
}

cublasStatus_t CudaDenseOperations::norm(int count, const double* values, int stride,
                                         double* result) const {
  return cublasDnrm2(handle_, count, values, stride, result);
}

cublasStatus_t CudaDenseOperations::product(bool transpose_left, bool transpose_right, int rows,
                                            int columns, int reduction, const double* alpha,
                                            const double* left, int left_ld, const double* right,
                                            int right_ld, const double* beta, double* output,
                                            int output_ld) const {
  return generativeqc_tensor::submit_column_gemm({handle_, transpose_left, transpose_right, rows,
                                                  columns, reduction, alpha, left, left_ld, right,
                                                  right_ld, beta, output, output_ld});
}

namespace {

template <class Handle>
cublasStatus_t transpose_sum(Handle handle, int order, const double* input, double* output,
                             std::uint64_t* successful_submissions) {
  if (order <= 0 || order > std::numeric_limits<int>::max() / order || !input || !output ||
      input == output)
    return CUBLAS_STATUS_INVALID_VALUE;
  const double one = 1.0;
  const auto record = [&](cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS && successful_submissions) ++*successful_submissions;
    return status;
  };
  if constexpr (generativeqc_tensor::column_geam_available<Handle>) {
    return record(generativeqc_tensor::submit_column_geam(
        generativeqc_tensor::ColumnGeamView<Handle>{handle, false, true, order, order, &one, input,
                                                    order, &one, input, order, output, order}));
  } else {
    auto status = record(cublasDcopy(handle, order * order, input, 1, output, 1));
    for (int column = 0; status == CUBLAS_STATUS_SUCCESS && column < order; ++column)
      status = record(cublasDaxpy(handle, order, &one, input + column, order,
                                  output + static_cast<std::size_t>(column) * order, 1));
    return status;
  }
}

}  // namespace

cublasStatus_t CudaDenseOperations::transpose_sum(int order, const double* input, double* output,
                                                  std::uint64_t* successful_submissions) const {
  return tensor::transpose_sum(handle_, order, input, output, successful_submissions);
}

}  // namespace generativeqc::tensor
