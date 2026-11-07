#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "tensor/cuda_dense_operations.hpp"

namespace {
Handle provider;
int failure{}, fail_after{}, submitted{};
bool numerical{};
std::vector<std::uint64_t> record;

template <class Value>
std::uint64_t bits(Value value) {
  if constexpr (std::is_pointer_v<Value>)
    return reinterpret_cast<std::uintptr_t>(value);
  else
    return static_cast<std::uint64_t>(value);
}

template <class... Values>
int capture(Values... values) {
  record = {bits(values)...};
  ++submitted;
  return fail_after == 0 || submitted == fail_after ? failure : 0;
}
}  // namespace

int cublasCreate(cublasHandle_t* handle) {
  const auto status = capture(handle);
  if (!status) *handle = &provider;
  return status;
}
int cublasDestroy(cublasHandle_t handle) { return capture(handle); }
int cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
  const auto status = capture(handle, stream);
  if (!status) handle->stream = stream;
  return status;
}
int cublasSetPointerMode(cublasHandle_t handle, int mode) {
  const auto status = capture(handle, mode);
  if (!status) handle->pointer_mode = mode;
  return status;
}
int cublasDcopy(cublasHandle_t handle, int count, const double* source, int source_stride,
                double* destination, int destination_stride) {
  const auto status =
      capture(handle, count, source, source_stride, destination, destination_stride);
  if (numerical && !status)
    for (int index = 0; index < count; ++index)
      destination[index * destination_stride] = source[index * source_stride];
  return status;
}
int cublasDscal(cublasHandle_t handle, int count, const double* coefficient, double* values,
                int stride) {
  return capture(handle, count, coefficient, values, stride);
}
int cublasDaxpy(cublasHandle_t handle, int count, const double* coefficient, const double* source,
                int source_stride, double* destination, int destination_stride) {
  const auto status =
      capture(handle, count, coefficient, source, source_stride, destination, destination_stride);
  if (numerical && !status)
    for (int index = 0; index < count; ++index)
      destination[index * destination_stride] += *coefficient * source[index * source_stride];
  return status;
}
int cublasDdot(cublasHandle_t handle, int count, const double* left, int left_stride,
               const double* right, int right_stride, double* result) {
  return capture(handle, count, left, left_stride, right, right_stride, result);
}
int cublasDnrm2(cublasHandle_t handle, int count, const double* values, int stride,
                double* result) {
  return capture(handle, count, values, stride, result);
}
int cublasDgemm(cublasHandle_t handle, cublasOperation_t transpose_left,
                cublasOperation_t transpose_right, int rows, int columns, int reduction,
                const double* alpha, const double* left, int left_ld, const double* right,
                int right_ld, const double* beta, double* output, int output_ld) {
  return capture(handle, transpose_left, transpose_right, rows, columns, reduction, alpha, left,
                 left_ld, right, right_ld, beta, output, output_ld);
}
#ifdef GENERATIVEQC_TEST_HAS_GEAM
int cublasDgeam(cublasHandle_t handle, cublasOperation_t transpose_left,
                cublasOperation_t transpose_right, int rows, int columns, const double* alpha,
                const double* left, int left_ld, const double* beta, const double* right,
                int right_ld, double* output, int output_ld) {
  const auto status = capture(handle, transpose_left, transpose_right, rows, columns, alpha, left,
                              left_ld, beta, right, right_ld, output, output_ld);
  if (numerical && !status)
    for (int column = 0; column < columns; ++column)
      for (int row = 0; row < rows; ++row)
        output[row + column * output_ld] =
            *alpha * left[transpose_left == CUBLAS_OP_N ? row + column * left_ld
                                                        : column + row * left_ld] +
            *beta * right[transpose_right == CUBLAS_OP_N ? row + column * right_ld
                                                         : column + row * right_ld];
  return status;
}
#endif

int main() {
  using generativeqc::tensor::CudaDenseOperations;
  auto* stream = reinterpret_cast<void*>(0x71);
  const double coefficient = -0.5;
  const auto* input = reinterpret_cast<const double*>(0x9);
  auto* output = reinterpret_cast<double*>(0x19);
  CudaDenseOperations operations;
  failure = 3;
  assert(operations.create() == failure && !operations);
  assert(operations.close() == 0);
  failure = 0;
  assert(operations.create() == 0 && operations);
  assert(operations.create() == CUBLAS_STATUS_INVALID_VALUE);
  for (int failed_stage : {1, 2}) {
    provider = {};
    failure = 13;
    submitted = 0;
    fail_after = failed_stage;
    assert(operations.bind(stream) == failure);
    assert(provider.stream == (failed_stage == 1 ? nullptr : stream));
    assert(provider.pointer_mode == 71);
    assert(operations);
  }
  fail_after = failure = 0;
  assert(operations.bind(stream) == 0);
  assert(provider.stream == stream && provider.pointer_mode == CUBLAS_POINTER_MODE_HOST);

  const auto compare = [&](auto expected, auto actual) {
    const auto expected_status = expected();
    const auto expected_record = record;
    record.clear();
    assert(actual() == expected_status);
    assert(record == expected_record);
    assert(provider.stream == stream && provider.pointer_mode == CUBLAS_POINTER_MODE_HOST);
  };
  for (int status : {0, 3, 7, 13}) {
    failure = status;
    for (int count : {INT_MIN, -1, 0, 7, INT_MAX}) {
      compare([&] { return cublasDcopy(&provider, count, input, -3, output, 7); },
              [&] { return operations.copy(count, input, -3, output, 7); });
      compare([&] { return cublasDscal(&provider, count, &coefficient, output, 2); },
              [&] { return operations.scale(count, &coefficient, output, 2); });
      compare([&] { return cublasDaxpy(&provider, count, &coefficient, input, 3, output, -2); },
              [&] { return operations.add(count, &coefficient, input, 3, output, -2); });
      compare([&] { return cublasDdot(&provider, count, input, 3, input, -2, output); },
              [&] { return operations.dot(count, input, 3, input, -2, output); });
      compare([&] { return cublasDnrm2(&provider, count, input, 7, output); },
              [&] { return operations.norm(count, input, 7, output); });
      for (bool transpose_left : {false, true})
        for (bool transpose_right : {false, true})
          compare(
              [&] {
                return cublasDgemm(&provider, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N, count, 7, 3,
                                   &coefficient, input, 9, input, 11, &coefficient, output, 13);
              },
              [&] {
                return operations.product(transpose_left, transpose_right, count, 7, 3,
                                          &coefficient, input, 9, input, 11, &coefficient, output,
                                          13);
              });
    }
  }

  failure = 0;
  numerical = true;
  const std::array<double, 9> matrix{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<double, 9> actual{};
  std::uint64_t successful = 17;
  submitted = 0;
  assert(operations.transpose_sum(3, matrix.data(), actual.data(), &successful) == 0);
  assert(successful == 17 + static_cast<std::uint64_t>(submitted));
  const int dispatches = submitted;
  for (int column = 0; column < 3; ++column)
    for (int row = 0; row < 3; ++row)
      assert(actual[row + column * 3] == matrix[row + column * 3] + matrix[column + row * 3]);
  assert(operations.transpose_sum(3, matrix.data(), actual.data()) == 0);
  for (int invalid : {0, -1, INT_MAX})
    assert(operations.transpose_sum(invalid, matrix.data(), actual.data()) == 7);
  assert(operations.transpose_sum(3, actual.data(), actual.data()) == 7);
  assert(operations.transpose_sum(3, nullptr, actual.data()) == 7);
  assert(operations.transpose_sum(3, matrix.data(), nullptr) == 7);

  failure = 13;
  for (int failed_dispatch = 1; failed_dispatch <= dispatches; ++failed_dispatch) {
    submitted = 0;
    fail_after = failed_dispatch;
    successful = 17;
    const auto result = operations.transpose_sum(3, matrix.data(), actual.data(), &successful);
    assert(result == failure);
    assert(successful == 17 + static_cast<std::uint64_t>(submitted - 1));
  }
  fail_after = 0;
  assert(operations.close() == failure && !operations);
  assert(operations.close() == 0);
}
