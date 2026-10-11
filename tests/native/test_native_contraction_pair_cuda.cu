// Qualify only independent-pair execution; ordinary matrix tests remain separate.
#include <cstring>
#include <iostream>
#include <vector>

#include "tensor/cuda_contraction.cuh"

using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;

namespace {
struct DeviceBuffer {
  void* pointer{};
  explicit DeviceBuffer(std::size_t bytes) { cuda_check(cudaMalloc(&pointer, bytes)); }
  ~DeviceBuffer() { cudaFree(pointer); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  double* doubles() const { return static_cast<double*>(pointer); }
};

template <class Function>
void refused(Function&& function) {
  try {
    function();
  } catch (const std::logic_error&) {
    return;
  }
  throw std::runtime_error("independent pair accepted an invalid binding");
}

/** Dyadic inputs give an exact independent long-double oracle for all four
 * physical transpose combinations. NaN padding must never enter the audit. */
void check_pair(char left_transpose, char right_transpose, double beta, unsigned layout,
                bool shared_left, std::size_t columns = 5) {
  constexpr std::size_t rows = 3, reduction = 7;
  constexpr std::string_view identity =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const auto dtype = PrecisionDtype::Fp64;
  const auto left_operand = left_transpose == 'N'
                                ? ContractionOperand::dense({3, 0, 2}, {1, rows, reduction}, dtype)
                                : ContractionOperand::dense({3, 2, 0}, {1, reduction, rows}, dtype);
  const auto right_operand =
      right_transpose == 'N' ? ContractionOperand::dense({3, 2, 1}, {1, reduction, columns}, dtype)
                             : ContractionOperand::dense({3, 1, 2}, {1, columns, reduction}, dtype);
  ContractionRequest request{identity,
                             identity,
                             identity,
                             {left_operand, right_operand,
                              ContractionOperand::dense({3, 0, 1}, {1, rows, columns}, dtype)},
                             {dtype, dtype, dtype},
                             dtype,
                             left_transpose,
                             right_transpose,
                             1,
                             rows,
                             columns,
                             reduction,
                             -0.75};
  request.beta = beta;
  if (layout == 2)
    for (std::size_t operand = 0; operand < 3; ++operand) {
      const auto leading = request.operands[operand].shape[2] + 3;
      request.leading_dimensions[operand] = leading;
      request.operands[operand].strides[1] = leading;
      request.operands[operand].strides[0] = request.operands[operand].shape[1] * leading;
    }
  const auto left_leading = request.leading_dimension(0);
  const auto right_leading = request.leading_dimension(1);
  const auto output_leading = request.leading_dimension(2);
  const auto left_count = request.operands[0].storage_elements();
  const auto right_count = request.operands[1].storage_elements();
  const auto output_count = request.operands[2].storage_elements();
  const auto output_step = output_count + (layout == 1 ? 11 : 0);
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> left(2 * left_count, nan), right(2 * right_count, nan);
  std::vector<double> outputs(output_step + output_count, nan), expected(outputs);
  for (std::size_t product = 0; product < 2; ++product) {
    for (std::size_t row = 0; row < request.operands[0].shape[1]; ++row)
      for (std::size_t col = 0; col < request.operands[0].shape[2]; ++col)
        left[product * left_count + row * left_leading + col] =
            double(int((row * 11 + col + product) % 17) - 8) / 16;
    for (std::size_t row = 0; row < request.operands[1].shape[1]; ++row)
      for (std::size_t col = 0; col < request.operands[1].shape[2]; ++col)
        right[product * right_count + row * right_leading + col] =
            double(int((row * 7 + col + product) % 13) - 6) / 16;
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < columns; ++col) {
        long double sum = 0;
        for (std::size_t inner = 0; inner < reduction; ++inner) {
          const auto left_index =
              (shared_left ? 0 : product * left_count) +
              (left_transpose == 'N' ? row * left_leading + inner : inner * left_leading + row);
          const auto right_index =
              product * right_count +
              (right_transpose == 'N' ? inner * right_leading + col : col * right_leading + inner);
          sum += static_cast<long double>(left[left_index]) * right[right_index];
        }
        const auto output_index = product * output_step + row * output_leading + col;
        outputs[output_index] = 1;
        expected[output_index] = double(-0.75L * sum + beta);
      }
  }
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    CudaContractionContext provider;
    if (!provider.prepare(stream, CudaContractionContext::kOptionalWorkspaceBytes))
      throw std::runtime_error("pair provider preparation failed");
    PreparedContractions bindings;
    std::size_t calls{}, summands{};
    bindings.add(rows, columns, reduction, {request}, provider, calls, summands,
                 {ContractionAlgorithm::PedanticBlas});
    DeviceBuffer device_left(left.size() * sizeof(double)),
        device_right(right.size() * sizeof(double)), device_output(outputs.size() * sizeof(double)),
        device_error(sizeof(int)), scratch(sizeof(IndependentPairPointers));
    auto* error = static_cast<int*>(device_error.pointer);
    cuda_check(cudaMemcpyAsync(device_left.pointer, left.data(), left.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_right.pointer, right.data(), right.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_output.pointer, outputs.data(),
                               outputs.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(scratch.pointer, 0xa5, sizeof(IndependentPairPointers), stream));
    auto* first_output = device_output.doubles();
    auto* second_output = first_output + output_step;
    const auto* first_left = device_left.doubles();
    const auto* second_left = first_left + (shared_left ? 0 : left_count);
    const auto* first_right = device_right.doubles();
    const auto* second_right = first_right + right_count;
    const auto execute = [&](const double* other_left, const double* other_right, double* out_first,
                             double* out_second, int* arithmetic, void* table,
                             std::size_t table_bytes) {
      bindings.execute_independent_pair(0, rows, columns, reduction, stream, first_left, other_left,
                                        first_right, other_right, out_first, out_second, arithmetic,
                                        table, table_bytes);
    };
    const auto run = [&] {
      execute(second_left, second_right, first_output, second_output, error, scratch.pointer,
              sizeof(IndependentPairPointers));
    };
    std::vector<double> snapshot(outputs.size());
    std::vector<unsigned char> scratch_before(sizeof(IndependentPairPointers)),
        scratch_after(scratch_before);
    cuda_check(cudaMemcpyAsync(scratch_before.data(), scratch.pointer, scratch_before.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    // Cross-product aliases are dangerous even when each individual request
    // appears locally disjoint. Every refusal must precede the address binder.
    refused([&] {
      execute(nullptr, second_right, first_output, second_output, error, scratch.pointer, 48);
    });
    refused([&] {
      execute(second_left, second_right, first_output, first_output + 1, error, scratch.pointer,
              48);
    });
    refused([&] {
      execute(second_left, second_right, const_cast<double*>(second_left), second_output, error,
              scratch.pointer, 48);
    });
    refused([&] {
      execute(second_left, second_right, first_output, const_cast<double*>(first_right), error,
              scratch.pointer, 48);
    });
    refused([&] {
      execute(second_left, second_right, first_output, second_output, error, scratch.pointer, 47);
    });
    refused([&] {
      execute(second_left, second_right, first_output, second_output, error,
              static_cast<unsigned char*>(scratch.pointer) + 1, 48);
    });
    for (auto* alias :
         {device_left.pointer, device_right.pointer, device_output.pointer, device_error.pointer})
      refused([&] {
        execute(second_left, second_right, first_output, second_output, error, alias, 48);
      });
    for (auto* alias : {device_left.pointer, device_right.pointer, device_output.pointer})
      refused([&] {
        execute(second_left, second_right, first_output, second_output, static_cast<int*>(alias),
                scratch.pointer, 48);
      });
    calls = std::numeric_limits<std::size_t>::max();
    refused(run);
    if (calls != std::numeric_limits<std::size_t>::max() || summands)
      throw std::runtime_error("dispatch overflow changed counters");
    calls = 0;
    summands = std::numeric_limits<std::size_t>::max() - request.affine_summands();
    refused(run);
    if (calls || summands != std::numeric_limits<std::size_t>::max() - request.affine_summands())
      throw std::runtime_error("summand overflow changed counters");
    summands = 0;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    refused(run);
    refused([&] { bindings.independent_pair_supported(0, rows, columns, reduction, stream); });
    cudaGraph_t graph{};
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphDestroy(graph));
    refused([&] { bindings.independent_pair_supported(1, rows, columns, reduction, stream); });
    refused([&] { bindings.independent_pair_supported(0, rows, columns, reduction + 1, stream); });
    refused([&] { bindings.independent_pair_supported(0, rows, columns, reduction, nullptr); });
    int status{};
    cuda_check(cudaMemcpyAsync(snapshot.data(), device_output.pointer,
                               snapshot.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(scratch_after.data(), scratch.pointer, scratch_after.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (calls || summands || status || scratch_before != scratch_after ||
        std::memcmp(outputs.data(), snapshot.data(), outputs.size() * sizeof(double)))
      throw std::runtime_error("pre-enqueue refusal changed scientific state");
    if (!bindings.independent_pair_supported(0, rows, columns, reduction, stream))
      throw std::runtime_error("strict ordinary request did not offer pairing");
    run();
    cuda_check(cudaMemcpyAsync(snapshot.data(), device_output.pointer,
                               snapshot.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (calls != 1 || summands != 2 * rows * columns * reduction || status)
      throw std::runtime_error("pair diagnostic or finite-audit mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index)
      if (std::isnan(expected[index]) ? !std::isnan(snapshot[index])
                                      : snapshot[index] != expected[index])
        throw std::runtime_error("pair disagrees with independent long-double oracle");
    // Only the second output's final value is nonfinite, including tail lanes.
    right[right_count + (right_transpose == 'N' ? (reduction - 1) * right_leading + columns - 1
                                                : (columns - 1) * right_leading + reduction - 1)] =
        std::numeric_limits<double>::infinity();
    cuda_check(cudaMemcpyAsync(device_right.pointer, right.data(), right.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    run();
    cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (!status) throw std::runtime_error("second pair output escaped the finite audit");
    status = 7;
    cuda_check(cudaMemcpyAsync(error, &status, sizeof(int), cudaMemcpyHostToDevice, stream));
    run();
    cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (status != 7) throw std::runtime_error("pair overwrote sticky arithmetic failure");
    provider.reset();
    refused(run);
    refused([&] { bindings.independent_pair_supported(0, rows, columns, reduction, stream); });
  }
  cuda_check(cudaStreamDestroy(stream));
}

void check_unsupported() {
  constexpr std::string_view identity =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    CudaContractionContext provider;
    if (!provider.prepare(stream))
      throw std::runtime_error("unsupported fixture preparation failed");
    for (const auto dtype : {PrecisionDtype::Fp64, PrecisionDtype::Fp32})
      for (std::size_t batches : {1, 2})
        for (auto algorithm :
             {ContractionAlgorithm::PedanticBlas, ContractionAlgorithm::GeneratedOrdered}) {
          if (dtype == PrecisionDtype::Fp64 && batches == 1 &&
              algorithm == ContractionAlgorithm::PedanticBlas)
            continue;
          ContractionRequest request{identity,
                                     identity,
                                     identity,
                                     {ContractionOperand::dense({3, 0, 2}, {batches, 3, 7}, dtype),
                                      ContractionOperand::dense({3, 2, 1}, {batches, 7, 5}, dtype),
                                      ContractionOperand::dense({3, 0, 1}, {batches, 3, 5}, dtype)},
                                     {dtype, dtype, dtype},
                                     dtype,
                                     'N',
                                     'N',
                                     batches,
                                     3,
                                     5,
                                     7,
                                     1.0};
          PreparedContractions bindings;
          std::size_t calls{}, summands{};
          bindings.add(3, 5, 7, {request}, provider, calls, summands, {algorithm});
          if (bindings.independent_pair_supported(0, 3, 5, 7, stream))
            throw std::runtime_error("unsupported request offered a pair");
          refused([&] {
            bindings.execute_independent_pair(0, 3, 5, 7, stream, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, 0);
          });
          if (calls || summands) throw std::runtime_error("unsupported request changed counters");
        }
  }
  cuda_check(cudaStreamDestroy(stream));
}
}  // namespace

int main() {
  try {
    for (const char left : {'N', 'T'})
      for (const char right : {'N', 'T'})
        for (const double beta : {0.0, 1.0})
          for (unsigned layout : {0, 1, 2})
            for (const bool shared : {false, true}) check_pair(left, right, beta, layout, shared);
    check_pair('N', 'T', 1.0, 0, true, 97);
    check_pair('T', 'N', 0.0, 2, false, 97);
    check_unsupported();
    std::cout << "independent-pair: 50 oracle/layout cases and 7 unsupported cases passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
