// Standalone device qualification for the prepared rank-k capability.
// Compile with the generated portfolio include path and -lcublas.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated_symmetric_rank_k.cuh"
#include "tensor/cuda_symmetric_rank_k.cuh"

using namespace generativeqc::tensor;
namespace metadata = generativeqc::tensor::rank_k_generated;

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
  }
  ~DeviceBuffer() { if (pointer_) (void)cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return pointer_; }
  std::size_t bytes() const { return count_ * sizeof(T); }
 private:
  std::size_t count_{};
  T* pointer_{};
};

__global__ void materialize_weights(const double* occupations, const double* energies,
                                    double* weights, std::size_t count, int* error) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    double value{};
    if (!metadata::rank_k_energy_weight(occupations[i], energies[i], value)) {
      atomicCAS(error, 0, 1);
      value = 0.0;
    }
    weights[i] = value;
  }
}

static std::size_t panel_index(std::size_t batch, std::size_t row, std::size_t orbital,
                               std::size_t n, std::size_t k, RankKOrder order) {
  return batch * n * k +
      (order == RankKOrder::RowMajor ? row * k + orbital : row + orbital * n);
}

static std::size_t matrix_index(std::size_t batch, std::size_t row, std::size_t col,
                                std::size_t n, RankKOrder order) {
  return batch * n * n +
      (order == RankKOrder::RowMajor ? row * n + col : row + col * n);
}

static const std::array<generativeqc::runtime::NativeLoweringCandidate, 2>&
candidates(bool weighted, RankKOrder order) {
  if (weighted)
    return order == RankKOrder::RowMajor ? metadata::rank_k_weighted_density_row_candidates :
                                           metadata::rank_k_weighted_density_column_candidates;
  return order == RankKOrder::RowMajor ? metadata::rank_k_density_row_candidates :
                                         metadata::rank_k_density_column_candidates;
}

static const generativeqc::runtime::NativeLoweringRequest& request(bool weighted,
                                                                    RankKOrder order) {
  if (weighted)
    return order == RankKOrder::RowMajor ? metadata::rank_k_weighted_density_row_request :
                                           metadata::rank_k_weighted_density_column_request;
  return order == RankKOrder::RowMajor ? metadata::rank_k_density_row_request :
                                         metadata::rank_k_density_column_request;
}

static std::string_view target(bool weighted, RankKOrder order) {
  if (weighted)
    return order == RankKOrder::RowMajor ? metadata::rank_k_weighted_density_row_target :
                                           metadata::rank_k_weighted_density_column_target;
  return order == RankKOrder::RowMajor ? metadata::rank_k_density_row_target :
                                         metadata::rank_k_density_column_target;
}

static std::string_view compilation(bool weighted, RankKOrder order) {
  if (weighted)
    return order == RankKOrder::RowMajor ? metadata::rank_k_weighted_density_row_compilation :
                                           metadata::rank_k_weighted_density_column_compilation;
  return order == RankKOrder::RowMajor ? metadata::rank_k_density_row_compilation :
                                         metadata::rank_k_density_column_compilation;
}

static void verify(const std::vector<double>& result, const std::vector<double>& baseline,
                   const std::vector<double>& coefficients, const std::vector<double>& weights,
                   std::size_t n, std::size_t k, std::size_t batches, RankKOrder order,
                   double alpha, double beta) {
  for (std::size_t batch = 0; batch < batches; ++batch)
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t col = 0; col < n; ++col) {
        long double sum = 0;
        for (std::size_t orbital = 0; orbital < k; ++orbital)
          sum += static_cast<long double>(coefficients[panel_index(batch, row, orbital, n, k, order)]) *
                 static_cast<long double>(weights[batch * k + orbital]) *
                 static_cast<long double>(coefficients[panel_index(batch, col, orbital, n, k, order)]);
        const auto old = baseline[matrix_index(batch, std::min(row, col), std::max(row, col), n, order)];
        const auto expected = static_cast<double>(alpha * sum + beta * static_cast<long double>(old));
        const auto actual = result[matrix_index(batch, row, col, n, order)];
        const auto tolerance = 2e-11 * std::max(1.0, std::abs(expected));
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
          throw std::runtime_error("rank-k high-precision oracle mismatch");
      }
}

static void run_case(std::size_t n, std::size_t k, std::size_t batches,
                     RankKOrder order, bool weighted, bool want_library) {
  const auto panel_count = batches * n * k, matrix_count = batches * n * n;
  std::vector<double> coefficients(panel_count), occupations(batches * k),
      energies(batches * k), weights(batches * k), baseline(matrix_count);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t orbital = 0; orbital < k; ++orbital)
        coefficients[panel_index(batch, row, orbital, n, k, order)] =
            (static_cast<double>((row * 7 + orbital * 11 + batch * 3) % 29) - 14.0) / 19.0;
    for (std::size_t orbital = 0; orbital < k; ++orbital) {
      occupations[batch * k + orbital] = orbital % 5 == 0 ? 0.0 : 0.5 + 0.125 * (orbital % 3);
      energies[batch * k + orbital] = orbital == 0 ? -0.0 :
          (static_cast<double>((orbital * 7 + batch) % 11) - 6.0) / 3.0;
      weights[batch * k + orbital] = weighted ?
          occupations[batch * k + orbital] * energies[batch * k + orbital] :
          (orbital % 4 == 0 ? -0.75 : occupations[batch * k + orbital]);
    }
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t col = 0; col < n; ++col)
        baseline[matrix_index(batch, row, col, n, order)] =
            0.125 * (1 + batch + std::min(row, col) + std::max(row, col)) +
            (row > col ? 2.0 : 0.0);
  }
  DeviceBuffer<double> d_coefficients(panel_count), d_weights(batches * k),
      d_occupations(batches * k), d_energies(batches * k),
      d_output(matrix_count), d_baseline(matrix_count);
  DeviceBuffer<int> d_error(1);
  check(cudaMemcpy(d_coefficients.get(), coefficients.data(), d_coefficients.bytes(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_weights.get(), weights.data(), d_weights.bytes(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_occupations.get(), occupations.data(), d_occupations.bytes(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_energies.get(), energies.data(), d_energies.bytes(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_baseline.get(), baseline.data(), d_baseline.bytes(), cudaMemcpyHostToDevice));
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const double alpha = 1.25, beta = -0.5;
  SymmetricRankKInvocation invocation{n, k, batches, d_coefficients.get(), d_weights.get(),
                                       d_output.get(), d_error.get(), alpha, beta};
  bool rejected_order = false;
  try {
    CudaSymmetricRankK wrong(request(weighted, order), candidates(weighted, order),
                              target(weighted, order), compilation(weighted, order),
                              n, k, batches,
                              order == RankKOrder::RowMajor ? RankKOrder::ColumnMajor : RankKOrder::RowMajor,
                              stream, 0);
  } catch (const std::invalid_argument&) { rejected_order = true; }
  if (!rejected_order) throw std::runtime_error("rank-k accepted wrong physical order");
  {
    CudaSymmetricRankK binding(request(weighted, order), candidates(weighted, order),
                                target(weighted, order), compilation(weighted, order),
                                n, k, batches, order, stream,
                                want_library ? 256ULL << 20 : 0, want_library);
    const auto& diagnostic = binding.diagnostic();
    if ((diagnostic.selected.provider == "cublas") != want_library)
      throw std::runtime_error("rank-k selected wrong executable provider");
    auto enqueue = [&] {
      check(cudaMemcpyAsync(d_output.get(), d_baseline.get(), d_output.bytes(),
                            cudaMemcpyDeviceToDevice, stream));
      check(cudaMemsetAsync(d_error.get(), 0, sizeof(int), stream));
      if (weighted)
        materialize_weights<<<generativeqc_tensor::blocks(batches * k, 128), 128, 0, stream>>>(
            d_occupations.get(), d_energies.get(), d_weights.get(), batches * k,
            d_error.get());
      binding.execute(stream, invocation);
    };
    enqueue();
    check(cudaStreamSynchronize(stream));
    int error{};
    check(cudaMemcpy(&error, d_error.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (error) throw std::runtime_error("rank-k valid case marked nonfinite");
    std::vector<double> result(matrix_count);
    check(cudaMemcpy(result.data(), d_output.get(), d_output.bytes(), cudaMemcpyDeviceToHost));
    verify(result, baseline, coefficients, weights, n, k, batches, order, alpha, beta);

    // Captured replay reinitializes both output and error on the caller's stream.
    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 2; ++repeat) {
      check(cudaGraphLaunch(graph_exec, stream));
      check(cudaStreamSynchronize(stream));
      check(cudaMemcpy(&error, d_error.get(), sizeof(int), cudaMemcpyDeviceToHost));
      if (error) throw std::runtime_error("rank-k captured replay marked nonfinite");
      check(cudaMemcpy(result.data(), d_output.get(), d_output.bytes(), cudaMemcpyDeviceToHost));
      verify(result, baseline, coefficients, weights, n, k, batches, order, alpha, beta);
    }
    check(cudaGraphExecDestroy(graph_exec));
    check(cudaGraphDestroy(graph));

    // A nonfinite signed weight must leave the entire output intact.
    weights[0] = std::numeric_limits<double>::quiet_NaN();
    check(cudaMemcpy(d_weights.get(), weights.data(), d_weights.bytes(), cudaMemcpyHostToDevice));
    check(cudaMemcpyAsync(d_output.get(), d_baseline.get(), d_output.bytes(), cudaMemcpyDeviceToDevice, stream));
    check(cudaMemsetAsync(d_error.get(), 0, sizeof(int), stream));
    binding.execute(stream, invocation);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&error, d_error.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (!error) throw std::runtime_error("rank-k nonfinite weight was accepted");
    check(cudaMemcpy(result.data(), d_output.get(), d_output.bytes(), cudaMemcpyDeviceToHost));
    if (result != baseline) throw std::runtime_error("rank-k failure changed output");
    weights[0] = weighted ? occupations[0] * energies[0] : -0.75;
    check(cudaMemcpy(d_weights.get(), weights.data(), d_weights.bytes(), cudaMemcpyHostToDevice));

    bool rejected_alias = false;
    auto alias = invocation;
    alias.output = d_coefficients.get();
    try { binding.execute(stream, alias); }
    catch (const std::invalid_argument&) { rejected_alias = true; }
    if (!rejected_alias) throw std::runtime_error("rank-k output alias was accepted");

    cudaEvent_t begin{}, end{};
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int warmup = 0; warmup < 4; ++warmup) enqueue();
    check(cudaStreamSynchronize(stream));
    check(cudaEventRecord(begin, stream));
    constexpr int repetitions = 20;
    for (int repeat = 0; repeat < repetitions; ++repeat) enqueue();
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, begin, end));
    check(cudaEventDestroy(end));
    check(cudaEventDestroy(begin));
    const auto upper = batches * n * (n + 1) / 2;
    const auto products = want_library ? batches * n * n * k : 2 * upper * k;
    std::cout << std::setprecision(9)
              << "{\"status\":\"PASS\",\"n\":" << n << ",\"k\":" << k
              << ",\"batches\":" << batches << ",\"weighted\":" << weighted
              << ",\"order\":\"" << (order == RankKOrder::RowMajor ? "row" : "column")
              << "\",\"provider\":\"" << diagnostic.selected.provider
              << "\",\"endpoint_us\":" << (milliseconds * 1000.0 / repetitions)
              << ",\"semantic_products\":" << products
              << ",\"scale_elements\":" << (want_library ? panel_count : 0)
              << ",\"mirror_elements\":" << (upper - batches * n)
              << ",\"temporary_bytes\":" << diagnostic.temporary_bytes
              << ",\"provider_allowance\":" << diagnostic.provider_allowance
              << ",\"provider_retained\":" << diagnostic.retained_provider_bytes
              << ",\"provider_version\":" << diagnostic.provider_version
              << ",\"scientific_identity\":\"" << request(weighted, order).scientific_identity
              << "\"}" << std::endl;
  }
  check(cudaStreamDestroy(stream));
}

int main() {
  try {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    std::cout << "{\"device\":\"" << device.name << "\",\"major\":" << device.major
              << ",\"minor\":" << device.minor << "}" << std::endl;
    for (const bool weighted : {false, true})
      for (const auto order : {RankKOrder::RowMajor, RankKOrder::ColumnMajor})
        for (const bool library : {false, true}) {
          run_case(3, 5, 2, order, weighted, library);
          run_case(17, 9, 2, order, weighted, library);
        }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank-k qualification failed: " << error.what() << std::endl;
    return 1;
  }
}
