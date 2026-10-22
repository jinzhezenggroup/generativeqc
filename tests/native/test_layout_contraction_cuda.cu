// Independent long-double oracle for the complete prepared layout-copy endpoint.
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "generated_layout_portfolios.hpp"
#include "tensor/cuda_layout_contraction.cuh"

using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;

struct Case {
  const generativeqc::runtime::NativeLoweringRequest* request;
  std::span<const generativeqc::runtime::NativeLoweringCandidate> candidates;
  std::string_view target, compilation;
  LayoutContractionPlan (*plan)(std::size_t);
};

template <class F>
void reject(F&& action) {
  bool caught = false;
  try {
    action();
  } catch (const std::logic_error&) {
    caught = true;
  }
  if (!caught) throw std::runtime_error("invalid layout endpoint was admitted");
}

struct Buffers {
  double *a{}, *b{}, *c{};
  void* scratch{};
  int* error{};
  Buffers(std::size_t aa, std::size_t bb, std::size_t cc, std::size_t scratch_bytes) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&a), aa));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b), bb));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&c), cc));
    cuda_check(cudaMalloc(&scratch, scratch_bytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
    cuda_check(cudaMemset(error, 0, sizeof(int)));
  }
  ~Buffers() {
    (void)cudaFree(a);
    (void)cudaFree(b);
    (void)cudaFree(c);
    (void)cudaFree(scratch);
    (void)cudaFree(error);
  }
};

std::size_t offset(const ContractionOperand& view, std::size_t row, std::size_t column) {
  return row * view.strides[0] + column * view.strides[1];
}

double run_case(const Case& c, std::size_t selected, CudaContractionContext& context) {
  const auto p = c.plan(selected);
  const auto& r = p.original;
  const auto& dims = p.physical;
  const auto a_size = r.operands[0].storage_elements();
  const auto b_size = r.operands[1].storage_elements();
  const auto c_size = r.operands[2].storage_elements();
  constexpr double padding = -987654.25;
  std::vector<double> a(a_size, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> b(b_size, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> output(c_size, padding);
  for (std::size_t i = 0; i < dims.m; ++i)
    for (std::size_t k = 0; k < dims.k; ++k)
      a[offset(r.operands[0], i, k)] = (double(i * 11 + k) - 13.0) / 37.0;
  for (std::size_t k = 0; k < dims.k; ++k)
    for (std::size_t j = 0; j < dims.n; ++j)
      b[offset(r.operands[1], k, j)] = (double(k * 3 + j) - 8.0) / 29.0;
  for (std::size_t i = 0; i < dims.m; ++i)
    for (std::size_t j = 0; j < dims.n; ++j)
      output[offset(r.operands[2], i, j)] =
          r.beta ? (double(i + j) + 1) / 19.0 : std::numeric_limits<double>::quiet_NaN();
  Buffers buffers(a_size * 8, b_size * 8, c_size * 8, p.scratch_bytes());
  auto stream = context.stream();
  cuda_check(cudaMemcpyAsync(buffers.a, a.data(), a_size * 8, cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(buffers.b, b.data(), b_size * 8, cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(buffers.c, output.data(), c_size * 8, cudaMemcpyHostToDevice, stream));
  const auto& candidate = c.candidates[selected];
  reject([&] {
    CudaLayoutContraction bad(p, *c.request, candidate, c.target, c.compilation, context, 0, 32768);
  });
  reject([&] {
    CudaLayoutContraction bad(p, *c.request, candidate, "stale", c.compilation, context, 128 << 20,
                              32768);
  });
  CudaLayoutContraction executor(p, *c.request, candidate, c.target, c.compilation, context,
                                 128 << 20, 32768);
  const auto execute = [&](std::size_t capacity, void* scratch, int* error, double* destination) {
    executor.execute(stream, buffers.a, a_size * 8, buffers.b, b_size * 8, destination, c_size * 8,
                     scratch, capacity, error);
  };
  reject([&] { execute(p.scratch_bytes() - 1, buffers.scratch, buffers.error, buffers.c); });
  reject([&] { execute(p.scratch_bytes(), buffers.c, buffers.error, buffers.c); });
  reject([&] {
    execute(p.scratch_bytes(), buffers.scratch, reinterpret_cast<int*>(buffers.a), buffers.c);
  });
  reject([&] { execute(p.scratch_bytes(), buffers.scratch, buffers.error, buffers.a); });
  cudaStream_t other{};
  cuda_check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  reject([&] {
    executor.execute(other, buffers.a, a_size * 8, buffers.b, b_size * 8, buffers.c, c_size * 8,
                     buffers.scratch, p.scratch_bytes(), buffers.error);
  });
  cuda_check(cudaStreamDestroy(other));
  reject([&] {
    executor.execute(stream, a.data(), a_size * 8, buffers.b, b_size * 8, buffers.c, c_size * 8,
                     buffers.scratch, p.scratch_bytes(), buffers.error);
  });
  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  reject([&] {
    CudaLayoutContraction during(p, *c.request, candidate, c.target, c.compilation, context,
                                 128 << 20, 32768);
  });
  reject([&] { execute(p.scratch_bytes(), buffers.scratch, buffers.error, buffers.c); });
  cudaGraph_t graph{};
  cuda_check(cudaStreamEndCapture(stream, &graph));
  if (graph) cuda_check(cudaGraphDestroy(graph));
  if (executor.calls() || executor.copy_launches())
    throw std::runtime_error("host-rejected endpoint enqueued work");
  std::vector<double> unchanged(c_size);
  cuda_check(
      cudaMemcpyAsync(unchanged.data(), buffers.c, c_size * 8, cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (std::memcmp(unchanged.data(), output.data(), c_size * 8))
    throw std::runtime_error("rejected endpoint changed output");
  double maximum_error = 0;
  for (int replay = 0; replay < 3; ++replay) {
    if (replay == 2) {
      for (std::size_t i = 0; i < dims.m; ++i)
        for (std::size_t k = 0; k < dims.k; ++k)
          a[offset(r.operands[0], i, k)] += 0.001 * (double(i) - double(k));
      cuda_check(cudaMemcpyAsync(buffers.a, a.data(), a_size * 8, cudaMemcpyHostToDevice, stream));
    }
    std::vector<long double> expected(dims.m * dims.n);
    for (std::size_t i = 0; i < dims.m; ++i)
      for (std::size_t j = 0; j < dims.n; ++j) {
        long double value = 0;
        for (std::size_t k = 0; k < dims.k; ++k)
          value += (long double)a[offset(r.operands[0], i, k)] * b[offset(r.operands[1], k, j)];
        expected[i * dims.n + j] =
            (long double)r.coefficient * value +
            (r.beta ? (long double)r.beta * output[offset(r.operands[2], i, j)] : 0);
      }
    execute(p.scratch_bytes(), buffers.scratch, buffers.error, buffers.c);
    cuda_check(
        cudaMemcpyAsync(output.data(), buffers.c, c_size * 8, cudaMemcpyDeviceToHost, stream));
    int error = -1;
    cuda_check(cudaMemcpyAsync(&error, buffers.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (error) throw std::runtime_error("layout endpoint arithmetic audit failed");
    std::vector<bool> logical(c_size);
    for (std::size_t i = 0; i < dims.m; ++i)
      for (std::size_t j = 0; j < dims.n; ++j) {
        auto x = offset(r.operands[2], i, j);
        logical[x] = true;
        const auto difference = std::abs((long double)output[x] - expected[i * dims.n + j]);
        maximum_error = std::max(maximum_error, double(difference));
        if (!std::isfinite(output[x]) ||
            difference > 3e-13L * (1 + std::abs(expected[i * dims.n + j])))
          throw std::runtime_error("layout endpoint differs from independent long-double oracle");
      }
    for (std::size_t i = 0; i < c_size; ++i)
      if (!logical[i] && output[i] != padding)
        throw std::runtime_error("layout publication overwrote padding");
  }
  if (executor.calls() != 3 || executor.summands() != 3 * dims.summands() ||
      executor.copy_launches() != 3 * p.copy_launches() ||
      executor.transfer_bytes() != 3 * p.transfer_bytes() ||
      executor.transfer_elements() != 3 * p.transfer_elements())
    throw std::runtime_error("complete layout endpoint work accounting differs");
  std::cout << "case provider=" << candidate.provider << " beta=" << r.beta
            << " copies=" << executor.copy_launches() << " bytes=" << executor.transfer_bytes()
            << " scratch=" << p.scratch_bytes() << " summands=" << executor.summands()
            << " max_error=" << std::setprecision(17) << maximum_error << " PASS\n";
  return maximum_error;
}

int main() {
  try {
    cudaDeviceProp device_properties{};
    cuda_check(cudaGetDeviceProperties(&device_properties, 0));
    std::cout << "device=" << device_properties.name << " capability=" << device_properties.major
              << "." << device_properties.minor << "\n";
    const std::array<Case, 8> cases{{
        {&case0_request, case0_candidates, case0_target, case0_compilation, case0_plan},
        {&case1_request, case1_candidates, case1_target, case1_compilation, case1_plan},
        {&case2_request, case2_candidates, case2_target, case2_compilation, case2_plan},
        {&case3_request, case3_candidates, case3_target, case3_compilation, case3_plan},
        {&case4_request, case4_candidates, case4_target, case4_compilation, case4_plan},
        {&case5_request, case5_candidates, case5_target, case5_compilation, case5_plan},
        {&case6_request, case6_candidates, case6_target, case6_compilation, case6_plan},
        {&case7_request, case7_candidates, case7_target, case7_compilation, case7_plan},
    }};
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CudaContractionContext context;
    if (!context.prepare(stream)) throw std::runtime_error("cuBLAS preparation unavailable");
    std::cout << "runtime=" << context.runtime_version() << " cublas=" << context.provider_version()
              << "\n";
    double error = 0;
    for (const auto& c : cases) {
      const auto decision = generativeqc::runtime::select_native_lowering(*c.request, c.candidates,
                                                                          c.target, c.compilation);
      if (decision.selected != 1 || decision.fallbacks.size() != 1 || decision.fallbacks[0] != 0)
        throw std::runtime_error(
            "joint selection/fallback did not preserve the generated candidate");
      for (std::size_t i = 0; i < 2; ++i) error = std::max(error, run_case(c, i, context));
    }
    context.reset();
    context.prepare_generated(stream);
    const auto& c = cases.back();
    auto p = c.plan(1);
    bool unavailable = false;
    try {
      CudaLayoutContraction trial(p, *c.request, c.candidates[1], c.target, c.compilation, context,
                                  128 << 20, 32768);
    } catch (const ContractionPreparationUnavailable&) {
      unavailable = true;
    }
    if (!unavailable) throw std::runtime_error("uncharged/unavailable provider was admitted");
    error = std::max(error, run_case(c, 0, context));
    auto generated = c.plan(0);
    {
      CudaLayoutContraction old(generated, *c.request, c.candidates[0], c.target, c.compilation,
                                context, 128 << 20, 32768);
      context.reset();
      context.prepare_generated(stream);
      reject([&] { old.execute(stream, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr); });
    }
    error = std::max(error, run_case(c, 0, context));
    context.reset();
    cuda_check(cudaStreamDestroy(stream));
    std::cout << "LAYOUT_GPU_PASS cases=18 ordinary_replays=54 max_error=" << std::setprecision(17)
              << error << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
