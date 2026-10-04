#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "tensor/cuda_cutensor.cuh"

using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;

template <class F>
void reject(F call) {
  try {
    call();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("invalid binding was accepted");
}

template <class T>
void check(double beta) {
  constexpr auto dtype = std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32;
  constexpr std::string_view identity =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  ContractionRequest request{identity,
                             identity,
                             identity,
                             {ContractionOperand::dense({0, 2, 3}, {2, 3, 4}, dtype),
                              ContractionOperand::dense({3, 2, 1}, {4, 3, 5}, dtype),
                              ContractionOperand::dense({1, 0}, {5, 2}, dtype)},
                             {dtype, dtype, dtype},
                             dtype};
  request.coefficient = -0.75;
  request.beta = beta;
  // Reduction axes have opposite physical order, the output is transposed,
  // and both input/output rows contain nonfinite padding. A dense GEMM recipe
  // cannot execute this request without packing; cuTENSOR consumes its modes.
  request.operands[0].strides = {20, 6, 1};
  request.operands[2].strides = {4, 1};
  request.validate_affine();
  if (request.affine_summands() != 120) throw std::runtime_error("affine semantic work count");
  reject([&] { request.validate(); });
  auto bad = request;
  bad.operands[0].strides[0] = 1;
  reject([&] { bad.validate_affine(); });
  bad = request;
  bad.operands[1].shape[0] += 1;
  reject([&] { bad.validate_affine(); });

  const T nan = std::numeric_limits<T>::quiet_NaN();
  std::vector<T> a(request.operands[0].storage_elements(), nan),
      b(request.operands[1].storage_elements()),
      initial(request.operands[2].storage_elements(), nan), expected(initial);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t l = 0; l < 4; ++l)
        a[i * 20 + k * 6 + l] = T(int(i * 12 + k * 4 + l) - 10) / 16;
  for (std::size_t l = 0; l < 4; ++l)
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t j = 0; j < 5; ++j)
        b[l * 15 + k * 5 + j] = T(int(l * 15 + k * 5 + j) - 20) / 32;
  for (std::size_t j = 0; j < 5; ++j)
    for (std::size_t i = 0; i < 2; ++i) {
      if (beta != 0) initial[j * 4 + i] = 1;
      double sum = 0;
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 4; ++l)
          sum += double(a[i * 20 + k * 6 + l]) * double(b[l * 15 + k * 5 + j]);
      expected[j * 4 + i] = T(-0.75 * sum + beta);
    }
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    CudaCutensorContraction binding;
    // The host allowance is an explicit qualification reservation; opaque
    // library host allocations are not exposed by the cuTENSOR query API.
    if (!binding.prepare(request, stream, 64ULL << 20, 256ULL << 20, 64ULL << 20))
      throw std::runtime_error(std::string(binding.rejection()));
    if (binding.workspace_bytes() > (64ULL << 20) || !binding.provider_version() ||
        binding.prepare_calls() != 1)
      throw std::runtime_error("invalid prepared provenance");
    T *da{}, *db{}, *dc{};
    int* error{};
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&db), b.size() * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dc), initial.size() * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
    cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    std::vector<T> actual(initial.size());
    for (int replay = 0; replay < 3; ++replay) {
      cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
      cuda_check(cudaMemcpyAsync(dc, initial.data(), initial.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
      binding.execute(stream, da, db, dc, error);
      cuda_check(cudaMemcpyAsync(actual.data(), dc, actual.size() * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream));
      int failed{};
      cuda_check(cudaMemcpyAsync(&failed, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
      cuda_check(cudaStreamSynchronize(stream));
      for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::isnan(expected[i]) ? !std::isnan(actual[i]) : actual[i] != expected[i])
          throw std::runtime_error("cuTENSOR differs from independent affine oracle");
      if (failed || binding.prepare_calls() != 1 || binding.calls() != std::size_t(replay + 1))
        throw std::runtime_error("cuTENSOR replay performed preparation or failed");
    }
    reject([&] { binding.execute(stream, da, db, da, error); });
    reject([&] { binding.execute(nullptr, da, db, dc, error); });
    reject([&] { binding.prepare(request, stream, 0, 0, 0); });
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    reject([&] { binding.execute(stream, da, db, dc, error); });
    cudaGraph_t graph{};
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphDestroy(graph));
    // Error publication remains sticky across a later valid replay.
    cuda_check(cudaMemcpyAsync(da, &nan, sizeof(T), cudaMemcpyHostToDevice, stream));
    binding.execute(stream, da, db, dc, error);
    int failed{};
    cuda_check(cudaMemcpyAsync(&failed, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (!failed) throw std::runtime_error("cuTENSOR nonfinite output was not audited");
    binding.reset();
    reject([&] { binding.execute(stream, da, db, dc, error); });
    if (binding.prepare(request, stream, 0, 0, 0) || binding.rejection().empty())
      throw std::runtime_error("missing host reservation did not retain rejection");
    cuda_check(cudaFree(da));
    cuda_check(cudaFree(db));
    cuda_check(cudaFree(dc));
    cuda_check(cudaFree(error));
  }
  cuda_check(cudaStreamDestroy(stream));
}

int main() {
  try {
    for (double beta : {0.0, 0.25}) {
      check<float>(beta);
      check<double>(beta);
    }
    std::cout << "cuTENSOR affine dtype/replay/resource/lifetime gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
