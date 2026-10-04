#include <iostream>
#include <vector>

#include "tensor/cuda_cublaslt.cuh"

using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;

template <class F>
void rejects(F call) {
  try {
    call();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("invalid cuBLASLt binding was accepted");
}

// Independent mode addressing: do not reuse the provider's matrix recognition.
std::size_t offset(const ContractionOperand& view, std::size_t batch, std::size_t m, std::size_t n,
                   std::size_t k) {
  const std::array<std::size_t, 4> coordinates{m, n, k, batch};
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < view.rank; ++axis)
    result += coordinates[view.modes[axis]] * view.strides[axis];
  return result;
}

template <class T>
void check(unsigned transposes, std::size_t batches, std::size_t m, std::size_t n, std::size_t k,
           double beta) {
  constexpr auto dtype = std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32;
  constexpr std::string_view identity =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const std::array<std::size_t, 4> extents{m, n, k, batches};
  const auto view = [&](int first, int second, bool transpose) {
    if (transpose) std::swap(first, second);
    auto result =
        batches > 1
            ? ContractionOperand::dense({3, first, second},
                                        {batches, extents[first], extents[second]}, dtype)
            : ContractionOperand::dense({first, second}, {extents[first], extents[second]}, dtype);
    // Nonfinite row/batch padding detects accidental packing or logical audits
    // that read extra physical elements. Every axis order gets this boundary.
    const auto ld = extents[second] + 2;
    result.strides[result.rank - 2] = ld;
    if (batches > 1) result.strides[0] = extents[first] * ld + 3;
    return result;
  };
  ContractionRequest request{
      identity,
      identity,
      identity,
      {view(0, 2, transposes & 1), view(2, 1, transposes & 2), view(0, 1, transposes & 4)},
      {dtype, dtype, dtype},
      dtype};
  request.coefficient = -0.75;
  request.beta = beta;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  std::vector<T> a(request.operands[0].storage_elements(), nan),
      b(request.operands[1].storage_elements(), nan),
      initial(request.operands[2].storage_elements(), nan), expected(initial), actual(initial);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t p = 0; p < k; ++p)
        a[offset(request.operands[0], batch, i, 0, p)] = T(int(3 * batch + i * k + p) - 7) / 16;
    for (std::size_t p = 0; p < k; ++p)
      for (std::size_t j = 0; j < n; ++j)
        b[offset(request.operands[1], batch, 0, j, p)] = T(int(5 * batch + p * n + j) - 11) / 32;
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0;
        for (std::size_t p = 0; p < k; ++p)
          sum += double(a[offset(request.operands[0], batch, i, 0, p)]) *
                 double(b[offset(request.operands[1], batch, 0, j, p)]);
        const auto out = offset(request.operands[2], batch, i, j, 0);
        if (beta) initial[out] = 1;
        expected[out] = T(-0.75 * sum + beta);
      }
  }
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    CudaCublasLtContraction binding;
    rejects([&] { (void)binding.provenance(); });
    if (binding.prepare(request, stream, 0, 0, 0) || binding.heuristic_calls())
      throw std::runtime_error("unreserved binding performed preparation");
    // Explicit test ceilings, not a production resource qualification. Cache
    // capacity and first-execution growth still need broader endpoint evidence.
    if (!binding.prepare(request, stream, 64ULL << 20, 256ULL << 20, 64ULL << 20))
      throw std::runtime_error(std::string(binding.rejection()));
    const auto provenance = binding.provenance();
    int device{}, major{}, minor{};
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (provenance.provider_version != cublasLtGetVersion() ||
        provenance.architecture != 10 * major + minor || !provenance.runtime_version ||
        provenance.workspace_bytes > (64ULL << 20) || binding.heuristic_calls() != 1 ||
        provenance.request.operands[2].strides != request.operands[2].strides)
      throw std::runtime_error("cuBLASLt prepared provenance changed");
    T *ra{}, *rb{}, *rc{};
    int* error{};
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ra), (a.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&rb), (b.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&rc), (initial.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
    // Actual pointers deliberately provide only the advertised scalar alignment.
    auto *da = ra + 1, *db = rb + 1, *dc = rc + 1;
    cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
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
          throw std::runtime_error("cuBLASLt differs from independent affine oracle");
      if (failed || binding.heuristic_calls() != 1 || binding.calls() != std::size_t(replay + 1) ||
          binding.provenance().algorithm != provenance.algorithm ||
          binding.provenance().workspace_bytes != provenance.workspace_bytes)
        throw std::runtime_error("cuBLASLt replay changed its prepared algorithm");
    }
    rejects([&] { binding.execute(stream, da, db, da, error); });
    rejects([&] { binding.execute(nullptr, da, db, dc, error); });
    rejects([&] { binding.prepare(request, stream, 0, 0, 0); });
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    rejects([&] { binding.execute(stream, da, db, dc, error); });
    cudaGraph_t graph{};
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphDestroy(graph));
    // A later finite replay cannot clear an earlier arithmetic failure.
    cuda_check(cudaMemcpyAsync(da, &nan, sizeof(T), cudaMemcpyHostToDevice, stream));
    binding.execute(stream, da, db, dc, error);
    cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(dc, initial.data(), initial.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    binding.execute(stream, da, db, dc, error);
    int failed{};
    cuda_check(cudaMemcpyAsync(&failed, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (!failed) throw std::runtime_error("cuBLASLt lost a sticky finite error");
    binding.release();
    if (binding.workspace_bytes() || binding.provider_bytes() || binding.host_bytes())
      throw std::runtime_error("cuBLASLt release retained resources");
    rejects([&] { binding.execute(stream, da, db, dc, error); });
    rejects([&] { (void)binding.provenance(); });
    cuda_check(cudaFree(ra));
    cuda_check(cudaFree(rb));
    cuda_check(cudaFree(rc));
    cuda_check(cudaFree(error));
  }
  cuda_check(cudaStreamDestroy(stream));
}

int main() {
  try {
    for (unsigned transpose = 0; transpose < 8; ++transpose)
      for (auto batches : {1U, 2U})
        for (const auto dims : {std::array<std::size_t, 3>{3, 5, 7}, {1, 3, 1}, {1, 1, 1}})
          for (auto beta : {0.0, 0.5}) {
            check<double>(transpose, batches, dims[0], dims[1], dims[2], beta);
            check<float>(transpose, batches, dims[0], dims[1], dims[2], beta);
          }
    std::cout << "192 cuBLASLt layout/precision/beta cases passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
