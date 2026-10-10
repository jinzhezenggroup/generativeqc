// Independent small-matrix oracle for the shared typed native provider.
#include <iostream>
#include <vector>

#include "tensor/cuda_contraction.cuh"

using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;

template <class F>
void rejected(F&& call) {
  try {
    call();
  } catch (const std::logic_error&) {
    return;
  }
  throw std::runtime_error("invalid binding was executed");
}

template <class T>
void check(char ta, char tb, std::size_t batch, bool padded = false,
           ContractionAlgorithm algorithm = ContractionAlgorithm::PedanticBlas,
           std::size_t workspace_bytes = 0) {
  constexpr std::size_t m = 3, n = 5, k = 7;
  constexpr auto dtype = std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32;
  constexpr std::string_view identity =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const auto operand = [&](bool first) {
    if (first)
      return ta == 'N' ? ContractionOperand::dense({3, 0, 2}, {batch, m, k}, dtype)
                       : ContractionOperand::dense({3, 2, 0}, {batch, k, m}, dtype);
    return tb == 'N' ? ContractionOperand::dense({3, 2, 1}, {batch, k, n}, dtype)
                     : ContractionOperand::dense({3, 1, 2}, {batch, n, k}, dtype);
  };
  ContractionRequest request{
      identity,
      identity,
      identity,
      {operand(true), operand(false), ContractionOperand::dense({3, 0, 1}, {batch, m, n}, dtype)},
      {dtype, dtype, dtype},
      dtype,
      ta,
      tb,
      batch,
      m,
      n,
      k,
      -0.75};
  const auto lda = (ta == 'N' ? k : m) + (padded ? 3 : 0),
             ldb = (tb == 'N' ? n : k) + (padded ? 5 : 0), ldc = n + (padded ? 2 : 0);
  if (padded) {
    if (batch != 1) throw std::logic_error("strided batch outside this qualification");
    for (std::size_t i = 0; i < 3; ++i) {
      const auto ld = i == 0 ? lda : i == 1 ? ldb : ldc;
      request.leading_dimensions[i] = ld;
      request.operands[i].strides[1] = ld;
      request.operands[i].strides[0] = request.operands[i].shape[1] * ld;
    }
    request.beta = 0.25;
  }
  const auto nan = std::numeric_limits<T>::quiet_NaN();
  std::vector<T> a(request.operands[0].storage_elements(), nan),
      b(request.operands[1].storage_elements(), nan),
      actual(request.operands[2].storage_elements(), nan), expected(actual.size(), nan);
  for (std::size_t q = 0; q < batch; ++q) {
    for (std::size_t row = 0; row < (ta == 'N' ? m : k); ++row)
      for (std::size_t col = 0; col < (ta == 'N' ? k : m); ++col)
        a[q * m * k + row * lda + col] = T(int((row * 11 + col) % 17) - 8) / 16;
    for (std::size_t row = 0; row < (tb == 'N' ? k : n); ++row)
      for (std::size_t col = 0; col < (tb == 'N' ? n : k); ++col)
        b[q * k * n + row * ldb + col] = T(int((row * 7 + col) % 13) - 6) / 16;
    for (std::size_t row = 0; row < m; ++row)
      for (std::size_t col = 0; col < n; ++col) actual[q * m * n + row * ldc + col] = 1;
  }
  // Independent semantic i,j,k loops, explicitly indexing the physical views.
  // Dyadic data makes the expected products exactly representable in FP32 too.
  for (std::size_t q = 0; q != batch; ++q)
    for (std::size_t i = 0; i != m; ++i)
      for (std::size_t j = 0; j != n; ++j) {
        double sum = 0;
        for (std::size_t x = 0; x != k; ++x)
          sum += double(a[q * m * k + (ta == 'N' ? i * lda + x : x * lda + i)]) *
                 double(b[q * k * n + (tb == 'N' ? x * ldb + j : j * ldb + x)]);
        expected[q * m * n + i * ldc + j] = T(-0.75 * sum + request.beta);
      }
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CudaContractionContext context;
  if (algorithm == ContractionAlgorithm::GeneratedOrdered)
    context.prepare_generated(stream);
  else if (!context.prepare(stream, workspace_bytes))
    throw std::runtime_error("provider preparation failed");
  PreparedContractions bindings;
#if !GENERATIVEQC_HAS_CUTLASS
  bool cutlass_unavailable{};
  try {
    std::size_t unused_calls{}, unused_summands{};
    bindings.add(m, n, batch, {request}, context, unused_calls, unused_summands,
                 {ContractionAlgorithm::CutlassAot});
  } catch (const ContractionPreparationUnavailable&) {
    cutlass_unavailable = true;
  }
  if (!cutlass_unavailable || bindings || bindings.optional_resources().cache_bytes)
    throw std::runtime_error("disabled CUTLASS did not preserve the complete fallback");
#endif
  std::size_t calls{}, summands{};
#if !GENERATIVEQC_HAS_CUTENSOR
  try {
    bindings.add(m, n, batch, {request}, context, calls, summands,
                 {ContractionAlgorithm::CutensorAffine});
    throw std::logic_error("provider-absent build accepted cuTENSOR");
  } catch (const ContractionPreparationUnavailable&) {
  }
  if (bindings) throw std::logic_error("unavailable provider changed table state");
#endif
  bindings.add(m, n, batch, {request}, context, calls, summands, {algorithm});
  T *da{}, *db{}, *dc{};
  int* error{};
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&db), b.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&dc), actual.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
  cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(dc, actual.data(), actual.size() * sizeof(T), cudaMemcpyHostToDevice,
                             stream));
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  auto run = [&] { bindings.execute(0, m, n, batch, stream, da, db, dc, error); };
  run();
  cuda_check(cudaMemcpyAsync(actual.data(), dc, actual.size() * sizeof(T), cudaMemcpyDeviceToHost,
                             stream));
  int status{};
  cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  bool equal = true;
  for (std::size_t i = 0; i < actual.size(); ++i)
    equal = equal && (std::isnan(expected[i]) ? std::isnan(actual[i]) : actual[i] == expected[i]);
  if (status || !equal || calls != 1 || summands != batch * m * n * k)
    throw std::runtime_error("typed provider disagrees with independent matrix oracle");
  rejected([&] { bindings.execute(0, m, n, batch + 1, stream, da, db, dc, error); });
  rejected([&] { bindings.execute(0, m, n, batch, nullptr, da, db, dc, error); });
  rejected([&] { bindings.execute(0, m, n, batch, stream, da, db, da, error); });
  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  rejected(run);
  rejected([&] {
    CudaContractionContext other;
    (void)other.prepare(stream);
  });
  cudaGraph_t graph{};
  cuda_check(cudaStreamEndCapture(stream, &graph));
  cuda_check(cudaGraphDestroy(graph));
  a[0] = std::numeric_limits<T>::infinity();
  cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  run();
  cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (!status) throw std::runtime_error("nonfinite provider output escaped the sticky audit");
  status = 7;
  cuda_check(cudaMemcpyAsync(error, &status, sizeof(int), cudaMemcpyHostToDevice, stream));
  run();
  cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (status != 7)
    throw std::runtime_error("provider audit overwrote the first arithmetic failure");
  if (workspace_bytes) {
    if (!context.release_workspace()) throw std::runtime_error("owned workspace was lost");
    rejected(run);
  }
  context.reset();
  rejected(run);
  cuda_check(cudaFree(error));
  cuda_check(cudaFree(dc));
  cuda_check(cudaFree(db));
  cuda_check(cudaFree(da));
  cuda_check(cudaStreamDestroy(stream));
}

int main() {
  try {
    check<double>('N', 'T', 1, false, ContractionAlgorithm::PedanticBlas,
                  CudaContractionContext::kOptionalWorkspaceBytes);
    check<double>('N', 'N', 2, false, ContractionAlgorithm::PedanticBlas,
                  CudaContractionContext::kOptionalWorkspaceBytes);
    for (auto a : {'N', 'T'})
      for (auto b : {'N', 'T'})
        for (std::size_t batches : {1, 2}) {
          for (auto algorithm :
               {ContractionAlgorithm::PedanticBlas, ContractionAlgorithm::GeneratedOrdered}) {
            check<float>(a, b, batches, false, algorithm);
            check<double>(a, b, batches, false, algorithm);
            if (batches == 1) {
              check<float>(a, b, batches, true, algorithm);
              check<double>(a, b, batches, true, algorithm);
            }
          }
        }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
