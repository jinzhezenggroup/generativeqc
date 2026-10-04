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
void check(char ta, char tb, std::size_t batch) {
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
  std::vector<T> a(batch * m * k), b(batch * k * n), actual(batch * m * n), expected(actual.size());
  for (std::size_t i = 0; i != a.size(); ++i) a[i] = T(int(i % 17) - 8) / 16;
  for (std::size_t i = 0; i != b.size(); ++i) b[i] = T(int(i % 13) - 6) / 16;
  // Independent semantic i,j,k loops, explicitly indexing the physical views.
  // Dyadic data makes the expected products exactly representable in FP32 too.
  for (std::size_t q = 0; q != batch; ++q)
    for (std::size_t i = 0; i != m; ++i)
      for (std::size_t j = 0; j != n; ++j) {
        double sum = 0;
        for (std::size_t x = 0; x != k; ++x)
          sum += double(a[q * m * k + (ta == 'N' ? i * k + x : x * m + i)]) *
                 double(b[q * k * n + (tb == 'N' ? x * n + j : j * k + x)]);
        expected[q * m * n + i * n + j] = T(-0.75 * sum);
      }
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CudaContractionContext context;
  if (!context.prepare(stream)) throw std::runtime_error("provider preparation failed");
  PreparedContractions bindings;
  std::size_t calls{}, summands{};
  bindings.add(m, n, batch, {request}, context, calls, summands);
  T *da{}, *db{}, *dc{};
  int* error{};
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&db), b.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&dc), actual.size() * sizeof(T)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
  cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  auto run = [&] { bindings.execute(0, m, n, batch, stream, da, db, dc, error); };
  run();
  cuda_check(cudaMemcpyAsync(actual.data(), dc, actual.size() * sizeof(T), cudaMemcpyDeviceToHost,
                             stream));
  int status{};
  cuda_check(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (status || actual != expected || calls != 1 || summands != batch * m * n * k)
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
    for (auto a : {'N', 'T'})
      for (auto b : {'N', 'T'})
        for (std::size_t batches : {1, 2}) {
          check<float>(a, b, batches);
          check<double>(a, b, batches);
        }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
