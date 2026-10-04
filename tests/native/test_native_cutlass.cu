#include <iostream>
#include <vector>

#include "tensor/cuda_cutlass.cuh"
using namespace generativeqc::tensor;
using generativeqc_tensor::cuda_check;
static std::string_view artifact;
template <class F>
void rejects(F call) {
  try {
    call();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("invalid CUTLASS binding accepted");
}
// Independent mode addressing: do not reuse the provider's matrix recognition.
std::size_t offset(const ContractionOperand& view, std::size_t batch, std::size_t m, std::size_t n,
                   std::size_t k) {
  const std::array<std::size_t, 4> coordinates{m, n, k, batch};
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < view.rank; ++axis) {
    const auto mode = view.modes[axis];
    auto coordinate = coordinates[mode % 4];
    if (mode >= 4) {
      coordinate %= view.shape[axis];
    } else {
      for (std::size_t second = 0; second < view.rank; ++second)
        if (view.modes[second] == mode + 4) coordinate /= view.shape[second];
    }
    result += coordinate * view.strides[axis];
  }
  return result;
}

template <class T>
void check(unsigned transposes, std::size_t batches, std::size_t m, std::size_t n, std::size_t k,
           double beta, bool grouped = false) {
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
    if (grouped) {
      // Split every physical dimension while retaining its exact addresses.
      // The oracle decodes those semantic axes independently of the recipe.
      const auto original = result;
      result.rank *= 2;
      for (std::size_t axis = 0; axis < original.rank; ++axis) {
        const auto outer = original.shape[axis] % 2 == 0 ? 2 : 1;
        const auto inner = original.shape[axis] / outer;
        result.modes[2 * axis] = original.modes[axis];
        result.modes[2 * axis + 1] = original.modes[axis] + 4;
        result.shape[2 * axis] = outer;
        result.shape[2 * axis + 1] = inner;
        result.strides[2 * axis] = inner * original.strides[axis];
        result.strides[2 * axis + 1] = original.strides[axis];
      }
    }
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
  if (grouped && k % 2 == 0) {
    auto invalid = request;
    auto& right = invalid.operands[1];
    for (std::size_t axis = 0; axis < right.rank; ++axis)
      if (right.modes[axis] == 2) ++right.strides[axis];
    rejects([&] { (void)MatrixContractionRecipe::from(invalid); });
  }
  const T nan = std::numeric_limits<T>::quiet_NaN();
  std::vector<T> a(request.operands[0].storage_elements(), nan),
      b(request.operands[1].storage_elements(), nan),
      initial(request.operands[2].storage_elements(), nan), expected(initial), actual(initial);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t p = 0; p < k; ++p)
        a[offset(request.operands[0], batch, i, 0, p)] =
            T(int((3 * batch + i * k + p) % 17) - 8) / 16;
    for (std::size_t p = 0; p < k; ++p)
      for (std::size_t j = 0; j < n; ++j)
        b[offset(request.operands[1], batch, 0, j, p)] =
            T(int((5 * batch + p * n + j) % 13) - 6) / 32;
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
    CudaCutlassContraction binding;
    rejects([&] { (void)binding.provenance(); });
    rejects([&] { binding.prepare(request, stream, "unverified", 1 << 20, 256ULL << 20); });
    if (binding.prepare(request, stream, artifact, 0, 256ULL << 20) ||
        binding.prepare(request, stream, artifact, 1 << 20, 0) || binding.preparations() ||
        binding.module_bytes())
      throw std::runtime_error("unreserved CUTLASS preparation");
#if defined(GENERATIVEQC_TEST_HOOKS)
    // Keep separate reservations for the poisoned owner and ordinary replay.
    // Both obligations remain charged until the enclosing CUDA context dies.
    constexpr std::size_t simultaneous_module_ceiling = 512ULL << 20;
    CudaCutlassContraction failed_binding;
    // Failure after loading must roll back descriptors while keeping the
    // context-retained reservation, including after an explicit release.
    cutlass_fail_after_module_load_for_test = true;
    rejects([&] { failed_binding.prepare(request, stream, artifact, 1 << 20, 256ULL << 20); });
    cutlass_fail_after_module_load_for_test = false;
    failed_binding.release();
    if (failed_binding.module_bytes() != (256ULL << 20) ||
        failed_binding.host_bytes() != sizeof(CudaCutlassContraction) ||
        failed_binding.preparations() || failed_binding.calls() || failed_binding.summands())
      throw std::runtime_error("CUTLASS failed preparation lost retained module charge");
    rejects([&] { (void)failed_binding.provenance(); });
    for (const auto reservation : {(256ULL << 20) - 1, 256ULL << 20, 512ULL << 20})
      rejects([&] { failed_binding.prepare(request, stream, artifact, 1 << 20, reservation); });
#endif
    if (!binding.prepare(request, stream, artifact, 1 << 20, 256ULL << 20))
      throw std::runtime_error("CUTLASS preparation unavailable");
#if defined(GENERATIVEQC_TEST_HOOKS)
    if (binding.module_bytes() + failed_binding.module_bytes() != simultaneous_module_ceiling)
      throw std::runtime_error("CUTLASS independent owners lost their combined reservations");
#endif
    const auto host = binding.host_bytes();
    const auto provenance = binding.provenance();
    if (std::string_view(provenance.artifact_identity.data(), 64) != artifact ||
        provenance.version != CUTLASS_VERSION || provenance.tile != std::array<int, 3>{32, 64, 8} ||
        !provenance.architecture)
      throw std::runtime_error("incomplete CUTLASS provenance");
    T *ra{}, *rb{}, *rc{};
    int* error{};
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ra), (a.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&rb), (b.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&rc), (initial.size() + 1) * sizeof(T)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
    auto *da = ra + 1, *db = rb + 1, *dc = rc + 1;
    cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    for (std::size_t replay = 0; replay < 3; ++replay) {
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
          throw std::runtime_error("CUTLASS differs from independent semantic oracle");
      if (failed || binding.calls() != replay + 1 || binding.preparations() != 1 ||
          binding.summands() != (replay + 1) * request.affine_summands())
        throw std::runtime_error("CUTLASS repeated work accounting");
    }
    rejects([&] { binding.execute(stream, da, db, da, error); });
    rejects([&] { binding.execute(nullptr, da, db, dc, error); });
    rejects([&] { binding.prepare(request, stream, artifact, 1 << 20, 256ULL << 20); });
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    rejects([&] { binding.execute(stream, da, db, dc, error); });
    cudaGraph_t graph{};
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphDestroy(graph));
    cuda_check(cudaMemcpyAsync(da, &nan, sizeof(T), cudaMemcpyHostToDevice, stream));
    binding.execute(stream, da, db, dc, error);
    cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(dc, initial.data(), initial.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    binding.execute(stream, da, db, dc, error);
    int failed{};
    cuda_check(cudaMemcpyAsync(&failed, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (!failed) throw std::runtime_error("CUTLASS lost sticky finite error");
    binding.release();
    rejects([&] { binding.execute(stream, da, db, dc, error); });
    // The CUDA context retains loaded code even after the local plan is gone.
    if (binding.module_bytes() != (256ULL << 20) || binding.host_bytes() >= host)
      throw std::runtime_error("CUTLASS lifetime accounting");
    if (binding.prepare(request, stream, artifact, host, 1))
      throw std::runtime_error("CUTLASS forgot retained module reservation");
    if (binding.prepare(request, stream, artifact, host - 1, 256ULL << 20))
      throw std::runtime_error("CUTLASS admitted one byte below exact host floor");
    if (!binding.prepare(request, stream, artifact, host, 256ULL << 20))
      throw std::runtime_error("CUTLASS rejected exact host floor");
    binding.release();
    // nullptr is a valid default stream, including its checked live release.
    if (!binding.prepare(request, nullptr, artifact, host, 256ULL << 20))
      throw std::runtime_error("CUTLASS rejected default stream");
    cuda_check(cudaMemset(error, 0, sizeof(int)));
    cuda_check(cudaMemcpy(dc, initial.data(), initial.size() * sizeof(T), cudaMemcpyHostToDevice));
    binding.execute(nullptr, da, db, dc, error);
    binding.release();
    cuda_check(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (std::isnan(expected[i]) ? !std::isnan(actual[i]) : actual[i] != expected[i])
        throw std::runtime_error("CUTLASS default-stream publication differs");
    cuda_check(cudaFree(error));
    cuda_check(cudaFree(ra));
    cuda_check(cudaFree(rb));
    cuda_check(cudaFree(rc));
  }
  cuda_check(cudaStreamDestroy(stream));
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected build artifact identity");
    artifact = argv[1];
    int count = 0;
    for (unsigned order = 0; order < 8; ++order)
      for (auto batches : {1u, 2u})
        for (double beta : {0.0, 0.5})
          for (bool grouped : {false, true}) {
            check<float>(order, batches, 5, 7, 11, beta, grouped);
            check<double>(order, batches, 5, 7, 11, beta, grouped);
            count += 2;
          }
    for (unsigned order = 0; order < 8; ++order) {
      check<float>(order, 1, 1, 1, 1, 0);
      check<double>(order, 1, 1, 1, 1, 0);
      check<float>(order, 2, 36, 66, 18, 0.5, true);
      check<double>(order, 2, 36, 66, 18, 0.5, true);
      count += 4;
    }
    std::cout << "CUTLASS " << CUTLASS_VERSION << ' ' << CudaCutlassContraction::kFamily << ": "
              << count << " independent layout/precision/beta cases passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
