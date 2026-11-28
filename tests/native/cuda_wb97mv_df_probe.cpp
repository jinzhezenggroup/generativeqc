#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_wb97mv_df_response.hpp"

namespace {
using namespace generativeqc::dft;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
template <class Value>
std::vector<Value> read(std::ifstream& input, std::size_t count) {
  require(count < (1ULL << 24), "oversized fixture");
  std::vector<Value> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value))),
      "short fixture");
  return values;
}
struct Storage {
  cudaStream_t stream{};
  std::vector<void*> allocations;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    cudaStreamSynchronize(stream);
    for (auto* pointer : allocations) cudaFree(pointer);
    cudaStreamDestroy(stream);
  }
  void* allocate(std::size_t bytes) {
    void* pointer{};
    check(cudaMalloc(&pointer, bytes));
    allocations.push_back(pointer);
    return pointer;
  }
  double* upload(const std::vector<double>& values) {
    auto* pointer = static_cast<double*>(allocate(values.size() * sizeof(double)));
    check(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    return pointer;
  }
  void set(double* pointer, double value) {
    check(cudaMemcpyAsync(pointer, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
};
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  unsigned remaining{UINT_MAX};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    if (!owner.remaining) return cudaErrorInvalidResourceHandle;
    if (owner.remaining != UINT_MAX) --owner.remaining;
    return cudaSuccess;
  }
};
void host_admission() {
  std::size_t bytes{};
  const char* detail{};
  CudaWb97mvDfOptions options;
  require(cuda_wb97mv_df_workspace({3, 5, 2}, options, bytes) == cudaSuccess && bytes,
          "ordinary host-only admission failed");
  options.max_workspace_bytes = bytes - 1;
  require(cuda_wb97mv_df_workspace({3, 5, 2}, options, bytes, &detail) != cudaSuccess && !bytes &&
              detail && std::string(detail).find("device bound") != std::string::npos,
          "byte budget ignored");
  options = {};
  options.max_affine_term_visits = 1;
  require(cuda_wb97mv_df_workspace({3, 5, 2}, options, bytes, &detail) != cudaSuccess && !bytes &&
              detail && std::string(detail).find("term-visit bound") != std::string::npos,
          "work budget ignored");
  options = {};
  options.exchange_fraction = 0.54;
  // Spherical HF/cc-pVTZ and O(f)-augmented cc-pVDZ H2O+ with full JKFIT.
  for (const auto shape : {CudaWb97mvDfShape{44, 95, 1}, {31, 113, 2}})
    require(cuda_wb97mv_df_workspace(shape, options, bytes) == cudaSuccess && bytes,
            "full-rank orbital-f qualification shape exceeded the default bounds");
  // The public fitted owner charges its retained raw panel before the maps.
  options.max_workspace_bytes = (64ULL << 20) - 48 * 48 * 226 * sizeof(double);
  for (const bool spectral : {false, true}) {
    options.spectral_metric = spectral;
    for (const bool coulomb : {true, false}) {
      options.coulomb = coulomb;
      for (const auto shape : {CudaWb97mvDfShape{48, 226, 1}, {48, 226, 2}})
        require(cuda_wb97mv_df_workspace(shape, options, bytes) == cudaSuccess && bytes,
                "full JKFIT water dimer exceeded the public fitted workspace budget");
    }
  }
  options.max_matrix_terms = 1;
  require(cuda_wb97mv_df_workspace({48, 226, 1}, options, bytes, &detail) != cudaSuccess &&
              !bytes && detail && std::string(detail).find("product bound") != std::string::npos,
          "tiled fitted response ignored its matrix work bound");
  options = {};
  options.exchange_fraction = 0.54;
  require(cuda_wb97mv_df_workspace({100, 226, 2}, options, bytes, &detail) != cudaSuccess && !bytes,
          "oversize fitted response ignored its default bounds");
  options.exchange_fraction = 0;
  // A cc-pVDZ water dimer with full JKFIT admits both separated metric schedules.
  for (const bool spectral : {false, true}) {
    options.spectral_metric = spectral;
    for (const auto shape : {CudaWb97mvDfShape{48, 226, 1}, {48, 226, 2}})
      require(cuda_wb97mv_df_workspace(shape, options, bytes) == cudaSuccess && bytes,
              "water-dimer Coulomb response exceeded the default bounds");
  }
  require(cuda_wb97mv_df_workspace({48, 280, 2}, options, bytes, &detail) != cudaSuccess &&
              !bytes && detail && std::string(detail).find("term-visit bound") != std::string::npos,
          "oversize spectral Coulomb response ignored its work bound");
  for (const auto shape : {CudaWb97mvDfShape{0, 3, 1},
                           {2, 0, 1},
                           {2, 3, 3},
                           {SIZE_MAX, 3, 1},
                           {2, SIZE_MAX, 2},
                           {SIZE_MAX / 8, 3, 1}})
    require(cuda_wb97mv_df_workspace(shape, {}, bytes) != cudaSuccess && !bytes,
            "invalid or overflowing shape admitted");
  options = {};
  options.symmetry_relative_tolerance = std::numeric_limits<double>::quiet_NaN();
  require(cuda_wb97mv_df_workspace({2, 3, 1}, options, bytes) != cudaSuccess,
          "nonfinite tolerance admitted");
  CudaWb97mvDfResponse empty;
  require(empty.apply(nullptr, nullptr, nullptr, {}, nullptr) == cudaErrorInvalidResourceHandle,
          "unprepared action admitted");
}
void run(const char* source, const char* result) {
  std::ifstream input(source, std::ios::binary);
  const auto header = read<std::uint64_t>(input, 6);
  const CudaWb97mvDfShape shape{header[0], header[1], header[2]};
  require(shape.nao >= 2 && shape.nao <= 32 && shape.naux >= 2 && shape.naux <= 64 &&
              shape.spins >= 1 && shape.spins <= 2,
          "invalid shape");
  const auto panels = shape.spins * shape.nao * shape.nao;
  const auto raw_count = shape.nao * shape.nao * shape.naux;
  const auto metric_count = shape.naux * shape.naux;
  Storage storage;
  auto* density = storage.upload(read<double>(input, panels));
  auto* raw = storage.upload(read<double>(input, raw_count));
  auto* whitening = storage.upload(read<double>(input, metric_count));
  auto* eigenvalues = storage.upload(read<double>(input, shape.naux));
  auto* delta_density = storage.upload(read<double>(input, panels));
  auto* delta_raw = storage.upload(read<double>(input, raw_count));
  auto* delta_metric = storage.upload(read<double>(input, metric_count));
  const auto threshold = read<double>(input, 1)[0];
  CudaWb97mvDfOptions options;
  options.exchange_fraction = read<double>(input, 1)[0];
  options.coulomb = header[4] == 0;
  options.spectral_metric = header[5] != 0;
  options.split_coulomb_action =
      options.coulomb && !options.spectral_metric && !options.exchange_fraction;
  auto* eigenvectors =
      options.spectral_metric ? storage.upload(read<double>(input, metric_count)) : nullptr;
  check(cudaStreamSynchronize(storage.stream));
  int device{-1};
  check(cudaGetDevice(&device));
  Owner owner;
  const CudaWb97mvDfReference reference{density,   raw,         whitening,      eigenvalues,
                                        threshold, device,      storage.stream, Owner::authenticate,
                                        &owner,    eigenvectors};
  std::size_t bytes{};
  check(cuda_wb97mv_df_workspace(shape, options, bytes));
  auto* arena = storage.allocate(bytes);
  CudaWb97mvDfResponse response;
  if (options.spectral_metric)
    require(response.prepare(shape, options, reference, arena, bytes - 1) == cudaErrorInvalidValue,
            "spectral response exceeded its arena");
  const char* detail{};
  const auto prepared = response.prepare(shape, options, reference, arena, bytes, &detail);
  if (header[3]) {
    require(prepared == cudaErrorInvalidValue && detail,
            "malformed reference admitted or undiagnosed");
    std::cout << "{\"prepare_rejected\":true,\"detail\":\"" << detail << "\"}\n";
    return;
  }
  if (prepared != cudaSuccess && detail) throw std::runtime_error(detail);
  check(prepared);
  const std::array<std::size_t, 7> counts{1,         panels,      panels, raw_count, metric_count,
                                          raw_count, metric_count};
  std::array<double*, 7> pointers{};
  for (unsigned index = 0; index < 7; ++index)
    pointers[index] = static_cast<double*>(storage.allocate(counts[index] * sizeof(double)));
  const CudaWb97mvDfOutput output{pointers[0], pointers[1], pointers[2], pointers[3],
                                  pointers[4], pointers[5], pointers[6]};
  const auto apply = [&](const CudaWb97mvDfOutput& destination) {
    return response.apply(delta_density, delta_raw, delta_metric, destination, storage.stream,
                          &detail);
  };
  const auto first_action = apply(output);
  if (first_action != cudaSuccess && detail) throw std::runtime_error(detail);
  check(first_action);
  std::vector<double> expected;
  for (unsigned index = 0; index < 7; ++index) {
    const auto values = download(pointers[index], counts[index]);
    expected.insert(expected.end(), values.begin(), values.end());
  }
  check(apply(output));
  const auto preserved = [&] {
    std::size_t offset{};
    for (unsigned index = 0; index < 7; ++index) {
      const auto values = download(pointers[index], counts[index]);
      require(std::equal(values.begin(), values.end(), expected.begin() + offset),
              "replay or rejected action changed output");
      offset += counts[index];
    }
  };
  preserved();
  if (options.split_coulomb_action) {
    constexpr double sentinel = -970.0;
    auto* fock = static_cast<double*>(storage.allocate(panels * sizeof(double)));
    storage.set(pointers[5], sentinel);
    auto fock_output = output;
    fock_output.energy = nullptr;
    fock_output.fock_direction = fock;
    check(response.apply(delta_density, delta_raw, delta_metric, fock_output, storage.stream,
                         &detail, CudaWb97mvDfAction::fock_direction));
    require(download(pointers[5], 1)[0] == sentinel, "unrequested weight was published");
    const auto frozen = download(fock, panels);
    const auto compare = [](double actual, double expected) {
      return std::abs(actual - expected) <= 5e-12 + 3e-12 * std::abs(expected);
    };
    require(std::equal(frozen.begin(), frozen.end(), expected.begin() + 1 + panels, compare),
            "partitioned frozen Fock differs from all-output action");
    auto weight_output = output;
    weight_output.fock_direction = nullptr;
    check(response.apply(delta_density, delta_raw, delta_metric, weight_output, storage.stream,
                         &detail, CudaWb97mvDfAction::source_weights));
    std::size_t offset = 1 + 2 * panels;
    for (unsigned index = 3; index < pointers.size(); ++index) {
      const auto weights = download(pointers[index], counts[index]);
      require(std::equal(weights.begin(), weights.end(), expected.data() + offset, compare),
              "partitioned raw weights differ from all-output action");
      offset += counts[index];
    }
    check(apply(output));
  }
  const auto completed_before_rejections = response.work().completed_actions;
  auto alias = output;
  alias.energy = pointers[1];
  require(apply(alias) != cudaSuccess, "overlapping outputs admitted");
  alias = output;
  alias.fock = delta_density;
  require(apply(alias) != cudaSuccess, "input/output alias admitted");
  alias = output;
  alias.energy = static_cast<double*>(arena);
  require(apply(alias) != cudaSuccess, "arena output alias admitted");
  if (options.spectral_metric) {
    alias = output;
    alias.weight_metric = eigenvectors;
    require(apply(alias) == cudaErrorInvalidValue, "borrowed spectral frame was writable");
  }
  require(response.apply(nullptr, delta_raw, delta_metric, output, storage.stream) != cudaSuccess,
          "null direction admitted");
  require(response.apply(static_cast<double*>(arena), delta_raw, delta_metric, output,
                         storage.stream) != cudaSuccess,
          "arena input alias admitted");
  const std::array<double*, 3> directions{delta_density, delta_raw, delta_metric};
  const std::array<std::size_t, 3> asymmetric_offsets{1, shape.naux, 1};
  for (unsigned index = 0; index < directions.size(); ++index) {
    auto* pointer = directions[index];
    const auto offset = asymmetric_offsets[index];
    const auto saved = download(pointer, offset + 1);
    storage.set(pointer, std::numeric_limits<double>::infinity());
    require(apply(output) != cudaSuccess, "nonfinite direction admitted");
    storage.set(pointer, saved[0]);
    storage.set(pointer + offset, saved[offset] + 1);
    require(apply(output) != cudaSuccess, "nonsymmetric direction admitted");
    storage.set(pointer + offset, saved[offset]);
  }
  cudaStream_t other{};
  check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  require(response.apply(delta_density, delta_raw, delta_metric, output, other) != cudaSuccess,
          "foreign stream admitted");
  check(cudaStreamDestroy(other));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeThreadLocal));
  require(apply(output) == cudaErrorStreamCaptureUnsupported, "capture admitted");
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(storage.stream, &graph));
  check(cudaGraphDestroy(graph));
  owner.remaining = 1;
  require(apply(output) == cudaErrorInvalidResourceHandle, "late revocation admitted");
  require(apply(output) == cudaErrorInvalidResourceHandle, "stale owner admitted");
  preserved();
  const auto work = response.work();
  require(work.completed_actions == completed_before_rejections,
          "failed action reported completed");
  std::ofstream file(result, std::ios::binary);
  file.write(reinterpret_cast<const char*>(expected.data()), expected.size() * sizeof(double));
  std::cout << "{\"lifecycle\":true,\"workspace_bytes\":" << work.workspace_bytes
            << ",\"reference_terms\":" << work.reference_terms
            << ",\"action_terms\":" << work.action_terms
            << ",\"matrix_reference_terms\":" << work.matrix_reference_terms
            << ",\"matrix_action_terms\":" << work.matrix_action_terms << "}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    host_admission();
    if (argc == 2 && std::string(argv[1]) == "--host-only") return 0;
    require(argc == 3, "usage: probe fixture.bin result.bin | --host-only");
    run(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
