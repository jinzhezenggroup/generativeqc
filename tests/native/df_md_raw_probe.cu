// Independent-input qualification seam. Native normalization and device E/R
// kernels consume the basis; no reference integral enters their preparation.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/df_md_raw.hpp"

using namespace generativeqc;
using namespace generativeqc::scf::cuda_execution;

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

core::System read_system() {
  core::System result;
  std::size_t atoms{}, shells{};
  int spherical{};
  std::cin >> atoms >> shells >> result.charge >> spherical;
  if (!atoms || atoms > 8 || !shells || shells > 16)
    throw std::runtime_error("fixture exceeds small-basis qualification bound");
  result.basis_representation =
      spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  for (std::size_t i = 0; i < atoms; ++i) {
    core::Atom atom;
    std::cin >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    result.atoms.push_back(atom);
  }
  for (std::size_t i = 0; i < shells; ++i) {
    core::Shell shell;
    std::size_t primitives{};
    std::cin >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!primitives || primitives > 16 || shell.angular_momentum > 4)
      throw std::runtime_error("fixture exceeds primitive/angular bound");
    shell.primitives.resize(primitives);
    for (auto& primitive : shell.primitives)
      std::cin >> primitive.exponent >> primitive.coefficient;
    result.shells.push_back(shell);
  }
  std::string detail;
  if (!std::cin || molecule::validate_and_normalize(result, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "invalid basis fixture" : detail);
  return result;
}

/** Test-only RAII owner. Production admission still needs budget/lifetime
 * integration; this fixture deliberately bounds all allocations independently. */
struct DeviceOwner {
  cudaStream_t stream{};
  std::vector<void*> allocations;
  DeviceOwner() { check(cudaStreamCreate(&stream)); }
  ~DeviceOwner() {
    (void)cudaStreamSynchronize(stream);
    for (void* pointer : allocations) (void)cudaFree(pointer);
    (void)cudaStreamDestroy(stream);
  }
  template <class T>
  T* allocate(std::size_t count) {
    if (!count || count > (1U << 24) / sizeof(T))
      throw std::runtime_error("test allocation exceeds 16 MiB bound");
    T* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    allocations.push_back(pointer);
    return pointer;
  }
  template <class T>
  T* upload(const std::vector<T>& values) {
    auto* pointer = allocate<T>(values.size());
    check(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice,
                          stream));
    // The input vector is temporary setup storage. Its upload must retire
    // before it is destroyed; replay storage remains owned until stream drain.
    check(cudaStreamSynchronize(stream));
    return pointer;
  }
};

/** Inject real allocation admission failures without exhausting the GPU. */
struct LedgerScope {
  std::shared_ptr<runtime::DeviceResourceLedger> previous{runtime::active_device_resource_ledger};
  std::shared_ptr<runtime::DeviceResourceLedger> ledger{
      std::make_shared<runtime::DeviceResourceLedger>()};
  explicit LedgerScope(std::size_t limit) {
    ledger->device = 0;
    ledger->limit = limit;
    runtime::active_device_resource_ledger = ledger;
  }
  ~LedgerScope() { runtime::active_device_resource_ledger = previous; }
};

/** Graphs must retire before their captured native owner and borrowed stream. */
struct GraphOwner {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ~GraphOwner() {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || !std::getenv("SLURM_JOB_ID")) return 3;
  try {
    const auto orbital = read_system(), auxiliary = read_system();
    DeviceOwner device;
    CudaDfMdRaw* raw{};
    CudaDfMdResources resources;
    std::string detail;
    if (create_cuda_df_md_raw(0, orbital, auxiliary, 0, 16U << 20, &raw, resources, detail) !=
            GENERATIVEQC_STATUS_OUT_OF_MEMORY ||
        raw)
      throw std::runtime_error("zero optional budget did not refuse transactionally");
    if (create_cuda_df_md_raw(0, orbital, auxiliary, 16U << 20, 0, &raw, resources, detail) !=
            GENERATIVEQC_STATUS_OUT_OF_MEMORY ||
        raw)
      throw std::runtime_error("zero host budget did not refuse transactionally");
    auto displaced = auxiliary;
    displaced.atoms[0].position[0] += 0.1;
    if (create_cuda_df_md_raw(0, orbital, displaced, 16U << 20, 16U << 20, &raw, resources,
                              detail) != GENERATIVEQC_STATUS_INVALID_ARGUMENT ||
        raw)
      throw std::runtime_error("mismatched geometry was admitted");
    auto status =
        create_cuda_df_md_raw(0, orbital, auxiliary, 16U << 20, 16U << 20, &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    const auto admitted = resources.device_bytes;
    destroy_cuda_df_md_raw(raw);
    raw = nullptr;
    LedgerScope ledger(admitted - 1);
    status =
        create_cuda_df_md_raw(0, orbital, auxiliary, 16U << 20, 16U << 20, &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_OUT_OF_MEMORY || raw || ledger.ledger->live ||
        !ledger.ledger->allocations || !ledger.ledger->rejected)
      throw std::runtime_error("partial allocation refusal did not roll back");
    ledger.ledger->limit = admitted;
    status =
        create_cuda_df_md_raw(0, orbital, auxiliary, 16U << 20, 16U << 20, &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    std::unique_ptr<CudaDfMdRaw, decltype(&destroy_cuda_df_md_raw)> prepared(
        raw, destroy_cuda_df_md_raw);
    if (resources.device_bytes != ledger.ledger->live ||
        resources.device_bytes < ledger.ledger->peak || resources.host_peak_bytes > (16U << 20))
      throw std::runtime_error("owner storage escaped its declared budget");
    const auto allocations = ledger.ledger->allocations;
    const auto n = resources.nbf, a = resources.naux, pairs = n * (n + 1) / 2;
    auto* output = device.allocate<double>(pairs * a);
    auto* tile = device.allocate<double>(7 * 3);
    auto* tiled = device.allocate<double>(pairs * a);
    if (cuda_df_md_raw_tile(raw, device.stream, pairs, 0, a, 0, output) != cudaSuccess ||
        cuda_df_md_raw_tile(raw, device.stream, pairs, 1, 0, a, output) != cudaErrorInvalidValue ||
        cuda_df_md_raw_tile(raw, device.stream, 1, std::numeric_limits<std::size_t>::max(), 0, a,
                            output) != cudaErrorInvalidValue ||
        cuda_df_md_raw_tile(raw, device.stream, 0, pairs, a, 1, output) != cudaErrorInvalidValue ||
        cuda_df_md_raw_tile(raw, device.stream, 0, pairs, 0, a, nullptr) != cudaErrorInvalidValue)
      throw std::runtime_error("raw tile extent validation failed");
    GraphOwner graph;
    check(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    check(cuda_df_md_raw_tile(raw, device.stream, 0, pairs, 0, a, output));
    for (std::size_t p = 0; p < pairs; p += 7)
      for (std::size_t q = 0; q < a; q += 3) {
        const auto np = std::min(std::size_t{7}, pairs - p);
        const auto na = std::min(std::size_t{3}, a - q);
        check(cuda_df_md_raw_tile(raw, device.stream, p, np, q, na, tile));
        check(cudaMemcpy2DAsync(tiled + p * a + q, a * sizeof(double), tile, na * sizeof(double),
                                na * sizeof(double), np, cudaMemcpyDeviceToDevice, device.stream));
      }
    check(cudaStreamEndCapture(device.stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    std::ofstream destination(argv[1], std::ios::binary);
    std::vector<double> values(pairs * a);
    for (unsigned replay = 0; replay < 2; ++replay) {
      // A dirty output on each replay exposes missing writes and dependence on
      // previous contents. The immutable source must overwrite every element.
      check(cudaMemsetAsync(output, 0xff, values.size() * sizeof(double), device.stream));
      check(cudaMemsetAsync(tiled, 0xff, values.size() * sizeof(double), device.stream));
      check(cudaGraphLaunch(graph.executable, device.stream));
      for (const auto* pointer : {output, tiled}) {
        check(cudaMemcpyAsync(values.data(), pointer, values.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, device.stream));
        check(cudaStreamSynchronize(device.stream));
        destination.write(reinterpret_cast<const char*>(values.data()),
                          values.size() * sizeof(double));
      }
    }
    if (!destination || ledger.ledger->allocations != allocations)
      throw std::runtime_error("raw replay allocated or failed to publish results");
    check(cudaGraphExecDestroy(graph.executable));
    graph.executable = nullptr;
    check(cudaGraphDestroy(graph.graph));
    graph.graph = nullptr;
    prepared.reset();
    if (ledger.ledger->live) throw std::runtime_error("raw Hermite owner leaked tracked storage");
    std::cout << "raw_hermite_device_bytes=" << admitted << " packed_values=" << pairs * a << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
