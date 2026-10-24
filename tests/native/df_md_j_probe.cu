// Independent-input qualification seam. Native normalization and device E/R
// kernels consume the basis; no reference integral enters their preparation.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/df_md_owner.hpp"

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
    // Long s-shell contractions exercise the automatic source-work policy.
    // The independent 16 MiB owner budgets below still bound actual storage;
    // accepting a larger input count does not enlarge those resource limits.
    if (!primitives || primitives > 128 || shell.angular_momentum > 4)
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
  const bool derivatives = argc == 3 && std::string(argv[2]) == "--gradient";
  if ((argc != 2 && !derivatives) || !std::getenv("SLURM_JOB_ID")) return 3;
  try {
    const auto orbital = read_system(), auxiliary = read_system();
    DeviceOwner owner;
    CudaDfMdOwner* raw = nullptr;
    CudaDfMdResources resources;
    std::string detail;
    if (create_cuda_df_md_owner(0, owner.stream, orbital, auxiliary, 0, 16U << 20, &raw, resources,
                                detail) != GENERATIVEQC_STATUS_OUT_OF_MEMORY ||
        raw)
      throw std::runtime_error("zero optional budget did not refuse transactionally");
    auto displaced = auxiliary;
    displaced.atoms[0].position[0] += 0.1;
    if (create_cuda_df_md_owner(0, owner.stream, orbital, displaced, 16U << 20, 16U << 20, &raw,
                                resources, detail) != GENERATIVEQC_STATUS_INVALID_ARGUMENT ||
        raw)
      throw std::runtime_error("mismatched source geometry was admitted");
    auto status = create_cuda_df_md_owner(0, owner.stream, orbital, auxiliary, 16U << 20, 16U << 20,
                                          &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    const auto admitted_bytes = resources.device_bytes;
    destroy_cuda_df_md_owner(raw);
    raw = nullptr;
    LedgerScope ledger(admitted_bytes - 1);
    status = create_cuda_df_md_owner(0, owner.stream, orbital, auxiliary, 16U << 20, 16U << 20,
                                     &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_OUT_OF_MEMORY || raw || ledger.ledger->live ||
        !ledger.ledger->allocations || !ledger.ledger->rejected) {
      destroy_cuda_df_md_owner(raw);
      throw std::runtime_error("partial optional allocation failure did not roll back");
    }
    ledger.ledger->limit = admitted_bytes;
    status = create_cuda_df_md_owner(0, owner.stream, orbital, auxiliary, 16U << 20, 16U << 20,
                                     &raw, resources, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    std::unique_ptr<CudaDfMdOwner, decltype(&destroy_cuda_df_md_owner)> prepared(
        raw, destroy_cuda_df_md_owner);
    if (ledger.ledger->live != resources.device_bytes ||
        ledger.ledger->peak > resources.device_bytes)
      throw std::runtime_error("reported optional storage differs from the actual ledger");
    const auto allocations_before_replay = ledger.ledger->allocations;
    const auto n = resources.nbf, a = resources.naux;
    if (!resources.device_bytes || resources.device_bytes > (16U << 20) ||
        resources.host_peak_bytes > (16U << 20))
      throw std::runtime_error("optional owner escaped its reported storage budget");
    std::vector<double> density(n * n), potential(a);
    for (auto& value : density) std::cin >> value;
    for (auto& value : potential) std::cin >> value;
    if (!std::cin) throw std::runtime_error("missing public input vectors");
    auto* d = owner.allocate<double>(n * n);
    auto* v = owner.allocate<double>(a);
    auto* j = owner.allocate<double>(n * n);
    auto* rho = owner.allocate<double>(a);
    const auto gradient_elements = 3 * orbital.atoms.size();
    auto* gradient = derivatives ? owner.allocate<double>(gradient_elements) : nullptr;
    if (cuda_df_md_charge(raw, d, d) != cudaErrorInvalidValue ||
        cuda_df_md_project(raw, nullptr, j) != cudaErrorInvalidValue)
      throw std::runtime_error("invalid replay buffers were not refused");
    GraphOwner graph;
    check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal));
    CudaDfMdOwner* forbidden = nullptr;
    CudaDfMdResources unused;
    if (create_cuda_df_md_owner(0, owner.stream, orbital, auxiliary, 16U << 20, 16U << 20,
                                &forbidden, unused,
                                detail) != GENERATIVEQC_STATUS_INVALID_ARGUMENT ||
        forbidden)
      throw std::runtime_error("owner preparation entered a captured stream");
    // Projection deliberately precedes charge: neither call may depend on a
    // previous invocation's Hermite input or stale intermediate potential.
    if (derivatives) check(cuda_df_md_three_center_gradient(raw, d, v, 1.25, gradient));
    check(cuda_df_md_project(raw, v, j));
    check(cuda_df_md_charge(raw, d, rho));
    check(cudaStreamEndCapture(owner.stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    std::ofstream output(argv[1], std::ios::binary);
    for (unsigned replay = 0; replay < 3; ++replay) {
      auto current_d = density, current_v = potential;
      const double scale_d = replay == 0 ? 1 : replay == 1 ? 0 : -0.37;
      const double scale_v = replay == 0 ? 1 : replay == 1 ? 0 : 2.0;
      for (auto& value : current_d) value *= scale_d;
      for (auto& value : current_v) value *= scale_v;
      check(cudaMemcpyAsync(d, current_d.data(), n * n * sizeof(double), cudaMemcpyHostToDevice,
                            owner.stream));
      check(cudaMemcpyAsync(v, current_v.data(), a * sizeof(double), cudaMemcpyHostToDevice,
                            owner.stream));
      // Nonzero existing output verifies addition, including the exact zero
      // replay. Captured derivatives must refresh both input vectors themselves.
      std::vector<double> current_gradient(gradient_elements, 0.25);
      if (derivatives)
        check(cudaMemcpyAsync(gradient, current_gradient.data(), gradient_elements * sizeof(double),
                              cudaMemcpyHostToDevice, owner.stream));
      if (replay == 0) {
        if (derivatives) check(cuda_df_md_three_center_gradient(raw, d, v, 1.25, gradient));
        check(cuda_df_md_project(raw, v, j));
        check(cuda_df_md_charge(raw, d, rho));
      } else {
        check(cudaGraphLaunch(graph.executable, owner.stream));
      }
      check(cudaMemcpyAsync(current_d.data(), j, n * n * sizeof(double), cudaMemcpyDeviceToHost,
                            owner.stream));
      check(cudaMemcpyAsync(current_v.data(), rho, a * sizeof(double), cudaMemcpyDeviceToHost,
                            owner.stream));
      if (derivatives)
        check(cudaMemcpyAsync(current_gradient.data(), gradient, gradient_elements * sizeof(double),
                              cudaMemcpyDeviceToHost, owner.stream));
      check(cudaStreamSynchronize(owner.stream));
      output.write(reinterpret_cast<const char*>(current_v.data()), a * sizeof(double));
      output.write(reinterpret_cast<const char*>(current_d.data()), n * n * sizeof(double));
      if (derivatives)
        output.write(reinterpret_cast<const char*>(current_gradient.data()),
                     gradient_elements * sizeof(double));
      if (!output) throw std::runtime_error("cannot write probe output");
    }
    if (ledger.ledger->allocations != allocations_before_replay)
      throw std::runtime_error("charge/J replay allocated native device storage");
    check(cudaGraphExecDestroy(graph.executable));
    graph.executable = nullptr;
    check(cudaGraphDestroy(graph.graph));
    graph.graph = nullptr;
    prepared.reset();
    if (ledger.ledger->live) throw std::runtime_error("DF Hermite owner leaked tracked storage");
    std::cout << "public_aos=" << n << " auxiliary_aos=" << a
              << " orbital_primitive_pairs=" << resources.orbital_primitives
              << " auxiliary_primitives=" << resources.auxiliary_primitives
              << " owner_device_bytes=" << resources.device_bytes << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
