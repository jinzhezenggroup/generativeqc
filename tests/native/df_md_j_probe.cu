// Independent-input qualification seam. Native normalization and device E/R
// kernels consume the basis; no reference integral enters their preparation.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/df_md_j.cuh"
#include "scf/cuda/df_md_layout.hpp"

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

struct Prepared {
  DfMdView view;
  std::array<std::size_t, 8> offsets{};
};

Prepared prepare(DeviceOwner& owner, const core::System& system, bool auxiliary) {
  const auto layout = make_df_md_layout(system, auxiliary, 16U << 20, 16U << 20);
  const auto& primitives = layout.primitives;
  Prepared result;
  result.offsets = layout.offsets;
  result.view.primitives = owner.upload(primitives);
  result.view.first_aos = result.view.second_aos = owner.upload(layout.aos);
  if (auxiliary) {
    DfMdAo unit;
    unit.count = 1;
    unit.coefficients[0] = 1;
    result.view.first_aos = owner.upload(std::vector<DfMdAo>{unit});
  }
  result.view.count = primitives.size();
  result.view.nbf = layout.aos.size();
  result.view.auxiliary = auxiliary;
  result.view.transforms = owner.allocate<double>(layout.transform_elements);
  result.view.density = owner.allocate<double>(layout.hermite_elements);
  result.view.potential = owner.allocate<double>(layout.hermite_elements);
  df_md_prepare_transforms<<<primitives.size(), 128, 0, owner.stream>>>(result.view);
  check(cudaPeekAtLastError());
  return result;
}

template <unsigned P, unsigned A>
void contract_class(const Prepared& pair, const Prepared& auxiliary, cudaStream_t stream) {
  const auto p = pair.offsets[P], np = pair.offsets[P + 1] - p;
  const auto a = auxiliary.offsets[A], na = auxiliary.offsets[A + 1] - a;
  if (!np || !na) return;
  df_md_contract<P, A, true><<<na, 128, 0, stream>>>(pair.view, p, np, auxiliary.view, a, na);
  check(cudaPeekAtLastError());
  df_md_contract<P, A, false><<<np, 128, 0, stream>>>(pair.view, p, np, auxiliary.view, a, na);
  check(cudaPeekAtLastError());
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || !std::getenv("SLURM_JOB_ID")) return 3;
  try {
    const auto orbital = read_system(), auxiliary = read_system();
    DeviceOwner owner;
    const auto pairs = prepare(owner, orbital, false), aux = prepare(owner, auxiliary, true);
    const auto n = pairs.view.nbf, a = aux.view.nbf;
    std::vector<double> density(n * n), potential(a);
    for (auto& value : density) std::cin >> value;
    for (auto& value : potential) std::cin >> value;
    if (!std::cin) throw std::runtime_error("missing public input vectors");
    auto* d = owner.allocate<double>(n * n);
    auto* v = owner.allocate<double>(a);
    auto* j = owner.allocate<double>(n * n);
    auto* rho = owner.allocate<double>(a);
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
      df_md_precontract<<<pairs.view.count, 128, 0, owner.stream>>>(pairs.view, d);
      check(cudaPeekAtLastError());
      df_md_precontract<<<aux.view.count, 128, 0, owner.stream>>>(aux.view, v);
      check(cudaPeekAtLastError());
#define CLASS(P, A) contract_class<P, A>(pairs, aux, owner.stream)
#define ROW(P) \
  CLASS(P, 0); \
  CLASS(P, 1); \
  CLASS(P, 2); \
  CLASS(P, 3); \
  CLASS(P, 4)
      ROW(0);
      ROW(1);
      ROW(2);
      ROW(3);
      ROW(4);
      ROW(5);
      ROW(6);
#undef ROW
#undef CLASS
      check(cudaMemsetAsync(j, 0, n * n * sizeof(double), owner.stream));
      check(cudaMemsetAsync(rho, 0, a * sizeof(double), owner.stream));
      df_md_project<<<pairs.view.count, 128, 0, owner.stream>>>(pairs.view, j);
      check(cudaPeekAtLastError());
      df_md_project<<<aux.view.count, 128, 0, owner.stream>>>(aux.view, rho);
      check(cudaPeekAtLastError());
      check(cudaMemcpyAsync(current_d.data(), j, n * n * sizeof(double), cudaMemcpyDeviceToHost,
                            owner.stream));
      check(cudaMemcpyAsync(current_v.data(), rho, a * sizeof(double), cudaMemcpyDeviceToHost,
                            owner.stream));
      check(cudaStreamSynchronize(owner.stream));
      output.write(reinterpret_cast<const char*>(current_v.data()), a * sizeof(double));
      output.write(reinterpret_cast<const char*>(current_d.data()), n * n * sizeof(double));
      if (!output) throw std::runtime_error("cannot write probe output");
    }
    std::cout << "public_aos=" << n << " auxiliary_aos=" << a
              << " orbital_primitive_pairs=" << pairs.view.count
              << " auxiliary_primitives=" << aux.view.count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
