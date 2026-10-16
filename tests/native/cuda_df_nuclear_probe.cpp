#include <algorithm>
#include <array>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/cuda_df_nuclear_response.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::integrals;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
template <class Value>
std::vector<Value> read(std::ifstream& input, std::size_t count) {
  require(count <= (1ULL << 24), "oversized fixture");
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
  double* allocate(std::size_t count) {
    void* pointer{};
    check(cudaMalloc(&pointer, count * sizeof(double)));
    allocations.push_back(pointer);
    return static_cast<double*>(pointer);
  }
  double* upload(const std::vector<double>& values) {
    auto* pointer = allocate(values.size());
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
  scf::CudaDensityFittingIntegralSource* source{};
  scf::CudaDfReadonlyBasis basis;
  unsigned remaining{UINT_MAX};
  ~Owner() { scf::destroy_cuda_density_fitting_integral_source(source); }
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    if (!owner.remaining || scf::cuda_density_fitting_readonly_basis(owner.source) != owner.basis)
      return cudaErrorInvalidResourceHandle;
    if (owner.remaining != UINT_MAX) --owner.remaining;
    return cudaSuccess;
  }
};
std::vector<CudaDfNuclearShell> topology(const core::System& system, std::size_t& primitive_begin) {
  std::vector<CudaDfNuclearShell> result;
  std::size_t ao_begin{}, cartesian_begin{};
  for (const auto& shell : system.shells) {
    const auto angular = static_cast<std::size_t>(shell.angular_momentum);
    const auto count = (angular + 1) * (angular + 2) / 2;
    const auto public_count =
        system.basis_representation == core::kSphericalBasis ? 2 * angular + 1 : count;
    result.push_back({shell.atom_index, angular, primitive_begin, shell.primitives.size(), ao_begin,
                      public_count, cartesian_begin});
    primitive_begin += shell.primitives.size();
    ao_begin += public_count;
    cartesian_begin += count;
  }
  return result;
}
void host_admission() {
  const CudaDfNuclearShape shape{2, 4, 1, 1, 1, 1};
  const std::array<CudaDfNuclearShell, 1> orbital{{{0, 0, 0, 2, 0, 1, 0}}},
      auxiliary{{{1, 0, 2, 2, 0, 1, 0}}};
  std::size_t bytes{};
  check(cuda_df_nuclear_workspace(shape, orbital, auxiliary, {}, bytes));
  const auto admitted = bytes;
  for (unsigned gate = 0; gate < 6; ++gate) {
    CudaDfNuclearOptions options;
    if (gate == 0) options.max_workspace_bytes = admitted - 1;
    if (gate == 1) options.max_groups = 1;
    if (gate == 2) options.max_component_evaluations = 1;
    if (gate == 3) options.max_projection_terms = 1;
    if (gate == 4) options.max_reduction_terms_bound = 1;
    if (gate == 5) options.radial_tile = 65;
    require(cuda_df_nuclear_workspace(shape, orbital, auxiliary, options, bytes) != cudaSuccess &&
                !bytes,
            "host resource bound ignored");
  }
  auto overflowing = shape;
  overflowing.nao = SIZE_MAX;
  require(cuda_df_nuclear_workspace(overflowing, orbital, auxiliary, {}, bytes) != cudaSuccess,
          "overflow admitted");
  auto malformed = orbital;
  malformed[0].angular = 4;
  require(cuda_df_nuclear_workspace(shape, malformed, auxiliary, {}, bytes) != cudaSuccess,
          "orbital g admitted");
  malformed = orbital;
  malformed[0].primitive_begin = 1;
  require(cuda_df_nuclear_workspace(shape, malformed, auxiliary, {}, bytes) != cudaSuccess,
          "noncontiguous topology admitted");
  CudaDfNuclearResponse empty;
  require(empty.values(nullptr, {}, nullptr) == cudaErrorInvalidResourceHandle &&
              empty.contract(nullptr, {}, {}, nullptr) == cudaErrorInvalidResourceHandle,
          "unprepared source admitted");
}
void run(const char* source, const char* result) {
  std::ifstream file(source, std::ios::binary);
  const auto header = read<std::uint64_t>(file, 7);
  require(header[0] >= 2 && header[0] <= 8 && header[1] > 0 && header[1] <= 8 && header[2] > 0 &&
              header[2] <= 8 && header[3] <= 1 && header[4] <= 1 && header[6] <= 4,
          "invalid fixture metadata");
  core::System orbital, auxiliary;
  const auto positions = read<double>(file, 3 * header[0]);
  for (std::size_t atom = 0; atom < header[0]; ++atom)
    orbital.atoms.push_back(
        {1, {positions[3 * atom], positions[3 * atom + 1], positions[3 * atom + 2]}});
  orbital.multiplicity = header[0] % 2 ? 2 : 1;
  auxiliary = orbital;
  orbital.basis_representation = header[3] ? core::kSphericalBasis : core::kCartesianBasis;
  auxiliary.basis_representation = header[4] ? core::kSphericalBasis : core::kCartesianBasis;
  for (unsigned kind = 0; kind < 2; ++kind) {
    auto& system = kind ? auxiliary : orbital;
    for (std::size_t index = 0; index < header[1 + kind]; ++index) {
      const auto metadata = read<std::uint64_t>(file, 3);
      require(metadata[0] < header[0] && metadata[1] <= (kind ? 4 : 3) && metadata[2] > 0 &&
                  metadata[2] <= 8,
              "invalid shell");
      const auto radial = read<double>(file, 2 * metadata[2]);
      core::Shell shell{static_cast<unsigned>(metadata[0]), static_cast<unsigned>(metadata[1]), {}};
      for (std::size_t primitive = 0; primitive < metadata[2]; ++primitive)
        shell.primitives.push_back({radial[2 * primitive], radial[2 * primitive + 1]});
      system.shells.push_back(std::move(shell));
    }
  }
  std::string detail;
  require(molecule::validate_and_normalize(orbital, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              molecule::validate_and_normalize(auxiliary, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  Storage storage;
  Owner owner;
  int device{-1};
  check(cudaGetDevice(&device));
  std::vector<double> source_metric;
  std::size_t nao{}, naux{};
  require(scf::create_cuda_density_fitting_integral_source(device, {orbital}, {auxiliary},
                                                           &owner.source, source_metric, nao, naux,
                                                           detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  owner.basis = scf::cuda_density_fitting_readonly_basis(owner.source);
  require(static_cast<bool>(owner.basis), "no actual readonly DF basis");
  const auto& basis = owner.basis;
  const CudaDfNuclearShape shape{basis.atoms, basis.primitives,    nao,
                                 naux,        basis.cartesian_nao, basis.cartesian_naux};
  std::size_t primitive_begin{};
  auto orbital_shells = topology(orbital, primitive_begin),
       auxiliary_shells = topology(auxiliary, primitive_begin);
  const auto raw_count = nao * nao * naux, metric_count = naux * naux;
  auto* direction = storage.upload(read<double>(file, 3 * shape.atoms));
  auto* weight_storage = storage.upload(read<double>(file, 2 * (raw_count + metric_count)));
  const CudaDfNuclearWeights weights{weight_storage, weight_storage + raw_count,
                                     weight_storage + raw_count + metric_count,
                                     weight_storage + 2 * raw_count + metric_count};
  CudaDfNuclearOptions options;
  options.radial_tile = header[5];
  std::size_t bytes{};
  check(cuda_df_nuclear_workspace(shape, orbital_shells, auxiliary_shells, options, bytes));
  auto* arena = storage.allocate((bytes + sizeof(double) - 1) / sizeof(double));
  CudaDfNuclearReference reference{owner.basis, storage.stream, Owner::authenticate, &owner};
  if (header[6] == 1) --bytes;
  if (header[6] == 2)
    reference.basis.exponents = reinterpret_cast<const double*>(
        reinterpret_cast<const char*>(reference.basis.exponents) + 1);
  if (header[6] == 3) orbital_shells[0].atom = (orbital_shells[0].atom + 1) % shape.atoms;
  if (header[6] == 4) ++reference.basis.orbital_shells;
  CudaDfNuclearResponse response;
  const auto prepared =
      response.prepare(shape, orbital_shells, auxiliary_shells, options, reference, arena, bytes);
  if (header[6]) {
    require(prepared != cudaSuccess, "malformed source admitted");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  auto* value_storage = storage.allocate(2 * (raw_count + metric_count));
  auto* vector_storage = storage.allocate(6 * shape.atoms);
  const CudaDfNuclearValues values{value_storage, value_storage + raw_count,
                                   value_storage + raw_count + metric_count,
                                   value_storage + 2 * raw_count + metric_count};
  const CudaDfNuclearVectors vectors{vector_storage, vector_storage + 3 * shape.atoms};
  check(response.values(direction, values, storage.stream));
  check(response.contract(direction, weights, vectors, storage.stream));
  const auto expected_values = download(value_storage, 2 * (raw_count + metric_count));
  const auto expected_vectors = download(vector_storage, 6 * shape.atoms);
  check(response.values(direction, values, storage.stream));
  check(response.contract(direction, weights, vectors, storage.stream));
  const auto preserved = [&] {
    require(download(value_storage, expected_values.size()) == expected_values &&
                download(vector_storage, expected_vectors.size()) == expected_vectors,
            "replay or rejected pass changed published outputs");
  };
  preserved();
  auto alias = values;
  alias.metric = alias.three_center;
  require(response.values(direction, alias, storage.stream) != cudaSuccess,
          "aliased value outputs admitted");
  alias = values;
  // Value evaluation does not read weights: use a real immutable basis alias.
  alias.three_center = const_cast<double*>(basis.exponents);
  require(response.values(direction, alias, storage.stream) != cudaSuccess,
          "immutable basis output alias admitted");
  alias.three_center = arena;
  require(response.values(direction, alias, storage.stream) != cudaSuccess,
          "arena output alias admitted");
  require(response.contract(direction, weights, {weight_storage, vectors.hvp}, storage.stream) !=
              cudaSuccess,
          "weight/output alias admitted");
  require(response.values(nullptr, values, storage.stream) != cudaSuccess &&
              response.contract(arena, weights, vectors, storage.stream) != cudaSuccess,
          "null or arena direction admitted");
  const auto saved_direction = download(direction, 1)[0];
  storage.set(direction, std::numeric_limits<double>::quiet_NaN());
  require(response.values(direction, values, storage.stream) != cudaSuccess &&
              response.contract(direction, weights, vectors, storage.stream) != cudaSuccess,
          "nonfinite direction admitted");
  storage.set(direction, saved_direction);
  const std::array<const double*, 4> panels{weights.three_center, weights.metric,
                                            weights.direction_three_center,
                                            weights.direction_metric};
  for (unsigned index = 0; index < panels.size(); ++index) {
    auto* pointer = const_cast<double*>(panels[index]);
    const auto extent = index % 2 == 0 ? nao : naux;
    const auto offset = extent > 1 ? (index % 2 == 0 ? naux : 1) : 0;
    const auto saved = download(pointer, offset + 1);
    storage.set(pointer, std::numeric_limits<double>::infinity());
    require(response.contract(direction, weights, vectors, storage.stream) != cudaSuccess,
            "nonfinite weight admitted");
    storage.set(pointer, saved[0]);
    if ((index % 2 == 0 ? nao : naux) > 1) {
      storage.set(pointer + offset, saved[offset] + 1);
      require(response.contract(direction, weights, vectors, storage.stream) != cudaSuccess,
              "nonsymmetric weight admitted");
      storage.set(pointer + offset, saved[offset]);
    }
  }
  cudaStream_t other{};
  check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  require(response.values(direction, values, other) != cudaSuccess &&
              response.contract(direction, weights, vectors, other) != cudaSuccess,
          "foreign stream admitted");
  check(cudaStreamDestroy(other));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeThreadLocal));
  require(response.values(direction, values, storage.stream) == cudaErrorStreamCaptureUnsupported &&
              response.contract(direction, weights, vectors, storage.stream) ==
                  cudaErrorStreamCaptureUnsupported,
          "capture admitted");
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(storage.stream, &graph));
  check(cudaGraphDestroy(graph));
  owner.remaining = 1;
  require(response.values(direction, values, storage.stream) == cudaErrorInvalidResourceHandle,
          "late value revocation admitted");
  owner.remaining = 1;
  require(response.contract(direction, weights, vectors, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "late contraction revocation admitted");
  require(response.values(direction, values, storage.stream) == cudaErrorInvalidResourceHandle &&
              response.contract(direction, weights, vectors, storage.stream) ==
                  cudaErrorInvalidResourceHandle,
          "stale owner admitted");
  preserved();
  const auto work = response.work();
  require(work.value_passes == 2 && work.contraction_passes == 2, "failed pass reported completed");
  std::ofstream output(result, std::ios::binary);
  output.write(reinterpret_cast<const char*>(expected_values.data()),
               expected_values.size() * sizeof(double));
  output.write(reinterpret_cast<const char*>(expected_vectors.data()),
               expected_vectors.size() * sizeof(double));
  std::cout << "{\"lifecycle\":true,\"workspace_bytes\":" << work.workspace_bytes
            << ",\"groups\":" << work.groups
            << ",\"records_per_pass\":" << work.radial_records_per_pass
            << ",\"components_per_pass\":" << work.component_evaluations_per_pass
            << ",\"projection_terms_bound\":" << work.projection_terms_bound_per_pass
            << ",\"reduction_terms_bound\":" << work.reduction_terms_bound_per_pass
            << ",\"submitted_records\":" << work.radial_records
            << ",\"submitted_components\":" << work.component_evaluations << "}\n";
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
