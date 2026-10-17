#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "integrals/cuda_ecp_nuclear_response.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::integrals;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
template <class Value>
std::vector<Value> read(std::ifstream& file, std::size_t count) {
  std::vector<Value> values(count);
  require(
      static_cast<bool>(file.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value))),
      "short fixture");
  return values;
}
struct Storage {
  cudaStream_t stream{};
  std::vector<void*> pointers;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    cudaStreamSynchronize(stream);
    for (auto* pointer : pointers) cudaFree(pointer);
    cudaStreamDestroy(stream);
  }
  template <class Value = double>
  Value* allocate(std::size_t count) {
    void* pointer{};
    check(cudaMalloc(&pointer, count * sizeof(Value)));
    pointers.push_back(pointer);
    return static_cast<Value*>(pointer);
  }
  template <class Value>
  Value* upload(const std::vector<Value>& values) {
    auto* pointer = allocate<Value>(values.size());
    check(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(Value),
                          cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
    return pointer;
  }
};
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  int device{-1};
  unsigned remaining{UINT32_MAX};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    int device{-1};
    if (!owner.remaining || cudaGetDevice(&device) != cudaSuccess || device != owner.device)
      return cudaErrorInvalidResourceHandle;
    if (owner.remaining != UINT32_MAX) --owner.remaining;
    return cudaSuccess;
  }
};
void host_only() {
  const CudaEcpNuclearShape shape{2, 2, 2, 3};
  const std::array<unsigned, 1> centers{0};
  std::size_t bytes{};
  check(cuda_ecp_nuclear_workspace(shape, centers, {}, bytes));
  const auto admitted = bytes;
  for (unsigned gate = 0; gate < 5; ++gate) {
    CudaEcpNuclearOptions options;
    if (gate == 0) options.max_workspace_bytes = admitted - 1;
    if (gate == 1) options.max_terms = 1;
    if (gate == 2) options.max_ao_evaluations = 1;
    if (gate == 3) options.max_affine_terms = 1;
    if (gate == 4) options.max_reduction_terms = 1;
    require(cuda_ecp_nuclear_workspace(shape, centers, options, bytes) != cudaSuccess && !bytes,
            "host work/byte bound ignored");
  }
  auto malformed = shape;
  malformed.nao = SIZE_MAX;
  require(cuda_ecp_nuclear_workspace(malformed, centers, {}, bytes) != cudaSuccess,
          "overflow admitted");
  require(
      cuda_ecp_nuclear_workspace(shape, std::array<unsigned, 2>{0, 0}, {}, bytes) != cudaSuccess,
      "duplicate center admitted");
  require(cuda_ecp_nuclear_workspace(shape, std::array<unsigned, 1>{2}, {}, bytes) != cudaSuccess,
          "foreign center admitted");
  CudaEcpNuclearResponse response;
  require(response.values(nullptr, {}, nullptr) == cudaErrorInvalidResourceHandle &&
              response.contract(nullptr, {}, {}, nullptr) == cudaErrorInvalidResourceHandle,
          "unprepared ECP response admitted");
}
void run(const char* input_path, const char* output_path) {
  std::ifstream file(input_path, std::ios::binary);
  const auto header = read<std::uint64_t>(file, 7);
  require(header[0] >= 2 && header[0] <= 8 && header[1] > 0 && header[1] <= 16 && header[2] <= 1 &&
              header[3] > 0 && header[3] <= 100 && header[4] > 0 && header[4] <= header[0] &&
              header[5] <= 10 && header[6] <= 1,
          "invalid fixture header");
  const auto atoms = read<double>(file, 5 * header[0]);
  core::System system;
  system.multiplicity = header[0] % 2 ? 2 : 1;
  system.basis_representation = header[2] ? core::kSphericalBasis : core::kCartesianBasis;
  for (std::size_t atom = 0; atom < header[0]; ++atom)
    system.atoms.push_back({static_cast<int>(atoms[5 * atom]),
                            {atoms[5 * atom + 2], atoms[5 * atom + 3], atoms[5 * atom + 4]},
                            static_cast<int>(atoms[5 * atom + 1])});
  for (std::size_t index = 0; index < header[1]; ++index) {
    const auto metadata = read<std::uint64_t>(file, 3);
    require(metadata[0] < header[0] && metadata[1] <= 3 && metadata[2] > 0 && metadata[2] <= 8,
            "invalid fixture shell");
    const auto radial = read<double>(file, 2 * metadata[2]);
    core::Shell shell{static_cast<unsigned>(metadata[0]), static_cast<unsigned>(metadata[1]), {}};
    for (std::size_t primitive = 0; primitive < metadata[2]; ++primitive)
      shell.primitives.push_back({radial[2 * primitive], radial[2 * primitive + 1]});
    system.shells.push_back(std::move(shell));
  }
  const auto term_data = read<double>(file, 5 * header[3]);
  for (std::size_t index = 0; index < header[3]; ++index)
    system.ecp_terms.push_back({static_cast<unsigned>(term_data[5 * index]),
                                static_cast<int>(term_data[5 * index + 1]),
                                static_cast<unsigned>(term_data[5 * index + 2]),
                                term_data[5 * index + 3], term_data[5 * index + 4]});
  const auto centers = read<unsigned>(file, header[4]);
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  dft::AoBasis basis(system);
  const CudaEcpNuclearShape shape{basis.natom, basis.nprimitive, basis.nao,
                                  system.ecp_terms.size()};
  const auto matrix = shape.nao * shape.nao, coordinates = 3 * shape.atoms;
  auto motion = read<double>(file, coordinates), density = read<double>(file, 2 * matrix);
  if (header[5] == 1) basis.packed[3 * shape.atoms] = std::numeric_limits<double>::quiet_NaN();
  if (header[5] == 2) basis.packed[3 * shape.atoms + 2 * shape.primitives] = shape.atoms;
  if (header[5] == 3) system.ecp_terms[0].channel = 4;
  if (header[5] == 4) system.ecp_terms[0].power = 5;
  if (header[5] == 5) system.ecp_terms[0].exponent = -1;
  if (header[5] == 8) motion[0] = std::numeric_limits<double>::infinity();
  if (header[5] == 9) density[1] += 1;
  Storage storage;
  Owner owner;
  check(cudaGetDevice(&owner.device));
  CudaEcpNuclearReference reference{storage.upload(basis.packed),
                                    storage.upload(system.ecp_terms),
                                    owner.device,
                                    storage.stream,
                                    Owner::authenticate,
                                    &owner};
  std::size_t bytes{};
  check(cuda_ecp_nuclear_workspace(shape, centers, {}, bytes));
  auto* arena = storage.allocate(bytes / sizeof(double));
  if (header[5] == 6) --bytes;
  if (header[5] == 7)
    reference.terms =
        reinterpret_cast<const core::EcpTerm*>(reinterpret_cast<const char*>(reference.terms) + 1);
  CudaEcpNuclearResponse response;
  const auto prepared = response.prepare(shape, centers, {}, reference, arena, bytes);
  if (header[5] && header[5] <= 7) {
    require(prepared != cudaSuccess, "malformed ECP source admitted");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  auto* direction = storage.upload(motion);
  auto* panels = storage.upload(density);
  auto* value_storage = storage.allocate(4 * matrix);
  auto* vector_storage = storage.allocate(4 * coordinates);
  const CudaEcpNuclearValues values{value_storage, value_storage + matrix,
                                    value_storage + 2 * matrix, value_storage + 3 * matrix};
  const CudaEcpNuclearWeights weights{panels, panels + matrix};
  const CudaEcpNuclearVectors vectors{vector_storage, vector_storage + coordinates,
                                      vector_storage + 2 * coordinates,
                                      vector_storage + 3 * coordinates};
  if (header[5]) {
    const std::vector<double> sentinel(4 * coordinates, 27);
    check(cudaMemcpy(vector_storage, sentinel.data(), sentinel.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    require(response.contract(direction, weights, vectors, storage.stream) != cudaSuccess &&
                download(vector_storage, sentinel.size()) == sentinel,
            "rejected ECP action published vectors");
    if (header[5] == 10) {
      const std::vector<double> matrix_sentinel(4 * matrix, 27);
      check(cudaMemcpy(value_storage, matrix_sentinel.data(),
                       matrix_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice));
      require(response.values(direction, values, storage.stream) == cudaErrorInvalidValue &&
                  download(value_storage, matrix_sentinel.size()) == matrix_sentinel &&
                  response.work().ao_evaluations == 2 * response.work().ao_evaluations_per_action &&
                  !response.work().value_actions && !response.work().contraction_actions,
              "unconverged ECP quadrature published values or skipped numerical work");
    }
    std::cout << "{\"action_rejected\":true}\n";
    return;
  }
  check(response.values(direction, values, storage.stream));
  check(response.contract(direction, weights, vectors, storage.stream));
  const auto expected_values = download(value_storage, 4 * matrix),
             expected_vectors = download(vector_storage, 4 * coordinates);
  const auto preserved = [&] {
    require(download(value_storage, expected_values.size()) == expected_values &&
                download(vector_storage, expected_vectors.size()) == expected_vectors,
            "rejected ECP action changed publication");
  };
  auto alias = values;
  alias.nonlocal = alias.local;
  require(response.values(direction, alias, storage.stream) != cudaSuccess,
          "aliased ECP outputs admitted");
  alias = values;
  alias.local = const_cast<double*>(reference.basis);
  require(response.values(direction, alias, storage.stream) != cudaSuccess,
          "immutable ECP basis output alias admitted");
  require(response.contract(
              direction, weights,
              {panels, vectors.gradient_nonlocal, vectors.hvp_local, vectors.hvp_nonlocal},
              storage.stream) != cudaSuccess,
          "density output alias admitted");
  require(response.values(nullptr, values, storage.stream) != cudaSuccess &&
              response.contract(arena, weights, vectors, storage.stream) != cudaSuccess,
          "null/arena ECP direction admitted");
  if (header[6]) {
    check(response.values(direction, values, storage.stream));
    check(response.contract(direction, weights, vectors, storage.stream));
    preserved();
    cudaStream_t other{};
    check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
    require(response.values(direction, values, other) != cudaSuccess &&
                response.contract(direction, weights, vectors, other) != cudaSuccess,
            "foreign ECP stream admitted");
    check(cudaStreamDestroy(other));
    check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeThreadLocal));
    require(
        response.values(direction, values, storage.stream) == cudaErrorStreamCaptureUnsupported &&
            response.contract(direction, weights, vectors, storage.stream) ==
                cudaErrorStreamCaptureUnsupported,
        "ECP capture admitted");
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(storage.stream, &graph));
    check(cudaGraphDestroy(graph));
    owner.remaining = 1;
    require(response.values(direction, values, storage.stream) == cudaErrorInvalidResourceHandle,
            "late ECP value revocation admitted");
    owner.remaining = 1;
    require(response.contract(direction, weights, vectors, storage.stream) ==
                cudaErrorInvalidResourceHandle,
            "late ECP vector revocation admitted");
    require(response.values(direction, values, storage.stream) == cudaErrorInvalidResourceHandle,
            "stale ECP owner admitted");
  }
  preserved();
  std::ofstream output(output_path, std::ios::binary);
  require(static_cast<bool>(output.write(reinterpret_cast<const char*>(expected_values.data()),
                                         expected_values.size() * sizeof(double))) &&
              static_cast<bool>(output.write(reinterpret_cast<const char*>(expected_vectors.data()),
                                             expected_vectors.size() * sizeof(double))),
          "cannot write ECP result");
  const auto& work = response.work();
  std::cout << "{\"lifecycle\":true,\"workspace_bytes\":" << work.workspace_bytes
            << ",\"layers_per_action\":" << work.radial_layers_per_action
            << ",\"ao_per_action\":" << work.ao_evaluations_per_action
            << ",\"projection_terms_per_action\":" << work.projection_terms_per_action
            << ",\"matrix_terms\":" << work.matrix_affine_terms
            << ",\"contraction_terms\":" << work.contraction_affine_terms
            << ",\"value_actions\":" << work.value_actions
            << ",\"contraction_actions\":" << work.contraction_actions << "}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--host-only") {
      host_only();
      return 0;
    }
    require(argc == 3, "usage: cuda_ecp_nuclear_probe input.bin output.bin | --host-only");
    run(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
