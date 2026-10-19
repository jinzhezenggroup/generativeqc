#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "dft/ao_grid.hpp"
#include "integrals/cuda_one_electron_response.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::integrals;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  cudaStream_t stream{};
  std::vector<double*> allocations;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    cudaStreamSynchronize(stream);
    for (auto* pointer : allocations) cudaFree(pointer);
    cudaStreamDestroy(stream);
  }
  double* allocate(std::size_t count) {
    double* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
  double* upload(const std::vector<double>& values) {
    auto* pointer = allocate(values.size());
    check(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    return pointer;
  }
};
template <class Value>
std::vector<Value> read(std::ifstream& input, std::size_t count) {
  require(count <= (1ULL << 24), "oversized probe fixture");
  std::vector<Value> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value))),
      "short probe fixture");
  return values;
}
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  int device{-1};
  bool live{true};
  unsigned calls_before_expiration{std::numeric_limits<unsigned>::max()};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    int device{-1};
    if (!owner.live || !owner.calls_before_expiration || cudaGetDevice(&device) != cudaSuccess ||
        device != owner.device)
      return cudaErrorInvalidResourceHandle;
    if (owner.calls_before_expiration != std::numeric_limits<unsigned>::max())
      --owner.calls_before_expiration;
    return cudaSuccess;
  }
};

void host_admission() {
  const CudaOneElectronShape shape{2, 4, 2};
  const std::array<CudaOneElectronShell, 2> shells{{{0, 0, 0, 2, 0, 1}, {1, 0, 2, 2, 1, 1}}};
  const CudaOneElectronOptions options;
  std::size_t bytes{};
  check(cuda_one_electron_workspace(shape, shells, options, bytes));
  require(bytes > 0, "empty admitted arena");
  for (unsigned kind = 0; kind < 9; ++kind) {
    auto invalid = options;
    if (kind == 0) invalid.max_workspace_bytes = 1;
    if (kind == 1) invalid.max_jobs = 1;
    if (kind == 2) invalid.max_primitive_component_evaluations = 1;
    if (kind == 3) invalid.max_projection_terms = 1;
    if (kind == 4) invalid.radial_tile = 0;
    if (kind == 5) invalid.radial_tile = 65;
    if (kind == 6) invalid.symmetry_absolute_tolerance = -1;
    if (kind == 7) invalid.symmetry_relative_tolerance = -1;
    if (kind == 8) invalid.symmetry_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
    bytes = 27;
    require(cuda_one_electron_workspace(shape, shells, invalid, bytes) == cudaErrorInvalidValue &&
                !bytes,
            "invalid work/arena options admitted");
  }
  for (unsigned kind = 0; kind < 8; ++kind) {
    auto invalid = shells;
    if (kind == 0) invalid[0].atom = shape.atoms;
    if (kind == 1) invalid[0].angular = 4;
    if (kind == 2) invalid[0].primitive_begin = 1;
    if (kind == 3) invalid[0].primitive_count = 0;
    if (kind == 4) invalid[0].ao_begin = 1;
    if (kind == 5) invalid[0].ao_count = 2;
    if (kind == 6) invalid[1].primitive_count = SIZE_MAX;
    if (kind == 7) invalid[1].primitive_begin = 0;
    bytes = 27;
    require(cuda_one_electron_workspace(shape, invalid, options, bytes) == cudaErrorInvalidValue &&
                !bytes,
            "invalid shell topology admitted");
  }
  for (unsigned kind = 0; kind < 3; ++kind) {
    auto invalid = shape;
    if (kind == 0) invalid.atoms = SIZE_MAX;
    if (kind == 1) invalid.primitives = SIZE_MAX;
    if (kind == 2) invalid.nao = SIZE_MAX;
    bytes = 27;
    require(cuda_one_electron_workspace(invalid, shells, options, bytes) == cudaErrorInvalidValue &&
                !bytes,
            "overflowed shape admitted");
  }
  std::cout << "{\"host_admission\":true}\n";
}

void run(const char* input_path, const char* output_path) {
  std::ifstream input(input_path, std::ios::binary);
  const auto header = read<std::uint64_t>(input, 5);
  require(header[0] > 0 && header[0] <= 16 && header[1] > 0 && header[1] <= 32 && header[2] <= 1 &&
              header[4] <= 6,
          "invalid fixture shape");
  core::System system;
  system.basis_representation = header[2];
  const auto atoms = read<double>(input, 5 * header[0]);
  for (std::size_t atom = 0; atom < header[0]; ++atom)
    system.atoms.push_back({static_cast<int>(atoms[5 * atom]),
                            {atoms[5 * atom + 2], atoms[5 * atom + 3], atoms[5 * atom + 4]},
                            static_cast<int>(atoms[5 * atom + 1])});
  for (std::size_t shell = 0; shell < header[1]; ++shell) {
    const auto metadata = read<std::uint64_t>(input, 3);
    require(metadata[0] < header[0] && metadata[1] <= 3 && metadata[2] > 0 && metadata[2] <= 8,
            "invalid fixture shell");
    const auto radial = read<double>(input, 2 * metadata[2]);
    core::Shell item{static_cast<unsigned>(metadata[0]), static_cast<unsigned>(metadata[1]), {}};
    for (std::size_t primitive = 0; primitive < metadata[2]; ++primitive)
      item.primitives.push_back({radial[2 * primitive], radial[2 * primitive + 1]});
    system.shells.push_back(std::move(item));
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  dft::AoBasis basis(system);
  const CudaOneElectronShape shape{basis.natom, basis.nprimitive, basis.nao};
  std::vector<CudaOneElectronShell> shells;
  std::size_t primitive_begin{}, ao_begin{};
  for (const auto& shell : system.shells) {
    const auto public_count = system.basis_representation == core::kCartesianBasis
                                  ? molecule::cartesian_count(shell.angular_momentum)
                                  : 2 * shell.angular_momentum + 1;
    shells.push_back({shell.atom_index, shell.angular_momentum, primitive_begin,
                      shell.primitives.size(), ao_begin, public_count});
    primitive_begin += shell.primitives.size();
    ao_begin += public_count;
  }
  const auto direction = read<double>(input, 3 * shape.atoms);
  const auto matrix = shape.nao * shape.nao;
  const auto weights = read<double>(input, 4 * matrix);
  CudaOneElectronOptions options;
  options.radial_tile = header[3];
  std::size_t bytes{};
  check(cuda_one_electron_workspace(shape, shells, options, bytes));
  std::vector<double> charges;
  for (const auto& atom : system.atoms) charges.push_back(atom.ionic_charge());
  // Poison only after the actual native normalizer and AoBasis owner have packed
  // the source, so admission must protect trusted leaf indexing, not the parser.
  if (header[4] == 1) basis.packed[3 * shape.atoms] = -1;
  if (header[4] == 2) basis.packed[3 * shape.atoms + 2 * shape.primitives] = shape.atoms;
  if (header[4] == 3) basis.packed[3 * shape.atoms + 2 * shape.primitives + 3] = 4;
  if (header[4] == 4) basis.packed.back() = std::numeric_limits<double>::quiet_NaN();
  if (header[4] == 5) charges[0] = -1;
  if (header[4] == 6) basis.packed[3 * shape.atoms + 2 * shape.primitives + 4] = 3;
  Storage storage;
  Owner owner;
  check(cudaGetDevice(&owner.device));
  const CudaOneElectronReference reference{storage.upload(basis.packed),
                                           storage.upload(charges),
                                           owner.device,
                                           storage.stream,
                                           Owner::authenticate,
                                           &owner};
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* motion = storage.upload(direction);
  auto* panels = storage.upload(weights);
  const CudaOneElectronWeights source{panels, panels + matrix, panels + 2 * matrix,
                                      panels + 3 * matrix};
  const std::array<std::size_t, 6> counts{matrix, matrix,          matrix,
                                          matrix, 3 * shape.atoms, 3 * shape.atoms};
  std::array<double*, 6> pointers;
  for (unsigned index = 0; index < 6; ++index) pointers[index] = storage.allocate(counts[index]);
  const CudaOneElectronOutput output{pointers[0], pointers[1], pointers[2],
                                     pointers[3], pointers[4], pointers[5]};
  CudaOneElectronResponse response;
  require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidResourceHandle,
          "unprepared source action succeeded");
  require(response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
              cudaErrorInvalidValue,
          "short arena admitted");
  const auto prepared = response.prepare(shape, shells, options, reference, arena, bytes);
  if (header[4]) {
    require(prepared == cudaErrorInvalidValue &&
                response.apply(motion, source, output, storage.stream) ==
                    cudaErrorInvalidResourceHandle,
            "malformed native source admitted");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  check(response.apply(motion, source, output, storage.stream));
  require(response.work().completed_actions == 1 &&
              response.work().component_evaluations ==
                  response.work().component_evaluations_per_action &&
              response.work().projection_terms == response.work().projection_terms_per_action,
          "semantic work receipt mismatch");
  const auto receipt = response.work();
  std::ofstream capture(output_path, std::ios::binary);
  for (unsigned index = 0; index < 6; ++index) {
    const auto values = download(pointers[index], counts[index]);
    require(static_cast<bool>(capture.write(reinterpret_cast<const char*>(values.data()),
                                            values.size() * sizeof(double))),
            "probe output write failed");
  }
  const auto sentinel = [&] {
    for (unsigned index = 0; index < 6; ++index) {
      const auto values = std::vector<double>(counts[index], 27);
      check(cudaMemcpy(pointers[index], values.data(), values.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
    }
  };
  const auto unchanged = [&] {
    for (unsigned index = 0; index < 6; ++index)
      require(download(pointers[index], counts[index]) == std::vector<double>(counts[index], 27),
              "failed source action published partial output");
  };
  sentinel();
  require(response.apply(motion, source, output, nullptr) == cudaErrorInvalidResourceHandle,
          "foreign stream admitted");
  unchanged();
  for (unsigned first = 0; first < 6; ++first)
    for (unsigned second = 0; second < first; ++second) {
      auto aliased = pointers;
      aliased[first] = aliased[second];
      const CudaOneElectronOutput bad{aliased[0], aliased[1], aliased[2],
                                      aliased[3], aliased[4], aliased[5]};
      require(response.apply(motion, source, bad, storage.stream) == cudaErrorInvalidValue,
              "output pair alias admitted");
      unchanged();
    }
  auto bad = output;
  bad.core = panels;
  require(response.apply(motion, source, bad, storage.stream) == cudaErrorInvalidValue,
          "input/output alias admitted");
  unchanged();
  bad.core = arena;
  require(response.apply(motion, source, bad, storage.stream) == cudaErrorInvalidValue,
          "arena/output alias admitted");
  unchanged();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpy(motion, &nan, sizeof(double), cudaMemcpyHostToDevice));
  require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidValue,
          "nonfinite direction admitted");
  unchanged();
  check(cudaMemcpy(motion, direction.data(), direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  for (unsigned panel = 0; panel < 4; ++panel) {
    check(cudaMemcpy(panels + panel * matrix, &nan, sizeof(double), cudaMemcpyHostToDevice));
    require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidValue,
            "nonfinite weight admitted");
    unchanged();
    check(cudaMemcpy(panels + panel * matrix, weights.data() + panel * matrix,
                     matrix * sizeof(double), cudaMemcpyHostToDevice));
  }
  if (shape.nao > 1) {
    const auto asymmetric = weights[1] + 1;
    check(cudaMemcpy(panels + 1, &asymmetric, sizeof(double), cudaMemcpyHostToDevice));
    require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidValue,
            "asymmetric weight admitted");
    unchanged();
    check(cudaMemcpy(panels + 1, weights.data() + 1, sizeof(double), cudaMemcpyHostToDevice));
  }
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeGlobal));
  const auto captured = response.apply(motion, source, output, storage.stream);
  cudaGraph_t graph{};
  const auto ended = cudaStreamEndCapture(storage.stream, &graph);
  if (graph) cudaGraphDestroy(graph);
  require(captured == cudaErrorStreamCaptureUnsupported && ended == cudaSuccess,
          "capture admission changed stream state");
  unchanged();
  owner.calls_before_expiration = 1;
  require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidResourceHandle,
          "late owner revocation published output");
  unchanged();
  owner.calls_before_expiration = std::numeric_limits<unsigned>::max();
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  owner.live = false;
  require(response.apply(motion, source, output, storage.stream) == cudaErrorInvalidResourceHandle,
          "stale owner admitted");
  unchanged();
  owner.live = true;
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  check(response.apply(motion, source, output, storage.stream));
  require(
      response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
              cudaErrorInvalidValue &&
          response.apply(motion, source, output, storage.stream) == cudaErrorInvalidResourceHandle,
      "failed reprepare retained a live source");
  std::cout << "{\"lifecycle\":true,\"jobs\":" << receipt.jobs
            << ",\"records\":" << receipt.primitive_records_per_action
            << ",\"components\":" << receipt.component_evaluations_per_action
            << ",\"projections\":" << receipt.projection_terms_per_action
            << ",\"workspace_bytes\":" << receipt.workspace_bytes << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "host")
      host_admission();
    else if (argc == 3)
      run(argv[1], argv[2]);
    else
      throw std::runtime_error("expected host or input/output fixture paths");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
