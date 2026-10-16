#include <array>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "dft/ao_grid.hpp"
#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::dft;
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
  require(count <= (1ULL << 24), "oversized fixture");
  std::vector<Value> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value))),
      "short fixture");
  return values;
}
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  bool live{true};
  unsigned remaining{UINT_MAX};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    if (!owner.live || !owner.remaining) return cudaErrorInvalidResourceHandle;
    if (owner.remaining != UINT_MAX) --owner.remaining;
    return cudaSuccess;
  }
};
void run(const char* source, const char* result) {
  std::ifstream input(source, std::ios::binary);
  const auto header = read<std::uint64_t>(input, 7);
  require(header[0] > 0 && header[0] <= 8 && header[1] > 0 && header[1] <= 8 && header[2] <= 1 &&
              header[3] >= 1 && header[3] <= 2 && header[4] <= 5 && header[6] <= 6,
          "invalid fixture header");
  core::System system;
  system.basis_representation = header[2] ? core::kSphericalBasis : core::kCartesianBasis;
  const auto atoms = read<double>(input, 5 * header[0]);
  for (std::size_t atom = 0; atom < header[0]; ++atom)
    system.atoms.push_back({static_cast<int>(atoms[5 * atom]),
                            {atoms[5 * atom + 2], atoms[5 * atom + 3], atoms[5 * atom + 4]},
                            static_cast<int>(atoms[5 * atom + 1])});
  for (std::size_t index = 0; index < header[1]; ++index) {
    const auto metadata = read<std::uint64_t>(input, 3);
    require(metadata[0] < header[0] && metadata[1] <= 3 && metadata[2] > 0 && metadata[2] <= 8,
            "invalid shell");
    const auto radial = read<double>(input, 2 * metadata[2]);
    core::Shell shell{static_cast<unsigned>(metadata[0]), static_cast<unsigned>(metadata[1]), {}};
    for (std::size_t primitive = 0; primitive < metadata[2]; ++primitive)
      shell.primitives.push_back({radial[2 * primitive], radial[2 * primitive + 1]});
    system.shells.push_back(std::move(shell));
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  AoBasis basis(system);
  const CudaWb97mvJkShape shape{basis.natom, basis.nprimitive, basis.nao, header[3]};
  std::vector<integrals::CudaOneElectronShell> shells;
  std::size_t primitive_begin{}, ao_begin{};
  for (const auto& shell : system.shells) {
    const auto count = header[2] ? 2 * shell.angular_momentum + 1
                                 : molecule::cartesian_count(shell.angular_momentum);
    shells.push_back({shell.atom_index, shell.angular_momentum, primitive_begin,
                      shell.primitives.size(), ao_begin, count});
    primitive_begin += shell.primitives.size();
    ao_begin += count;
  }
  const auto direction = read<double>(input, 3 * shape.atoms);
  const auto panels = shape.spins * shape.nao * shape.nao;
  const auto densities = read<double>(input, 2 * panels);
  CudaWb97mvJkOptions options;
  options.part = static_cast<CudaWb97mvJkPart>(header[4]);
  options.radial_tile = header[5];
  std::size_t bytes{};
  check(cuda_wb97mv_jk_workspace(shape, shells, options, bytes));
  const auto ao_offset = 3 * shape.atoms + 2 * shape.primitives;
  if (header[6] == 1) basis.packed[3 * shape.atoms] = -1;
  if (header[6] == 2) basis.packed[ao_offset] = shape.atoms;
  if (header[6] == 3) basis.packed[ao_offset + 3] = 4;
  if (header[6] == 4) basis.packed.back() = std::numeric_limits<double>::quiet_NaN();
  if (header[6] == 5) basis.packed[ao_offset + 4] = 3;
  if (header[6] == 6) basis.packed[ao_offset + 3] = std::numeric_limits<double>::quiet_NaN();
  Storage storage;
  Owner owner;
  int device{-1};
  check(cudaGetDevice(&device));
  const CudaWb97mvJkReference reference{storage.upload(basis.packed), device, storage.stream,
                                        Owner::authenticate, &owner};
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* motion = storage.upload(direction);
  auto* density = storage.upload(densities);
  auto* delta = density + panels;
  const std::array<std::size_t, 5> counts{1, panels, panels, 3 * shape.atoms, 3 * shape.atoms};
  std::array<double*, 5> pointers;
  for (unsigned index = 0; index < 5; ++index) pointers[index] = storage.allocate(counts[index]);
  const auto outputs = [](const auto& buffers) {
    return CudaWb97mvJkOutput{buffers[0], buffers[1], buffers[2], buffers[3], buffers[4]};
  };
  const auto output = outputs(pointers);
  CudaWb97mvNuclearJk response;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "unprepared action succeeded");
  require(response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
              cudaErrorInvalidValue,
          "short arena admitted");
  const auto prepared = response.prepare(shape, shells, options, reference, arena, bytes);
  if (header[6]) {
    require(prepared == cudaErrorInvalidValue &&
                response.apply(motion, density, delta, output, storage.stream) ==
                    cudaErrorInvalidResourceHandle,
            "malformed basis admitted");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  check(response.apply(motion, density, delta, output, storage.stream));
  const auto receipt = response.work();
  require(receipt.completed_actions == 1 &&
              receipt.component_evaluations == receipt.component_evaluations_per_action &&
              receipt.projection_terms_bound == receipt.projection_terms_bound_per_action &&
              receipt.reduction_terms == receipt.reduction_terms_per_action,
          "semantic work receipt mismatch");
  std::ofstream capture(result, std::ios::binary);
  for (unsigned index = 0; index < 5; ++index) {
    const auto values = download(pointers[index], counts[index]);
    require(static_cast<bool>(capture.write(reinterpret_cast<const char*>(values.data()),
                                            values.size() * sizeof(double))),
            "output write failed");
  }
  const auto sentinel = [&] {
    for (unsigned index = 0; index < 5; ++index) {
      const std::vector<double> values(counts[index], 27);
      check(cudaMemcpy(pointers[index], values.data(), values.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
    }
  };
  const auto unchanged = [&] {
    for (unsigned index = 0; index < 5; ++index)
      require(download(pointers[index], counts[index]) == std::vector<double>(counts[index], 27),
              "failure published partial output");
  };
  sentinel();
  require(response.apply(motion, density, delta, output, nullptr) == cudaErrorInvalidResourceHandle,
          "foreign stream admitted");
  unchanged();
  for (unsigned first = 0; first < 5; ++first)
    for (unsigned second = 0; second < first; ++second) {
      auto aliased = pointers;
      aliased[first] = aliased[second];
      require(response.apply(motion, density, delta, outputs(aliased), storage.stream) ==
                  cudaErrorInvalidValue,
              "output alias admitted");
      unchanged();
    }
  for (const auto* input_pointer :
       {motion, density, delta, arena, const_cast<double*>(reference.basis)}) {
    auto aliased = pointers;
    aliased[0] = const_cast<double*>(input_pointer);
    require(response.apply(motion, density, delta, outputs(aliased), storage.stream) ==
                cudaErrorInvalidValue,
            "input/arena output alias admitted");
    unchanged();
  }
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpy(motion, &nan, sizeof(double), cudaMemcpyHostToDevice));
  require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "nonfinite direction admitted");
  unchanged();
  check(cudaMemcpy(motion, direction.data(), direction.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  for (unsigned index = 0; index < 2 * shape.spins; ++index) {
    const auto offset = index * shape.nao * shape.nao;
    check(cudaMemcpy(density + offset, &nan, sizeof(double), cudaMemcpyHostToDevice));
    require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
            "nonfinite density admitted");
    unchanged();
    check(cudaMemcpy(density + offset, densities.data() + offset, sizeof(double),
                     cudaMemcpyHostToDevice));
    if (shape.nao > 1) {
      const auto asymmetric = densities[offset + 1] + 1;
      check(cudaMemcpy(density + offset + 1, &asymmetric, sizeof(double), cudaMemcpyHostToDevice));
      require(
          response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "asymmetric density admitted");
      unchanged();
      check(cudaMemcpy(density + offset + 1, densities.data() + offset + 1, sizeof(double),
                       cudaMemcpyHostToDevice));
    }
  }
  const auto enormous = std::numeric_limits<double>::max();
  check(cudaMemcpy(density, &enormous, sizeof(double), cudaMemcpyHostToDevice));
  require(response.apply(motion, density, delta, output, storage.stream) == cudaErrorInvalidValue,
          "overflowing finite contraction published output");
  unchanged();
  check(cudaMemcpy(density, densities.data(), sizeof(double), cudaMemcpyHostToDevice));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeGlobal));
  const auto captured = response.apply(motion, density, delta, output, storage.stream);
  cudaGraph_t graph{};
  const auto ended = cudaStreamEndCapture(storage.stream, &graph);
  if (graph) cudaGraphDestroy(graph);
  require(captured == cudaErrorStreamCaptureUnsupported && ended == cudaSuccess,
          "capture admission changed stream");
  unchanged();
  owner.remaining = 1;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "late owner revocation admitted");
  unchanged();
  owner.remaining = UINT_MAX;
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  owner.live = false;
  require(response.apply(motion, density, delta, output, storage.stream) ==
              cudaErrorInvalidResourceHandle,
          "stale owner admitted");
  unchanged();
  owner.live = true;
  check(response.prepare(shape, shells, options, reference, arena, bytes));
  check(response.apply(motion, density, delta, output, storage.stream));
  sentinel();
  require(response.prepare(shape, shells, options, reference, arena, bytes - 1) ==
                  cudaErrorInvalidValue &&
              response.apply(motion, density, delta, output, storage.stream) ==
                  cudaErrorInvalidResourceHandle,
          "failed reprepare retained old binding");
  unchanged();
  std::cout << "{\"lifecycle\":true,\"bytes\":" << receipt.workspace_bytes
            << ",\"groups\":" << receipt.groups
            << ",\"records\":" << receipt.primitive_records_per_action
            << ",\"components\":" << receipt.component_evaluations
            << ",\"projections\":" << receipt.projection_terms_bound
            << ",\"reductions\":" << receipt.reduction_terms << "}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: jk-probe source.bin output.bin");
    run(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
