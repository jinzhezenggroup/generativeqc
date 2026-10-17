#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_grid_geometry.hpp"

namespace {
using namespace generativeqc::dft;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  cudaStream_t stream{};
  std::vector<double*> allocations;
  Storage() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Storage() {
    if (stream) cudaStreamSynchronize(stream);
    for (auto* pointer : allocations) cudaFree(pointer);
    if (stream) cudaStreamDestroy(stream);
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
std::vector<double> read(std::ifstream& input, std::size_t count) {
  require(count <= (1ULL << 26), "oversized probe fixture");
  std::vector<double> values(count);
  require(
      static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), count * sizeof(double))),
      "short probe fixture");
  return values;
}
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> values(count);
  check(cudaMemcpy(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Owner {
  Storage& storage;
  std::array<const double*, 4> immutable;
  int device{-1};
  bool live{true};
  static cudaError_t authenticate(void* context) noexcept {
    auto& owner = *static_cast<Owner*>(context);
    int device = -1;
    if (!owner.live || cudaGetDevice(&device) != cudaSuccess || device != owner.device)
      return cudaErrorInvalidResourceHandle;
    for (const auto* pointer : owner.immutable) {
      bool found = false;
      for (const auto* allocated : owner.storage.allocations) found = found || pointer == allocated;
      if (!found) return cudaErrorInvalidResourceHandle;
    }
    return cudaSuccess;
  }
};

void host_admission() {
  CudaGridGeometryOptions options;
  const CudaGridGeometryShape shape{3, 8, 4, 9, 3};
  std::size_t bytes{};
  check(cuda_grid_geometry_workspace(shape, options, bytes));
  require(bytes > 0, "empty admitted geometry arena");
  for (unsigned kind = 0; kind < 4; ++kind) {
    auto invalid = shape;
    if (kind == 0) invalid.points = 8;
    if (kind == 1) invalid.atoms = SIZE_MAX;
    if (kind == 2) invalid.primitives = SIZE_MAX;
    if (kind == 3) invalid.nao = SIZE_MAX;
    bytes = 27;
    require(
        cuda_grid_geometry_workspace(invalid, options, bytes) == cudaErrorInvalidValue && !bytes,
        "invalid/overflowed geometry shape admitted");
  }
  for (unsigned kind = 0; kind < 10; ++kind) {
    auto invalid = options;
    if (kind == 0) invalid.max_workspace_bytes = 1;
    if (kind == 1) invalid.max_ao_jet_values = 1;
    if (kind == 2) invalid.max_partition_pairs = 1;
    if (kind == 3) invalid.partition_iterations = 0;
    if (kind == 4) invalid.partition_iterations = 6;
    if (kind == 5) invalid.point_tile = 0;
    if (kind == 6) invalid.coincident_tolerance = -1;
    if (kind == 7) invalid.reference_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
    if (kind == 8) invalid.reference_relative_tolerance = -1;
    if (kind == 9) invalid.max_ao_primitive_terms = 1;
    bytes = 27;
    require(cuda_grid_geometry_workspace(shape, invalid, bytes) == cudaErrorInvalidValue && !bytes,
            "invalid geometry policy admitted");
  }
  options.max_partition_pairs = 0;
  check(cuda_grid_geometry_workspace({1, 2, 1, 3, 3}, options, bytes));
  std::cout << "{\"host_admission_without_cuda_initialization\":true}\n";
}

void run(const char* fixture, const char* output, bool reject, std::size_t point_tile,
         const std::string& topology) {
  std::ifstream input(fixture, std::ios::binary);
  std::array<std::uint64_t, 5> header{};
  require(static_cast<bool>(input.read(reinterpret_cast<char*>(header.data()), sizeof(header))),
          "missing probe fixture header");
  CudaGridGeometryShape shape{header[0], header[1], header[2], header[0] * header[3], header[3]};
  CudaGridGeometryOptions options;
  options.point_tile = point_tile;
  options.partition_iterations = static_cast<unsigned>(header[4]);
  options.coincident_tolerance = read(input, 1)[0];
  std::size_t bytes{};
  check(cuda_grid_geometry_workspace(shape, options, bytes));
  Storage storage;
  const auto basis = read(input, 3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao);
  if (!topology.empty()) {
    const auto offset = 3 * shape.atoms + 2 * shape.primitives;
    for (std::size_t index = 0; index < shape.nao; ++index)
      shape.ao_primitive_terms +=
          static_cast<std::size_t>(basis[offset + 16 * index + 2] * basis[offset + 16 * index + 3]);
    if (topology == "understated") --shape.ao_primitive_terms;
    options.max_ao_primitive_terms = 20 * shape.points * shape.ao_primitive_terms;
    check(cuda_grid_geometry_workspace(shape, options, bytes));
  }
  const auto points = read(input, 3 * shape.points);
  const auto weights = read(input, shape.points);
  const auto atomic = read(input, shape.points);
  const auto left = read(input, 3 * shape.atoms);
  const auto right = read(input, 3 * shape.atoms);
  require(input.peek() == std::char_traits<char>::eof(), "trailing probe fixture bytes");
  Owner owner{storage,
              {storage.upload(basis), storage.upload(points), storage.upload(weights),
               storage.upload(atomic)}};
  check(cudaGetDevice(&owner.device));
  CudaGridGeometryReference reference{
      owner.immutable[0], owner.immutable[1], owner.immutable[2],  owner.immutable[3],
      owner.device,       storage.stream,     Owner::authenticate, &owner};
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* left_device = storage.upload(left);
  auto* right_device = storage.upload(right);
  const auto ao_count = 12 * shape.points * shape.nao;
  const auto weight_count = 3 * shape.points;
  auto* ao_output = storage.allocate(ao_count);
  auto* weight_output = storage.allocate(weight_count);
  CudaGridGeometryResponse response;
  const double* jets{};
  require(response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
          "unprepared geometry published jets");
  require(response.prepare(shape, options, reference, arena, bytes - 1) == cudaErrorInvalidValue,
          "geometry ignored arena capacity");
  auto insufficient = options;
  insufficient.max_ao_jet_values = 1;
  std::size_t invalid_bytes = 27;
  require(
      cuda_grid_geometry_workspace(shape, insufficient, invalid_bytes) == cudaErrorInvalidValue &&
          !invalid_bytes,
      "geometry ignored AO work capacity");
  if (shape.atoms > 1) {
    insufficient = options;
    insufficient.max_partition_pairs = 0;
    require(
        cuda_grid_geometry_workspace(shape, insufficient, invalid_bytes) == cudaErrorInvalidValue &&
            !invalid_bytes,
        "geometry ignored partition work capacity");
  }
  const auto prepared = response.prepare(shape, options, reference, arena, bytes);
  if (reject) {
    require(prepared == cudaErrorInvalidValue &&
                response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
            "geometry accepted a nonsmooth or unauthenticated primal source");
    std::cout << "{\"prepare_rejected\":true}\n";
    return;
  }
  check(prepared);
  check(response.spatial_jets(jets));
  const double* point_directions{};
  require(response.point_directions(point_directions) == cudaErrorInvalidResourceHandle &&
              !point_directions,
          "geometry exposed motion before an action");
  check(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream));
  check(response.point_directions(point_directions));
  std::ofstream capture(output, std::ios::binary);
  for (const auto& values :
       {download(jets, 20 * shape.points * shape.nao), download(ao_output, ao_count),
        download(weight_output, weight_count), download(point_directions, 6 * shape.points)})
    require(static_cast<bool>(capture.write(reinterpret_cast<const char*>(values.data()),
                                            values.size() * sizeof(double))),
            "probe output write failed");
  const auto sentinel_ao = std::vector<double>(ao_count, 27);
  const auto sentinel_weights = std::vector<double>(weight_count, 27);
  const auto sentinel = [&] {
    check(cudaMemcpyAsync(ao_output, sentinel_ao.data(), ao_count * sizeof(double),
                          cudaMemcpyHostToDevice, storage.stream));
    check(cudaMemcpyAsync(weight_output, sentinel_weights.data(), weight_count * sizeof(double),
                          cudaMemcpyHostToDevice, storage.stream));
    check(cudaStreamSynchronize(storage.stream));
  };
  const auto unchanged = [&] {
    require(download(ao_output, ao_count) == sentinel_ao &&
                download(weight_output, weight_count) == sentinel_weights,
            "failed geometry action published partial output");
    require(response.point_directions(point_directions) == cudaErrorInvalidResourceHandle &&
                !point_directions,
            "failed geometry action retained an expired motion lease");
  };
  sentinel();
  require(response.mixed(left_device, right_device, ao_output, weight_output, nullptr) ==
              cudaErrorInvalidResourceHandle,
          "geometry accepted another stream");
  unchanged();
  require(response.mixed(left_device, right_device, ao_output, ao_output, storage.stream) ==
              cudaErrorInvalidValue,
          "geometry accepted aliased outputs");
  unchanged();
  auto poisoned = left;
  poisoned[0] = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpyAsync(left_device, poisoned.data(), poisoned.size() * sizeof(double),
                        cudaMemcpyHostToDevice, storage.stream));
  require(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream) ==
              cudaErrorInvalidValue,
          "geometry accepted a nonfinite direction");
  unchanged();
  check(cudaMemcpyAsync(left_device, left.data(), left.size() * sizeof(double),
                        cudaMemcpyHostToDevice, storage.stream));
  check(cudaStreamSynchronize(storage.stream));
  check(cudaStreamBeginCapture(storage.stream, cudaStreamCaptureModeGlobal));
  const auto capture_status =
      response.mixed(left_device, right_device, ao_output, weight_output, storage.stream);
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(storage.stream, &graph));
  check(cudaGraphDestroy(graph));
  require(capture_status == cudaErrorStreamCaptureUnsupported, "geometry accepted capture");
  unchanged();
  owner.live = false;
  require(response.mixed(left_device, right_device, ao_output, weight_output, storage.stream) ==
                  cudaErrorInvalidResourceHandle &&
              response.spatial_jets(jets) == cudaErrorInvalidResourceHandle && !jets,
          "geometry accepted revoked storage");
  unchanged();
  const auto& work = response.work();
  std::cout << "{\"capacity_alias_finite_capture_revocation_gates\":true,\"workspace_bytes\":"
            << work.workspace_bytes << ",\"prepared_ao_values\":" << work.prepared_ao_jet_values
            << ",\"prepared_partition_pairs\":" << work.prepared_partition_pairs
            << ",\"action_ao_values\":" << work.action_ao_jet_values
            << ",\"action_partition_pairs\":" << work.action_partition_pairs
            << ",\"ao_primitive_term_bound\":" << work.prepared_ao_primitive_term_bound << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--host-check") {
      host_admission();
      return 0;
    }
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    require(argc == 3 || argc == 4,
            "usage: cuda_grid_geometry_probe input output [reject|point-tile]");
    const std::string topology =
        argc == 4 && (std::string(argv[3]) == "topology" || std::string(argv[3]) == "understated")
            ? argv[3]
            : "";
    const auto reject =
        argc == 4 && (std::string(argv[3]) == "reject" || topology == "understated");
    const auto tile = argc == 4 && !reject && topology.empty()
                          ? std::stoull(argv[3])
                          : CudaGridGeometryOptions{}.point_tile;
    run(argv[1], argv[2], reject, tile, topology);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
