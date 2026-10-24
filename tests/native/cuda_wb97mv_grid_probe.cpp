#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_wb97mv_grid.hpp"

namespace {
namespace response = generativeqc::dft;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Storage {
  std::vector<void*> allocations;
  ~Storage() {
    for (auto* pointer : allocations) cudaFree(pointer);
  }
  double* allocate(std::size_t count) {
    double* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
  double* upload(std::vector<double> values) {
    auto* pointer = allocate(values.size());
    check(
        cudaMemcpy(pointer, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    return pointer;
  }
};

std::vector<double> read(std::size_t count) {
  if (count > (1ULL << 22)) throw std::runtime_error("probe input exceeds finite limit");
  std::vector<double> values(count);
  for (auto& value : values) {
    std::string token;
    if (!(std::cin >> token)) throw std::runtime_error("missing probe input");
    value = std::stod(token);
  }
  return values;
}

struct Owner {
  unsigned authentications{}, stale_after{std::numeric_limits<unsigned>::max()};
};

cudaError_t authenticate(void* context) noexcept {
  auto& owner = *static_cast<Owner*>(context);
  return ++owner.authentications > owner.stale_after ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

void array(const double* pointer, std::size_t size) {
  std::vector<double> values(size);
  if (pointer)
    check(cudaMemcpy(values.data(), pointer, size * sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void contract() {
  response::CudaWb97mvGridShape shape{1, 1, 2, 3, 2};
  response::CudaWb97mvGridOptions options;
  std::size_t bytes{};
  check(response::cuda_wb97mv_grid_workspace(shape, options, bytes));
  const auto reject = [](auto invalid, auto settings) {
    std::size_t value = 17;
    if (response::cuda_wb97mv_grid_workspace(invalid, settings, value) != cudaErrorInvalidValue ||
        value)
      throw std::runtime_error("workspace admission accepted invalid contract");
  };
  auto invalid = shape;
  invalid.nao = SIZE_MAX;
  reject(invalid, options);
  invalid = shape;
  invalid.spins = 3;
  reject(invalid, options);
  invalid = shape;
  invalid.points = 0;
  reject(invalid, options);
  auto settings = options;
  settings.max_workspace_bytes = bytes - 1;
  reject(shape, settings);
  settings = options;
  settings.pair_tile_points = 257;
  reject(shape, settings);
  settings = options;
  settings.symmetry_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
  reject(shape, settings);
  settings = options;
  settings.nuclear_response = true;
  check(response::cuda_wb97mv_grid_workspace(shape, settings, bytes));
  settings.max_nuclear_affine_terms = 1;
  reject(shape, settings);
  response::CudaWb97mvGridResponse grid;
  const double* potential = reinterpret_cast<const double*>(8);
  if (grid.reference_potential(potential) != cudaErrorInvalidResourceHandle || potential)
    throw std::runtime_error("unprepared potential escaped");
  require(grid.reference_energy(potential) == cudaErrorInvalidResourceHandle && !potential,
          "unprepared energy escaped");
  std::cout << "{\"contract\":true}\n";
}

void evaluate(const std::string& mode) {
  response::CudaWb97mvGridShape shape;
  std::cin >> shape.atoms >> shape.primitives >> shape.nao >> shape.points >> shape.spins;
  if (!std::cin || shape.nao > 128 || shape.points > 4096 || shape.spins > 2 || shape.atoms > 128 ||
      shape.primitives > 4096)
    throw std::runtime_error("probe shape exceeds finite limits");
  response::CudaWb97mvGridOptions options;
  const bool nuclear = mode.rfind("nuclear", 0) == 0;
  options.nuclear_response = nuclear;
  options.replay_nuclear_maps = mode != "nuclear-serial";
  std::cin >> options.pair_tile_points >> options.max_pair_evaluations;
  if (mode == "nuclear-tight") {
    auto serial = options;
    serial.replay_nuclear_maps = false;
    check(response::cuda_wb97mv_grid_workspace(shape, serial, options.max_workspace_bytes));
  }
  std::size_t bytes{};
  check(response::cuda_wb97mv_grid_workspace(shape, options, bytes));
  cudaStream_t stream{}, other{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  Storage storage;
  auto* basis = storage.upload(read(3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao));
  auto* points = storage.upload(read(3 * shape.points));
  auto* weights = storage.upload(read(shape.points));
  const auto matrices = shape.spins * shape.nao * shape.nao;
  auto density = read(matrices);
  if (mode == "bad-reference") density[0] = -10;
  auto* base = storage.upload(density);
  auto direction = read(matrices);
  if (mode == "nan-direction") direction[0] = std::numeric_limits<double>::quiet_NaN();
  if (mode == "overflow-direction")
    std::fill(direction.begin(), direction.end(), std::numeric_limits<double>::max());
  if (mode == "asymmetric-direction") direction[1] += 0.1;
  auto* moving = storage.upload(direction);
  auto* output = storage.upload(std::vector<double>(matrices, 91.25));
  auto* workspace = storage.allocate(bytes / sizeof(double) + 1);
  Owner owner;
  if (mode == "stale-entry") owner.stale_after = 0;
  int device = -1;
  check(cudaGetDevice(&device));
  response::CudaWb97mvGridReference reference{basis,  points, weights,      base,
                                              device, stream, authenticate, &owner};
  if (mode == "wrong-device") reference.device_id = device + 1;
  if (mode == "alias-prepare") workspace = base;
  response::CudaWb97mvGridResponse grid;
  if (mode == "capture-prepare")
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const auto prepared =
      grid.prepare(shape, options, reference, workspace, mode == "short-arena" ? bytes - 1 : bytes);
  if (mode == "capture-prepare") {
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphDestroy(graph));
  }
  const double* potential{};
  if (prepared == cudaSuccess) check(grid.reference_potential(potential));
  if (nuclear) {
    check(prepared);
    const auto panel = 4 * shape.points * shape.nao;
    auto* right_ao = storage.upload(read(panel));
    auto* left_ao = storage.upload(read(panel));
    auto* mixed_ao = storage.upload(read(panel));
    auto* right_weights = storage.upload(read(shape.points));
    auto* left_weights = storage.upload(read(shape.points));
    auto* mixed_weights = storage.upload(read(shape.points));
    auto* right_points = storage.upload(read(3 * shape.points));
    auto* left_points = storage.upload(read(3 * shape.points));
    auto* scalar = storage.upload({91.25});
    const double* energy{};
    check(grid.reference_energy(energy));
    const auto mixed = [&] {
      return grid.nuclear_mixed(left_ao, mixed_ao, left_weights, mixed_weights, left_points, scalar,
                                stream);
    };
    const auto arm = [&] {
      return grid.prepare_nuclear_direction(right_ao, right_weights, right_points, moving, output,
                                            stream);
    };
    require(mixed() == cudaErrorInvalidResourceHandle, "unarmed nuclear cache accepted");
    check(arm());
    check(mixed());
    std::cout << std::setprecision(17) << "{\"energy\":";
    array(energy, 1);
    std::cout << ",\"potential\":";
    array(potential, matrices);
    std::cout << ",\"response\":";
    array(output, matrices);
    std::cout << ",\"mixed\":";
    array(scalar, 1);
    const auto pair_work = grid.work().nuclear_response_pairs;
    // A replay must not retain the caller's previous left-input pointers.
    const auto relocate = [&](double* source, std::size_t count) {
      auto* replacement = storage.allocate(count);
      check(cudaMemcpyAsync(replacement, source, count * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
      check(cudaMemsetAsync(source, 0, count * sizeof(double), stream));
      return replacement;
    };
    left_ao = relocate(left_ao, panel);
    mixed_ao = relocate(mixed_ao, panel);
    left_weights = relocate(left_weights, shape.points);
    mixed_weights = relocate(mixed_weights, shape.points);
    left_points = relocate(left_points, 3 * shape.points);
    check(cudaMemsetAsync(right_ao, 0, panel * sizeof(double), stream));
    check(cudaMemsetAsync(moving, 0, matrices * sizeof(double), stream));
    check(cudaMemsetAsync(right_weights, 0, shape.points * sizeof(double), stream));
    check(cudaMemsetAsync(right_points, 0, 3 * shape.points * sizeof(double), stream));
    check(mixed());
    std::cout << ",\"retained_mixed\":";
    array(scalar, 1);
    require(grid.work().nuclear_response_pairs == pair_work,
            "left nuclear contraction repeated the VV10 pairs");
    const auto sentinel = [&] {
      const std::vector<double> values(matrices, 91.25);
      const double value = 91.25;
      check(cudaMemcpyAsync(output, values.data(), matrices * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      check(cudaMemcpyAsync(scalar, &value, sizeof(double), cudaMemcpyHostToDevice, stream));
      check(cudaStreamSynchronize(stream));
    };
    const auto unchanged = [&] {
      std::vector<double> values(matrices);
      double value{};
      check(cudaMemcpy(values.data(), output, matrices * sizeof(double), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(&value, scalar, sizeof(double), cudaMemcpyDeviceToHost));
      require(value == 91.25 && values == std::vector<double>(matrices, 91.25),
              "failed nuclear action published output");
    };
    sentinel();
    require(grid.nuclear_mixed(left_ao, mixed_ao, left_weights, mixed_weights, left_points, base,
                               stream) == cudaErrorInvalidValue,
            "nuclear contraction accepted aliased density output");
    unchanged();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double original{};
    check(cudaMemcpy(&original, left_weights, sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(left_weights, &nan, sizeof(double), cudaMemcpyHostToDevice));
    require(mixed() == cudaErrorInvalidValue, "nuclear contraction accepted nonfinite motion");
    unchanged();
    check(cudaMemcpy(left_weights, &original, sizeof(double), cudaMemcpyHostToDevice));
    require(grid.prepare_nuclear_direction(right_ao, right_weights, right_points, moving, moving,
                                           stream) == cudaErrorInvalidValue &&
                mixed() == cudaErrorInvalidResourceHandle,
            "aliased direction retained nuclear cache");
    unchanged();
    require(grid.prepare_nuclear_direction(right_ao, right_weights, right_points, moving, output,
                                           other) == cudaErrorInvalidResourceHandle,
            "nuclear direction accepted another stream");
    require(mixed() == cudaErrorInvalidResourceHandle, "failed direction retained cache");
    unchanged();
    check(arm());
    sentinel();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const auto captured = arm();
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphDestroy(graph));
    require(
        captured == cudaErrorStreamCaptureUnsupported && mixed() == cudaErrorInvalidResourceHandle,
        "capture direction retained nuclear cache");
    unchanged();
    check(arm());
    sentinel();
    require(grid.apply(nullptr, output, stream) == cudaErrorInvalidValue &&
                mixed() == cudaErrorInvalidResourceHandle,
            "ordinary apply attempt retained nuclear cache");
    unchanged();
    check(cudaMemcpy(right_weights, &nan, sizeof(double), cudaMemcpyHostToDevice));
    require(arm() == cudaErrorInvalidValue && mixed() == cudaErrorInvalidResourceHandle,
            "nonfinite direction retained nuclear cache");
    unchanged();
    check(cudaMemsetAsync(right_weights, 0, shape.points * sizeof(double), stream));
    owner.stale_after = owner.authentications + 1;
    require(arm() == cudaErrorInvalidResourceHandle && mixed() == cudaErrorInvalidResourceHandle,
            "revoked nuclear direction published output");
    unchanged();
    const auto& work = grid.work();
    std::cout << ",\"nuclear_reference_pairs\":" << work.nuclear_reference_pairs
              << ",\"nuclear_response_pairs\":" << work.nuclear_response_pairs
              << ",\"nuclear_affine_terms\":" << work.nuclear_affine_terms
              << ",\"nuclear_contractions\":" << work.nuclear_contractions
              << ",\"nuclear_graph_replays\":" << work.nuclear_graph_replays
              << ",\"nuclear_lifecycle\":true}\n";
    check(cudaStreamDestroy(other));
    check(cudaStreamDestroy(stream));
    return;
  }
  if (mode == "stale-action") owner.stale_after = owner.authentications;
  if (mode == "stale-publish") owner.stale_after = owner.authentications + 1;
  auto* target = mode == "alias-output" ? base : output;
  if (mode == "capture-action")
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const auto applied = prepared == cudaSuccess
                           ? grid.apply(moving, target, mode == "wrong-stream" ? other : stream)
                           : cudaErrorInvalidResourceHandle;
  if (mode == "capture-action") {
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphDestroy(graph));
  }
  const auto work = grid.work();
  std::cout << std::setprecision(17) << "{\"prepared\":" << static_cast<int>(prepared)
            << ",\"applied\":" << static_cast<int>(applied)
            << ",\"workspace\":" << work.workspace_bytes
            << ",\"active\":" << work.active_nonlocal_points
            << ",\"reference_pairs\":" << work.reference_pairs
            << ",\"response_pairs\":" << work.response_pairs
            << ",\"actions\":" << work.response_actions << ",\"potential\":";
  array(potential, matrices);
  std::cout << ",\"response\":";
  array(output, matrices);
  std::cout << "}\n";
  check(cudaStreamDestroy(other));
  check(cudaStreamDestroy(stream));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "normal";
    if (mode == "contract")
      contract();
    else
      evaluate(mode);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
