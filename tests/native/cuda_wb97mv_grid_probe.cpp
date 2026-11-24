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
#include "dft/split_hybrid_cuda_domain.hpp"
#include "generated_split_hybrid_registry.cuh"

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
  settings = options;
  settings.functional_code = response::semilocal_family_code(response::SemilocalFamily::Pbe);
  settings.nuclear_response = true;
  settings.max_pair_evaluations = settings.pair_tile_points = 0;
  std::size_t semilocal_bytes{};
  check(response::cuda_wb97mv_grid_workspace(shape, settings, semilocal_bytes));
  require(semilocal_bytes < bytes, "semilocal workspace reserves nonlocal pairs");
  settings.exchange_scale = -1;
  reject(shape, settings);
  settings.exchange_scale = 1;
  settings.correlation_scale = std::numeric_limits<double>::quiet_NaN();
  reject(shape, settings);
  settings.correlation_scale = 1;
  settings.functional_code = response::semilocal_family_code(response::SemilocalFamily::B3lyp);
  check(response::cuda_wb97mv_grid_workspace(shape, settings, semilocal_bytes));
  settings.exchange_scale = 0.8;
  reject(shape, settings);
  settings.exchange_scale = 1;
  settings.functional_code = response::semilocal_family_code(response::SemilocalFamily::R2scan);
  check(response::cuda_wb97mv_grid_workspace(shape, settings, semilocal_bytes));
  settings.correlation_scale = 0.8;
  reject(shape, settings);
  settings.functional_code = response::generated::kM062XFunctionalCode;
  reject(shape, settings);
  settings.correlation_scale = 1;
  check(response::cuda_wb97mv_grid_workspace(shape, settings, semilocal_bytes));
  response::CudaWb97mvGridResponse grid;
  const double* potential = reinterpret_cast<const double*>(8);
  if (grid.reference_potential(potential) != cudaErrorInvalidResourceHandle || potential)
    throw std::runtime_error("unprepared potential escaped");
  require(grid.reference_energy(potential) == cudaErrorInvalidResourceHandle && !potential,
          "unprepared energy escaped");
  std::cout << "{\"contract\":true}\n";
}

void evaluate(std::string mode) {
  const bool bounded_factored = mode == "nuclear-pbe-factored";
  const bool tight_adjoint = mode == "nuclear-pbe-adjoint-tight";
  if (bounded_factored || tight_adjoint) mode = "nuclear-pbe";
  const bool tight_points = mode.ends_with("-points-tight");
  const bool cache_points = tight_points || mode.ends_with("-points");
  if (cache_points) mode.resize(mode.size() - (tight_points ? 13 : 7));
  response::CudaWb97mvGridShape shape;
  std::cin >> shape.atoms >> shape.primitives >> shape.nao >> shape.points >> shape.spins;
  // Leave room to exercise the final partial tile beyond 4096 points.
  if (!std::cin || shape.nao > 128 || shape.points > 8192 || shape.spins > 2 || shape.atoms > 128 ||
      shape.primitives > 4096)
    throw std::runtime_error("probe shape exceeds finite limits");
  response::CudaWb97mvGridOptions options;
  const bool nuclear = mode.rfind("nuclear", 0) == 0;
  const bool scaled = mode == "nuclear-lda" || mode == "nuclear-pbe" || mode == "nuclear-b3lyp" ||
                      mode == "nuclear-r2scan" || mode == "nuclear-m06-2x";
  options.nuclear_response = nuclear;
  options.replay_nuclear_maps = mode != "nuclear-serial";
  const bool test_adjoint = mode == "nuclear-pbe" && shape.spins == 1;
  if (test_adjoint) options.max_nuclear_adjoint_bytes = 64ULL << 20;
  if (tight_adjoint)
    options.max_nuclear_adjoint_bytes = 4 * shape.nao * shape.points * sizeof(double) + (128 << 10);
  if (mode == "nuclear-split-semilocal" || mode == "fixed-split-semilocal")
    options.component = response::CudaWb97mvGridComponent::semilocal_only;
  if (mode == "nuclear-split-vv10" || mode == "fixed-split-vv10")
    options.component = response::CudaWb97mvGridComponent::vv10_only;
  std::cin >> options.pair_tile_points >> options.max_pair_evaluations;
  if (mode == "nuclear-rvv10" || mode == "fixed-rvv10") {
    response::nlc::Vv10Parameters revised{response::nlc::Vv10Variant::rvv10};
    std::cin >> revised.b >> revised.c >> revised.coefficient;
    options.revised_nonlocal = revised;
    options.functional_code = response::semilocal_family_code(response::SemilocalFamily::Pbe);
    options.component = response::CudaWb97mvGridComponent::vv10_only;
  }
  if (scaled) {
    options.functional_code = response::semilocal_family_code(
        mode == "nuclear-lda"      ? response::SemilocalFamily::Lda
        : mode == "nuclear-pbe"    ? response::SemilocalFamily::Pbe
        : mode == "nuclear-r2scan" ? response::SemilocalFamily::R2scan
                                   : response::SemilocalFamily::B3lyp);
    std::cin >> options.exchange_scale >> options.correlation_scale;
  }
  if (mode == "fixed-m06-2x" || mode == "nuclear-m06-2x")
    options.functional_code = response::generated::kM062XFunctionalCode;
  if (mode == "nuclear-tight") {
    auto serial = options;
    serial.replay_nuclear_maps = false;
    check(response::cuda_wb97mv_grid_workspace(shape, serial, options.max_workspace_bytes));
  }
  if (tight_points) {
    std::size_t baseline_bytes{};
    check(response::cuda_wb97mv_grid_workspace(shape, options, baseline_bytes));
    options.max_workspace_bytes = baseline_bytes;
  }
  if (cache_points) options.max_point_linearization_bytes = 16ULL << 20;
  if (bounded_factored) {
    options.max_nuclear_affine_terms = 1ULL << 30;
    auto no_adjoint = options;
    no_adjoint.max_nuclear_adjoint_bytes = 0;
    std::size_t unusable{};
    require(response::cuda_wb97mv_grid_workspace(shape, no_adjoint, unusable) ==
                cudaErrorInvalidValue,
            "over-budget affine grid was admitted without its factored replacement");
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
  if (mode == "private-replay") {
    check(prepared);
    require(shape.nao > 1, "private replay needs an off-diagonal AO pair");
    const auto matrix_bytes = matrices * sizeof(double);
    std::vector<double> expected(matrices), replay(matrices);
    check(grid.apply(moving, output, stream));
    check(cudaMemcpy(expected.data(), output, matrix_bytes, cudaMemcpyDeviceToHost));
    direction[1] += 0.1;
    const auto* asymmetric = storage.upload(direction);
    check(grid.enqueue_private_response(asymmetric, output, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(replay.data(), output, matrix_bytes, cudaMemcpyDeviceToHost));
    require(
        std::all_of(replay.begin(), replay.end(), [](double value) { return std::isnan(value); }),
        "unchecked response did not propagate the device audit failure");
    check(grid.enqueue_private_response(moving, output, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(replay.data(), output, matrix_bytes, cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < matrices; ++index)
      require(std::abs(expected[index] - replay[index]) <= 2e-13 * (1 + std::abs(expected[index])),
              "private replay disagrees with the checked response");
    std::cout << "{\"private_replay\":true}\n";
    check(cudaStreamDestroy(other));
    check(cudaStreamDestroy(stream));
    return;
  }
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
    const auto input_right_ao = read(panel);
    auto* right_ao = storage.upload(input_right_ao);
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
    auto* fixed = scaled ? storage.allocate(matrices) : nullptr;
    if (fixed) check(grid.apply(moving, fixed, stream));
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
    std::cout << std::setprecision(17) << "{\"workspace_bytes\":" << grid.work().workspace_bytes
              << ",\"energy\":";
    array(energy, 1);
    std::cout << ",\"potential\":";
    array(potential, matrices);
    std::cout << ",\"response\":";
    array(output, matrices);
    if (fixed) {
      std::cout << ",\"fixed_response\":";
      array(fixed, matrices);
    }
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
    const auto measured_work = grid.work();
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
    auto* transaction = reinterpret_cast<int*>(storage.allocate(1));
    const auto private_mixed = [&] {
      check(grid.enqueue_private_nuclear_mixed(left_ao, mixed_ao, left_weights, mixed_weights,
                                               left_points, scalar, transaction, stream));
      check(cudaStreamSynchronize(stream));
    };
    check(cudaMemsetAsync(transaction, 0, sizeof(int), stream));
    private_mixed();
    int status{};
    double candidate{};
    check(cudaMemcpy(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
    require(status && std::isnan(candidate), "private mixed audit was lost");
    check(cudaMemcpy(left_weights, &original, sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemsetAsync(transaction, 0, sizeof(int), stream));
    private_mixed();
    check(cudaMemcpy(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
    require(!status && std::isfinite(candidate), "failed private mixed action contaminated replay");
    check(mixed());
    double checked{};
    check(cudaMemcpy(&checked, scalar, sizeof(checked), cudaMemcpyDeviceToHost));
    require(std::abs(checked - candidate) <= 2e-13 * (1 + std::abs(checked)),
            "private mixed replay disagrees with checked output");
    const auto checked_pair_count = grid.work().nuclear_response_pairs;
    const auto private_direction = [&] {
      check(cudaMemsetAsync(transaction, 0, sizeof(int), stream));
      check(grid.enqueue_private_nuclear_direction(right_ao, right_weights, right_points, moving,
                                                   output, transaction, stream));
      require(mixed() == cudaErrorInvalidResourceHandle,
              "checked action consumed an unverified right cache");
      check(cudaStreamSynchronize(stream));
      check(cudaMemcpy(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost));
    };
    check(cudaMemcpy(right_weights, &nan, sizeof(double), cudaMemcpyHostToDevice));
    private_direction();
    require(status, "private direction did not report a failed audit");
    private_mixed();
    check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
    require(std::isnan(candidate), "failed private direction escaped into a mixed result");
    check(cudaMemsetAsync(right_weights, 0, shape.points * sizeof(double), stream));
    private_direction();
    require(!status, "failed private direction contaminated valid replay");
    std::vector<double> replayed_direction(matrices), checked_direction(matrices);
    check(cudaMemcpy(replayed_direction.data(), output, matrices * sizeof(double),
                     cudaMemcpyDeviceToHost));
    double cached_mixed{};
    if (test_adjoint) {
      require(grid.admits_cartesian_mixed(), "private AO adjoint not admitted");
      private_mixed();
      check(cudaMemcpy(&cached_mixed, scalar, sizeof(cached_mixed), cudaMemcpyDeviceToHost));
      private_mixed();
      check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
      require(candidate == cached_mixed, "private replay changed the cached left cotangent");
      const auto before = grid.work().nuclear_affine_terms;
      check(grid.enqueue_private_nuclear_covector(right_ao, right_weights, right_points, moving,
                                                  transaction, stream));
      const auto covector_terms = grid.work().nuclear_affine_terms - before;
      private_mixed();
      check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
      require(candidate == cached_mixed, "covector-only preparation changed the mixed result");
      const auto before_full = grid.work().nuclear_affine_terms;
      private_direction();
      require(!status && covector_terms < grid.work().nuclear_affine_terms - before_full,
              "covector-only preparation did not remove potential work");
    }
    check(arm());
    check(cudaMemcpy(checked_direction.data(), output, matrices * sizeof(double),
                     cudaMemcpyDeviceToHost));
    require(replayed_direction == checked_direction,
            "private right replay disagrees with the checked direction");
    if (test_adjoint) {
      check(mixed());
      check(cudaMemcpy(&checked, scalar, sizeof(checked), cudaMemcpyDeviceToHost));
      require(std::abs(checked - cached_mixed) <= 2e-13 * (1 + std::abs(checked)),
              "private adjoint disagrees after rearming the right tangent");
    }
    if (tight_adjoint) {
      // Rebind nonzero signed AO and density motion after the zero-direction
      // replay. The expanded checked contraction does not use the tiled VJP.
      check(cudaMemcpyAsync(right_ao, input_right_ao.data(), panel * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      check(cudaMemcpyAsync(moving, direction.data(), matrices * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      check(arm());
      check(mixed());
      double expanded{};
      check(cudaMemcpy(&expanded, scalar, sizeof(expanded), cudaMemcpyDeviceToHost));
      require(std::abs(expanded - cached_mixed) > 1e-12, "right AO motion did not affect oracle");
      check(cudaMemsetAsync(transaction, 0, sizeof(int), stream));
      check(grid.enqueue_private_nuclear_covector(right_ao, right_weights, right_points, moving,
                                                  transaction, stream));
      for (int repeat = 0; repeat < 2; ++repeat) {
        private_mixed();
        check(cudaMemcpy(&status, transaction, sizeof(status), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&candidate, scalar, sizeof(candidate), cudaMemcpyDeviceToHost));
        require(!status && std::abs(candidate - expanded) <= 2e-13 * (1 + std::abs(expanded)),
                "bounded private adjoint disagrees with expanded contraction");
      }
      check(arm());
    }
    const auto injected_pair_count = grid.work().nuclear_response_pairs - checked_pair_count;
    sentinel();
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
              << ",\"nuclear_response_pairs\":" << work.nuclear_response_pairs - injected_pair_count
              << ",\"active_nonlocal_points\":" << work.active_nonlocal_points
              << ",\"fixed_response_pairs\":" << work.response_pairs
              << ",\"nuclear_affine_terms\":" << work.nuclear_affine_terms
              << ",\"nuclear_contractions\":" << measured_work.nuclear_contractions
              << ",\"nuclear_graph_replays\":" << measured_work.nuclear_graph_replays
              << ",\"private_mixed_replay\":true"
              << ",\"private_direction_replay\":true"
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
    if (mode == "has-m06-2x")
      std::cout << (response::kCudaM062XVariational ? "yes\n" : "no\n");
    else if (mode == "contract")
      contract();
    else
      evaluate(mode);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
