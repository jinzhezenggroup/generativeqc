/** Diagnostic only: use master's canonical launch ABI, never an ERI oracle.
 * Buckets partition canonical AO pairs by angular sum, not shell-quartet cost.
 * Unscreened FP64 full-range J/K only; no production selector consumes this ABI.
 */
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_jk_plan.hpp"

namespace {
using namespace generativeqc;
using namespace scf::cuda_execution;
using runtime::cuda_resource_check;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

struct BucketProbe {
  std::unique_ptr<scf::CudaDirectJkPlan, decltype(&scf::destroy_cuda_direct_jk_plan)> plan{
      nullptr, scf::destroy_cuda_direct_jk_plan};
  runtime::OwnedCudaBuffer<double> density, output;
  runtime::OwnedCudaBuffer<std::uint64_t> census;
  runtime::OwnedCudaBuffer<int> failure;
  std::vector<double> direct_host, replay_host;
  std::size_t matrix_elements{};
  int host_failure{};

  ~BucketProbe() {
    // Host download destinations and borrowed buffers outlive even failed work.
    if (plan) {
      int previous = 0;
      const auto previous_status = cudaGetDevice(&previous);
      (void)cudaSetDevice(plan->device_id);
      (void)cudaStreamSynchronize(plan->stream);
      if (previous_status == cudaSuccess) (void)cudaSetDevice(previous);
    }
  }
};

struct StreamFence {
  cudaStream_t stream;
  ~StreamFence() { (void)cudaStreamSynchronize(stream); }
};

void error_message(char* error, std::size_t capacity, const std::exception& failure) {
  if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
}
}  // namespace

/** Return the actual provider DSO, so Python can reject mixed native identities. */
extern "C" const char* direct_eri_bucket_native_library() noexcept {
  Dl_info identity{};
  return dladdr(reinterpret_cast<void*>(&scf::create_cuda_direct_jk_plan), &identity)
             ? identity.dli_fname
             : nullptr;
}

/** Create one geometry owner; the provider budget also bounds probe scratch.
 * Inventory: public/canonical AO counts, provider/probe bytes, driver/runtime.
 * Device zero is relative to Slurm's unmodified visible-device assignment.
 */
extern "C" int direct_eri_bucket_create(void* opaque, std::size_t provider_budget, void** output,
                                        std::uint64_t* inventory, char* device,
                                        std::size_t device_capacity, char* error,
                                        std::size_t error_capacity) noexcept {
  if (output) *output = nullptr;
  try {
    if (!opaque || !output || !inventory || !device || !device_capacity ||
        !std::getenv("SLURM_JOB_ID"))
      throw std::invalid_argument("bucket profiling requires metadata and a Slurm allocation");
    runtime::CudaDeviceScope scope(0);
    auto probe = std::make_unique<BucketProbe>();
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    scf::CudaDirectJkPlan* plan = nullptr;
    scf::CudaDirectJkDiagnostic diagnostic;
    std::string detail;
    const auto status = scf::create_cuda_direct_jk_plan(0, {raw.orbital()}, 0, 0.0, provider_budget,
                                                        &plan, diagnostic, detail);
    probe->plan.reset(plan);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    if (!plan->canonical_pairs || plan->canonical_pair_offsets.size() != 1)
      throw std::runtime_error("master provider has no single-item canonical bucket domain");
    const auto dimension = static_cast<std::size_t>(plan->canonical_batch.nbf);
    probe->matrix_elements = runtime::size_mul(dimension, dimension, "matrix overflow");
    const auto probe_bytes = runtime::size_add(
        runtime::size_mul(probe->matrix_elements, 3 * sizeof(double), "scratch overflow"),
        2 * sizeof(std::uint64_t) + sizeof(int), "scratch overflow");
    if (diagnostic.device_bytes > provider_budget ||
        probe_bytes > provider_budget - diagnostic.device_bytes)
      throw std::runtime_error("provider budget does not admit diagnostic scratch");
    probe->density.allocate(0, probe->matrix_elements, plan->stream);
    probe->output.allocate(0, 2 * probe->matrix_elements, plan->stream);
    probe->census.allocate(0, 2, plan->stream);
    probe->failure.allocate(0, 1, plan->stream);
    probe->direct_host.resize(2 * probe->matrix_elements);
    probe->replay_host.resize(2 * probe->matrix_elements);
    std::vector<double> host_density(probe->matrix_elements);
    // Include signed off-diagonal elements without borrowing converged SCF data.
    for (std::size_t row = 0; row < dimension; ++row)
      for (std::size_t column = 0; column < dimension; ++column)
        host_density[row * dimension + column] =
            (static_cast<int>(((row + 1) % 37) * ((column + 1) % 37) % 37) - 18) / 100.0;
    StreamFence fence{plan->stream};
    cuda_resource_check(cudaMemcpyAsync(probe->density.get(), host_density.data(),
                                        host_density.size() * sizeof(double),
                                        cudaMemcpyHostToDevice, plan->stream));
    cuda_resource_check(cudaStreamSynchronize(plan->stream));
    cudaDeviceProp properties{};
    int driver = 0, runtime_version = 0;
    cuda_resource_check(cudaGetDeviceProperties(&properties, 0));
    cuda_resource_check(cudaDriverGetVersion(&driver));
    cuda_resource_check(cudaRuntimeGetVersion(&runtime_version));
    char uuid[33]{};
    for (unsigned index = 0; index < 16; ++index)
      std::snprintf(uuid + 2 * index, 3, "%02x",
                    static_cast<unsigned char>(properties.uuid.bytes[index]));
    std::snprintf(device, device_capacity, "%s\n%s", uuid, properties.name);
    inventory[0] = raw.nbf();
    inventory[1] = dimension;
    inventory[2] = diagnostic.device_bytes;
    inventory[3] = probe_bytes;
    inventory[4] = driver;
    inventory[5] = runtime_version;
    *output = probe.release();
    return 0;
  } catch (const std::exception& failure) {
    error_message(error, error_capacity, failure);
    return 1;
  }
}

/** Measure a complete bucket's stand-alone build and clean J/K actions.
 * Counts: pair sizes, values, bytes, status (empty/capacity/measured/allocation), then six
 * work counters. Samples: build/direct/replay arrays, allocation/release costs,
 * maximum absolute/scaled parity errors. Observed census actions are untimed;
 * clean action timings include resets but exclude host downloads and fences.
 * Build samples include finite audit and publication fence; allocation and
 * release are charged separately. They are not complete endpoint timings.
 */
extern "C" int direct_eri_bucket_profile(void* opaque, unsigned first, unsigned second,
                                         std::size_t maximum_cache_bytes, unsigned repeats,
                                         std::uint64_t* counts, double* samples, char* error,
                                         std::size_t error_capacity) noexcept {
  try {
    if (!opaque || !counts || !samples || !repeats || repeats > 512 || first > 6 || second > first)
      throw std::invalid_argument("invalid bucket profiling arguments");
    auto& probe = *static_cast<BucketProbe*>(opaque);
    runtime::CudaDeviceScope scope(0);
    auto& plan = *probe.plan;
    StreamFence fence{plan.stream};
    const auto& offsets = plan.canonical_pair_offsets.front();
    const auto first_count = offsets[first + 1] - offsets[first];
    const auto second_count = offsets[second + 1] - offsets[second];
    const auto value_count =
        first == second
            ? runtime::size_mul(first_count, runtime::size_add(first_count, 1, "bucket overflow"),
                                "bucket overflow") /
                  2
            : runtime::size_mul(first_count, second_count, "bucket overflow");
    const auto value_bytes = runtime::size_mul(value_count, sizeof(double), "bucket overflow");
    counts[0] = first_count;
    counts[1] = second_count;
    counts[2] = value_count;
    counts[3] = value_bytes;
    counts[4] = value_count ? 1 : 0;
    if (!value_count || value_bytes > maximum_cache_bytes) return 0;
    std::size_t available = 0, total = 0;
    cuda_resource_check(cudaMemGetInfo(&available, &total));
    constexpr std::size_t reserve = 256ULL << 20;
    if (available <= reserve || value_bytes > available - reserve) return 0;
    runtime::OwnedCudaBuffer<double> values;
    const auto allocation_start = Clock::now();
    try {
      values.allocate(0, value_count, plan.stream);
    } catch (const std::bad_alloc&) {
      const auto pending = cudaGetLastError();
      if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation)
        cuda_resource_check(pending);
      cuda_resource_check(cudaStreamSynchronize(plan.stream));
      counts[4] = 3;
      return 0;
    }
    samples[3 * repeats] = seconds_since(allocation_start);
    auto* coulomb = probe.output.get();
    auto* exchange = coulomb + probe.matrix_elements;
    runtime::OwnedCudaEvent begin(0), end(0);
    const auto reset_output = [&] {
      cuda_resource_check(cudaMemsetAsync(probe.output.get(), 0,
                                          2 * probe.matrix_elements * sizeof(double), plan.stream));
    };
    const auto launch_direct = [&](bool build, std::uint64_t* census) {
      launch_canonical_jk_kernel(
          plan.stream, plan.canonical_batch, plan.canonical_cartesian, 0, first + second,
          plan.canonical_pairs, {}, offsets[first], first_count, offsets[second], second_count,
          first == second, true, !build, false, DirectCoulombRange::Full, 0.0, 0.0,
          plan.canonical_bounds, build ? nullptr : probe.density.get(), build ? nullptr : coulomb,
          build ? nullptr : exchange, census, build ? values.get() : nullptr);
      cuda_resource_check(cudaGetLastError());
    };
    const auto launch_replay = [&](std::uint64_t* census) {
      launch_resident_canonical_jk_kernel(
          plan.stream, plan.canonical_batch, 0, plan.canonical_pairs, {}, offsets[first],
          first_count, offsets[second], second_count, first == second, true, true, false,
          values.get(), probe.density.get(), coulomb, exchange, census);
      cuda_resource_check(cudaGetLastError());
    };
    const auto build = [&] {
      const auto started = Clock::now();
      cuda_resource_check(cudaMemsetAsync(probe.failure.get(), 0, sizeof(int), plan.stream));
      launch_direct(true, nullptr);
      launch_independent_jk_finite_kernel(plan.stream, values.get(), value_count,
                                          probe.failure.get());
      cuda_resource_check(cudaGetLastError());
      cuda_resource_check(cudaMemcpyAsync(&probe.host_failure, probe.failure.get(), sizeof(int),
                                          cudaMemcpyDeviceToHost, plan.stream));
      cuda_resource_check(cudaStreamSynchronize(plan.stream));
      if (probe.host_failure) throw std::runtime_error("nonfinite bucket source");
      return seconds_since(started);
    };
    const auto action = [&](bool replay) {
      begin.record(plan.stream);
      reset_output();
      if (replay)
        launch_replay(nullptr);
      else
        launch_direct(false, nullptr);
      end.record(plan.stream);
      end.synchronize();
      return end.elapsed_since(begin) / 1000.0;
    };
    const auto download = [&](std::vector<double>& host) {
      cuda_resource_check(cudaMemcpyAsync(host.data(), probe.output.get(),
                                          host.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                          plan.stream));
      cuda_resource_check(cudaStreamSynchronize(plan.stream));
    };
    (void)build();
    (void)action(false);
    (void)action(true);
    double maximum_absolute = 0.0, maximum_scaled = 0.0;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
      samples[repeat] = build();
      // Alternate action order to avoid systematically favoring the second arm.
      for (unsigned arm = 0; arm < 2; ++arm) {
        const bool replay = ((repeat + arm) % 2) != 0;
        samples[(replay ? 2 : 1) * repeats + repeat] = action(replay);
        download(replay ? probe.replay_host : probe.direct_host);
      }
      for (std::size_t index = 0; index < probe.direct_host.size(); ++index) {
        const auto reference = probe.direct_host[index], candidate = probe.replay_host[index];
        if (!std::isfinite(reference) || !std::isfinite(candidate))
          throw std::runtime_error("nonfinite bucket J/K action");
        const auto difference = std::abs(reference - candidate);
        maximum_absolute = std::max(maximum_absolute, difference);
        maximum_scaled = std::max(maximum_scaled, difference / (1.0 + std::abs(reference)));
      }
    }
    if (maximum_scaled > 2e-11) throw std::runtime_error("bucket direct/replay parity failed");
    samples[3 * repeats + 2] = maximum_absolute;
    samples[3 * repeats + 3] = maximum_scaled;
    // Keep work observers out of clean timing, and verify all three domains.
    for (unsigned operation = 0; operation < 3; ++operation) {
      reset_output();
      cuda_resource_check(
          cudaMemsetAsync(probe.census.get(), 0, 2 * sizeof(std::uint64_t), plan.stream));
      if (operation == 2)
        launch_replay(probe.census.get());
      else
        launch_direct(operation == 1, probe.census.get());
      cuda_resource_check(cudaMemcpyAsync(counts + 5 + 2 * operation, probe.census.get(),
                                          2 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                                          plan.stream));
      cuda_resource_check(cudaStreamSynchronize(plan.stream));
      if (counts[5 + 2 * operation] != value_count ||
          counts[6 + 2 * operation] != (operation == 2 ? 0 : value_count))
        throw std::runtime_error("bucket semantic work census mismatch");
    }
    const auto release_start = Clock::now();
    values.reset();
    samples[3 * repeats + 1] = seconds_since(release_start);
    counts[4] = 2;
    return 0;
  } catch (const std::exception& failure) {
    error_message(error, error_capacity, failure);
    return 1;
  }
}

/** Drain work before retiring the geometry metadata and its diagnostic scratch. */
extern "C" void direct_eri_bucket_destroy(void* opaque) noexcept {
  delete static_cast<BucketProbe*>(opaque);
}
