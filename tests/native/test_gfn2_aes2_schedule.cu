// Compare independent-peer tiles against the retained ordered fused traversal.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_aes2.cuh"
#include "generated_gfn2_aes2_native.cuh"

using namespace generativeqc::xtb::detail::cuda;
using I = std::int64_t;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Storage {
  std::vector<void*> owners;
  ~Storage() {
    for (void* p : owners) cudaFree(p);
  }
  template <class T>
  T* make(I size) {
    T* result = nullptr;
    check(cudaMallocManaged(&result, std::max(I{1}, size) * sizeof(T)));
    owners.push_back(result);
    std::fill(result, result + size, T{});
    return result;
  }
};
enum class Operation { potential, vjp, scc };
enum class Fault { none, multipole, cache, overflow, mismatch, seed, inactive, closed };
struct Output {
  std::vector<double> values;
  std::vector<std::uint32_t> errors;
  std::uint32_t error;
};

void compare(const Output& tiled, const Output& fused, Operation op) {
  for (std::size_t i = 0; i < tiled.values.size(); ++i) {
    const double actual = tiled.values[i], expected = fused.values[i];
    if (op != Operation::vjp || std::isnan(expected)) {
      require(std::memcmp(&actual, &expected, sizeof(double)) == 0,
              "schedule changed exact potential or preserved seed");
    } else {
      // Materializing each generated pair VJP changes NVCC's FMA contraction
      // boundary. An isolated --fmad=false diagnostic is bitwise equal; normal
      // production flags must stay within a tight binary64 roundoff gate.
      const double limit =
          8 * std::numeric_limits<double>::epsilon() * std::max(0.125, std::abs(expected));
      require(std::abs(actual - expected) <= limit, "VJP schedule exceeded binary64 roundoff gate");
    }
  }
}

// Padding with independent single atoms selects the fused policy without
// changing any original system's geometry, multipoles, pair slice or science.
Output run(std::vector<I> sizes, bool padded, bool graph, Operation op, Fault fault) {
  const I original_systems = sizes.size();
  const I original_atoms = std::accumulate(sizes.begin(), sizes.end(), I{0});
  const I bad = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
  if (padded) {
    I atoms = original_atoms;
    while (generativeqc::xtb::generated::gfn2_aes2_atom_tiles(atoms, sizes.size()) != 1) {
      sizes.push_back(1);
      ++atoms;
    }
  }
  const I systems = sizes.size();
  const I n = std::accumulate(sizes.begin(), sizes.end(), I{0});
  const I pairs = std::accumulate(sizes.begin(), sizes.end(), I{0},
                                  [](I sum, I count) { return sum + count * (count - 1) / 2; });
  Storage s;
  auto* offsets = s.make<I>(systems + 1);
  auto* pair_offsets = s.make<I>(systems + 1);
  for (I i = 0; i < systems; ++i) {
    offsets[i + 1] = offsets[i] + sizes[i];
    pair_offsets[i + 1] = pair_offsets[i] + sizes[i] * (sizes[i] - 1) / 2;
  }
  Gfn2AES2DeviceBatch batch{};
  batch.batch_size = systems;
  batch.total_atoms = n;
  batch.total_pairs = pairs;
  batch.plan_token = 7;
  batch.atom_offsets = offsets;
  batch.pair_offsets = pair_offsets;
  batch.atom_offset_count = batch.pair_offset_count = systems + 1;
  auto* dipole_kernel = s.make<double>(n);
  auto* quadrupole_kernel = s.make<double>(n);
  auto* radius = s.make<double>(n);
  auto* valence = s.make<double>(n);
  batch.dipole_kernel = dipole_kernel;
  batch.quadrupole_kernel = quadrupole_kernel;
  batch.multipole_radius = radius;
  batch.multipole_valence_cn = valence;
  batch.dipole_kernel_count = batch.quadrupole_kernel_count = n;
  batch.multipole_radius_count = batch.multipole_valence_cn_count = n;
  auto* positions = s.make<double>(3 * n);
  auto* cn = s.make<double>(n);
  auto* charges = s.make<double>(n);
  auto* dipoles = s.make<double>(3 * n);
  auto* quadrupoles = s.make<double>(6 * n);
  for (I i = 0; i < n; ++i) {
    dipole_kernel[i] = 0.13;
    quadrupole_kernel[i] = 0.07;
    radius[i] = 2.2;
    valence[i] = 1.2;
    positions[3 * i] = 1.2 * (i % 9);
    positions[3 * i + 1] = 1.3 * ((i / 9) % 9);
    positions[3 * i + 2] = 1.4 * (i / 81);
    cn[i] = 1.1 + 0.01 * (i % 7);
    charges[i] = 0.03 * (i % 5 - 2);
    for (int j = 0; j < 3; ++j) dipoles[3 * i + j] = 0.02 * ((i + j) % 5 - 2);
    for (int j = 0; j < 6; ++j) quadrupoles[6 * i + j] = 0.01 * ((i + j) % 7 - 3);
  }
  Gfn2AES2DeviceCache cache{s.make<double>(5 * pairs), 5 * pairs, 11, 7};
  Gfn2AES2DeviceWorkspace work{};
#define BUFFER(pointer, count, size)   \
  work.pointer = s.make<double>(size); \
  work.count = size
  BUFFER(pair_scratch, pair_elements, 5 * pairs);
  BUFFER(potential_scratch, potential_elements, 10 * n);
  BUFFER(gradient_scratch, gradient_elements, 3 * n);
  BUFFER(coordination_scratch, coordination_elements, n);
#undef BUFFER
  work.scc_peer_error_scratch = s.make<std::uint32_t>(1);
  work.scc_peer_error_elements = 1;
  auto* errors = s.make<std::uint32_t>(systems);
  auto* error = s.make<std::uint32_t>(1);
  auto* active = s.make<std::uint8_t>(systems);
  std::fill(active, active + systems, 1);
  auto* sequence = s.make<std::uint32_t>(1);
  *sequence = 1;
  Gfn2SccIterationDeviceActivity activity{active, sequence, systems, 1, 7};
  auto* charge_out = s.make<double>(n);
  auto* dipole_out = s.make<double>(3 * n);
  auto* quadrupole_out = s.make<double>(6 * n);
  auto* gradient_out = s.make<double>(3 * n);
  auto* cn_out = s.make<double>(n);
  // Binary-exact, nonzero seeds expose accidental overwrite or extra replay.
  constexpr double sentinel = -0.125;
  auto initialize = [&] {
    std::fill(charge_out, charge_out + n, sentinel);
    std::fill(dipole_out, dipole_out + 3 * n, sentinel);
    std::fill(quadrupole_out, quadrupole_out + 6 * n, sentinel);
    std::fill(gradient_out, gradient_out + 3 * n, sentinel);
    std::fill(cn_out, cn_out + n, sentinel);
  };
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  check(reset_gfn2_aes2_device_errors_cuda(systems, errors, error, stream));
  check(update_gfn2_aes2_geometry_cache_cuda(batch, positions, cn, cache, work, errors, error,
                                             stream));
  check(cudaStreamSynchronize(stream));
  require(*error == 0, "valid geometry-cache fixture failed");
  const I last = offsets[bad + 1] - 1;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (fault == Fault::multipole) charges[last] = nan;
  if (fault == Fault::cache) cache.pair_data[5 * pair_offsets[bad + 1] - 1] = nan;
  if (fault == Fault::mismatch) cache.pair_data[5 * pair_offsets[bad + 1] - 1] *= 1.01;
  if (fault == Fault::overflow) {
    dipole_kernel[last] = 1e308;
    for (I i = offsets[bad]; i <= last; ++i) {
      charges[i] = 1e308;
      for (int j = 0; j < 3; ++j) dipoles[3 * i + j] = 1e308;
    }
  }
  if (fault == Fault::inactive) {
    active[bad] = 0;
    charges[last] = nan;  // Inactive data must not become a published failure.
  }
  if (fault == Fault::closed) *sequence = 0;
  auto launch = [&] {
    check(reset_gfn2_aes2_device_errors_cuda(systems, errors, error, stream));
    if (op == Operation::vjp) {
      check(add_gfn2_aes2_vjp_cuda(batch, cache, positions, cn, 11, charges, dipoles, quadrupoles,
                                   gradient_out, cn_out, work, errors, error, stream));
    } else if (op == Operation::scc) {
      check(evaluate_gfn2_aes2_scc_potential_cuda(batch, cache, 11, activity, charges, dipoles,
                                                  quadrupoles, charge_out, dipole_out,
                                                  quadrupole_out, work, errors, error, stream));
    } else {
      check(evaluate_gfn2_aes2_potential_cuda(batch, cache, charges, dipoles, quadrupoles,
                                              charge_out, dipole_out, quadrupole_out, work, errors,
                                              error, stream));
    }
  };
  initialize();
  if (fault == Fault::seed) gradient_out[3 * last + 2] = nan;
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &captured));
    check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
      initialize();
      if (fault == Fault::seed) gradient_out[3 * last + 2] = nan;
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(captured));
  } else {
    launch();
    check(cudaStreamSynchronize(stream));
  }
  Output result;
  for (I i = 0; i < original_atoms; ++i) {
    if (op == Operation::vjp) {
      for (int j = 0; j < 3; ++j) result.values.push_back(gradient_out[3 * i + j]);
      result.values.push_back(cn_out[i]);
    } else {
      result.values.push_back(charge_out[i]);
      for (int j = 0; j < 3; ++j) result.values.push_back(dipole_out[3 * i + j]);
      for (int j = 0; j < 6; ++j) result.values.push_back(quadrupole_out[6 * i + j]);
    }
  }
  result.errors.assign(errors, errors + original_systems);
  result.error = *error;
  const bool gated = fault == Fault::inactive || fault == Fault::closed;
  std::uint32_t expected = 0;
  if (fault == Fault::multipole) expected = 8;
  if (fault == Fault::cache) expected = 9;
  if (fault == Fault::overflow) expected = op == Operation::vjp ? 15 : 10;
  if (fault == Fault::mismatch) expected = 13;
  if (fault == Fault::seed) expected = 14;
  require(result.errors[bad] == expected, "wrong semantic error for original failing system");
  require(result.error == (op == Operation::scc ? 0 : expected), "wrong aggregate/plan error");
  const I width = op == Operation::vjp ? 4 : 10;
  for (I system = 0; system < original_systems; ++system) {
    if (system != bad) require(result.errors[system] == 0, "healthy peer acquired error");
    if ((system == bad && (expected || gated)) || fault == Fault::closed) {
      for (I atom = offsets[system]; atom < offsets[system + 1]; ++atom) {
        for (I component = 0; component < width; ++component) {
          const double value = result.values[width * atom + component];
          if (fault == Fault::seed && atom == last && component == 2)
            require(std::isnan(value), "invalid seed was overwritten");
          else
            require(value == sentinel, "failed/inactive system published partial output");
        }
      }
    }
  }
  check(cudaStreamDestroy(stream));
  return result;
}

int main() {
  try {
    for (Operation op : {Operation::potential, Operation::vjp, Operation::scc}) {
      for (bool graph : {false, true}) {
        for (const std::vector<I>& sizes : {std::vector<I>{31},
                                            {32},
                                            {33},
                                            {127},
                                            {128},
                                            {129},
                                            {255},
                                            {256},
                                            {257},
                                            {3, 193, 17},
                                            {1, 513, 2}}) {
          const auto tiled = run(sizes, false, graph, op, Fault::none);
          const auto fused = run(sizes, true, graph, op, Fault::none);
          require(tiled.errors == fused.errors, "schedule changed success diagnostics");
          compare(tiled, fused, op);
        }
        for (Fault fault : {Fault::multipole, Fault::cache, Fault::overflow, Fault::mismatch,
                            Fault::seed, Fault::inactive, Fault::closed}) {
          if ((fault == Fault::mismatch || fault == Fault::seed) && op != Operation::vjp) continue;
          if ((fault == Fault::inactive || fault == Fault::closed) && op != Operation::scc)
            continue;
          const std::vector<I> sizes{3, 193, 17};
          const auto tiled = run(sizes, false, graph, op, fault);
          const auto fused = run(sizes, true, graph, op, fault);
          const auto healthy = run(sizes, false, graph, op, Fault::none);
          require(tiled.errors == fused.errors, "schedule changed failure diagnostics");
          compare(tiled, fused, op);
          if (fault != Fault::closed) {
            const I width = op == Operation::vjp ? 4 : 10;
            for (I atom : {I{0}, I{2}, I{196}, I{212}})
              require(
                  std::memcmp(tiled.values.data() + width * atom,
                              healthy.values.data() + width * atom, width * sizeof(double)) == 0,
                  "failing member changed a healthy peer");
          }
        }
      }
    }
    std::puts("AES2 schedules: ragged, tails, grid strides, Graph replay and failure gates passed");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "AES2 schedule failure: %s\n", e.what());
    return 1;
  }
}
