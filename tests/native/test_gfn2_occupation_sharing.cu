// Exact restricted-spin task reuse against separately solved duplicate spectra.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_occupations.cuh"

using namespace generativeqc::xtb::detail;
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
    check(cudaMallocManaged(&result, size * sizeof(T)));
    owners.push_back(result);
    std::fill(result, result + size, T{});
    return result;
  }
};
struct Case {
  I count;
  double alpha, beta, temperature;
  bool degenerate = false;
  std::uint8_t active = 1;
  bool invalid_spectrum = false;
};
struct Output {
  std::vector<double> values;
  std::vector<std::uint32_t> errors;
  std::uint32_t error;
};

// Two explicit identical spectra exercise the unchanged two-solve fallback.
// Mixed ragged cases prevent one system's shared state from masking its peers.
Output run(const std::vector<Case>& cases, bool duplicate_spectra, bool graph) {
  Storage s;
  const I systems = cases.size();
  I n = 0;
  for (const auto& c : cases) n += c.count;
  auto* offsets = s.make<I>(systems + 1);
  auto* channels = s.make<std::int32_t>(systems);
  auto* channel_offsets = s.make<I>(systems + 1);
  auto* spin_offsets = s.make<I>(systems + 1);
  auto* populations = s.make<double>(2 * systems);
  auto* temperatures = s.make<double>(systems);
  auto* active = s.make<std::uint8_t>(systems);
  for (I i = 0; i < systems; ++i) {
    channels[i] = duplicate_spectra || i % 3 == 1 ? 2 : 1;
    offsets[i + 1] = offsets[i] + cases[i].count;
    channel_offsets[i + 1] = channel_offsets[i] + channels[i];
    spin_offsets[i + 1] = spin_offsets[i] + cases[i].count * channels[i];
    populations[2 * i] = cases[i].alpha;
    populations[2 * i + 1] = cases[i].beta;
    temperatures[i] = cases[i].temperature;
    active[i] = cases[i].active;
  }
  const I sn = spin_offsets[systems];
  auto* spectra = s.make<double>(sn);
  for (I i = 0; i < systems; ++i) {
    for (int spin = 0; spin < channels[i]; ++spin) {
      for (I j = 0; j < cases[i].count; ++j) {
        double value = cases[i].degenerate ? 0.25 : 0.003 * j - 0.4;
        // Mixed unrestricted members retain distinct spectra even when their
        // populations match. Their second solve must never be replaced.
        if (i % 3 == 1 && spin == 1) value = 1.3 * value + 0.031;
        spectra[spin_offsets[i] + spin * cases[i].count + j] = value;
      }
    }
    if (cases[i].invalid_spectrum)
      spectra[spin_offsets[i]] = std::numeric_limits<double>::quiet_NaN();
  }
  Gfn2OccupationsDeviceBatch batch{systems, n,       systems + 1, 2 * systems,  systems, systems,
                                   1,       offsets, populations, temperatures, active};
  Gfn2WavefunctionLayoutView layout{};
  layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  layout.plan_token = 1;
  layout.batch_size = systems;
  layout.total_spin_channels = channel_offsets[systems];
  layout.total_spin_orbitals = sn;
  layout.spin_channels = channels;
  layout.spin_channel_count = systems;
  layout.spin_channel_offsets = channel_offsets;
  layout.spin_channel_offset_count = systems + 1;
  layout.spin_orbital_offsets = spin_offsets;
  layout.spin_orbital_offset_count = systems + 1;
  Gfn2OccupationsDeviceResults output{};
  Gfn2OccupationsDeviceWorkspace work{};
  output.plan_token = work.plan_token = 1;
#define BUFFER(owner, pointer, count, size) \
  owner.pointer = s.make<double>(size);     \
  owner.count = size
  BUFFER(output, occupations, occupation_elements, 2 * n);
  BUFFER(output, chemical_potentials, chemical_potential_elements, 2 * systems);
  BUFFER(output, electron_sums, electron_sum_elements, 2 * systems);
  BUFFER(output, entropies, entropy_elements, systems);
  BUFFER(work, occupation_scratch, occupation_elements, 2 * n);
  BUFFER(work, chemical_potential_scratch, chemical_potential_elements, 2 * systems);
  BUFFER(work, electron_sum_scratch, electron_sum_elements, 2 * systems);
  BUFFER(work, entropy_scratch, entropy_elements, systems);
#undef BUFFER
  work.sequence_active = s.make<std::uint32_t>(1);
  work.sequence_active_elements = 1;
  auto* errors = s.make<std::uint32_t>(systems);
  auto* error = s.make<std::uint32_t>(1);
  std::fill(output.occupations, output.occupations + 2 * n, -97.0);
  std::fill(output.chemical_potentials, output.chemical_potentials + 2 * systems, -97.0);
  std::fill(output.electron_sums, output.electron_sums + 2 * systems, -97.0);
  std::fill(output.entropies, output.entropies + systems, -97.0);
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  const auto launch = [&] {
    check(reset_gfn2_occupations_device_errors_cuda(systems, errors, error, stream));
    check(evaluate_gfn2_occupations_cuda(batch, layout, spectra, sn, output, work, errors, error,
                                         stream));
  };
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &captured));
    check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(captured));
  } else {
    launch();
    check(cudaStreamSynchronize(stream));
  }
  check(cudaStreamDestroy(stream));
  for (I i = 0; i < systems; ++i) {
    const auto& c = cases[i];
    if (c.active == 0 || errors[i] != 0) {
      require(output.occupations[2 * offsets[i]] == -97.0 && output.entropies[i] == -97.0,
              "inactive/failed system published output");
      continue;
    }
    for (int spin = 0; spin < 2; ++spin) {
      const double target = populations[2 * i + spin];
      require(std::abs(output.electron_sums[2 * i + spin] - target) < 1e-10,
              "published electron count differs from independent target");
      if (c.degenerate && c.temperature > 0) {
        for (I j = 0; j < c.count; ++j)
          require(std::abs(output.occupations[2 * offsets[i] + spin * c.count + j] -
                           target / c.count) < 1e-12,
                  "degenerate spectrum differs from uniform analytic occupations");
      }
    }
  }
  Output result;
  result.values.assign(output.occupations, output.occupations + 2 * n);
  result.values.insert(result.values.end(), output.chemical_potentials,
                       output.chemical_potentials + 2 * systems);
  result.values.insert(result.values.end(), output.electron_sums,
                       output.electron_sums + 2 * systems);
  result.values.insert(result.values.end(), output.entropies, output.entropies + systems);
  result.errors.assign(errors, errors + systems);
  result.error = *error;
  return result;
}
int main() {
  try {
    std::vector<Case> cases;
    for (I n : {1, 2, 7, 63, 64, 65, 192, 384}) {
      for (double t : {0.0, 0.00095}) {
        cases.push_back({n, 0, 0, t});
        cases.push_back({n, double(n), double(n), t});
        cases.push_back({n, 0.4 * n, 0.4 * n, t});
        cases.push_back({n, 0.4 * n, 0.3 * n, t});
        cases.push_back({n, 0.4 * n, 0.4 * n, t, true});
      }
    }
    for (bool graph : {false, true}) {
      for (bool failures : {false, true}) {
        auto probes = cases;
        if (failures) {
          probes[0].beta = std::numeric_limits<double>::quiet_NaN();
          probes[4].invalid_spectrum = true;
          probes[7].active = 0;
        }
        const auto shared = run(probes, false, graph), separate = run(probes, true, graph);
        require(shared.errors == separate.errors && shared.error == separate.error,
                "sharing changed failure classification");
        require(failures || shared.error == 0, "valid occupation fixture failed");
        require(shared.values.size() == separate.values.size() &&
                    std::memcmp(shared.values.data(), separate.values.data(),
                                shared.values.size() * sizeof(double)) == 0,
                "sharing changed binary64 occupations or diagnostics");
      }
    }
    std::puts(
        "occupation sharing: ragged/spin, serial/cooperative, degeneracy, graph and failures "
        "passed");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
