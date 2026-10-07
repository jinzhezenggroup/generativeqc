// Exercise actual CUDA tile ownership, ragged spin packing and atomic publication.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_hamiltonian.cuh"

using namespace generativeqc::xtb::detail;
using namespace generativeqc::xtb::detail::cuda;
using I = std::int64_t;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Managed allocations keep this isolated test fixture small; production storage
// and launch policy are exercised through the unmodified native entry points.
struct Storage {
  std::vector<void*> allocations;
  ~Storage() {
    for (void* pointer : allocations) cudaFree(pointer);
  }
  template <class T>
  T* make(I count) {
    T* pointer = nullptr;
    check(cudaMallocManaged(&pointer, sizeof(T) * count));
    allocations.push_back(pointer);
    std::fill(pointer, pointer + count, T{});
    return pointer;
  }
  I* offsets(const std::vector<I>& values) {
    I* result = make<I>(values.size());
    std::copy(values.begin(), values.end(), result);
    return result;
  }
};

// Reuse the same deliberately imbalanced physical/spin topology for density.
int run_density(Storage& storage, const Gfn2HamiltonianDeviceBatch& topology,
                const Gfn2WavefunctionLayoutView& layout, bool spin, bool capture) {
  const I systems = topology.batch_size, n = topology.total_orbitals;
  const I sn = layout.total_spin_orbitals, sm = layout.total_spin_matrix_elements;
  const I nc = layout.total_spin_channels;
  Gfn2DensityDeviceBatch batch{
      systems,     n, topology.total_matrix_elements, systems + 1,
      systems + 1, 1, topology.batch_orbital_offsets, topology.matrix_offsets};
  auto* c = storage.make<double>(sm);
  auto* e = storage.make<double>(sn);
  auto* occ = storage.make<double>(2 * n);
  auto* active = storage.make<std::uint8_t>(systems);
  for (I i = 0; i < sm; ++i) c[i] = 0.001 * (i % 13) - 0.005;
  for (I i = 0; i < sn; ++i) e[i] = 0.01 * (i % 17) - 0.1;
  for (I i = 0; i < 2 * n; ++i) occ[i] = 0.1 * (i % 7);
  std::fill(active, active + systems, 1);
  Gfn2DensityDeviceInput input{c, sm, e, sn, occ, 2 * n, active, systems, 1};
  Gfn2DensityDeviceResults results{};
  Gfn2DensityDeviceWorkspace workspace{};
  results.plan_token = workspace.plan_token = 1;
  // Pointer/count pairs are independent allocations, exercising native alias admission.
#define BUFFER(owner, pointer, count, size)   \
  owner.pointer = storage.make<double>(size); \
  owner.count = size
  BUFFER(results, density, density_elements, sm);
  BUFFER(results, energy_weighted_density, weighted_density_elements, sm);
  BUFFER(results, band_energies, band_energy_elements, systems);
  BUFFER(results, occupation_sums, occupation_sum_elements, systems);
  BUFFER(results, density_traces, density_trace_elements, systems);
  BUFFER(results, weighted_density_traces, weighted_density_trace_elements, systems);
  BUFFER(results, channel_band_energies, channel_band_energy_elements, nc);
  BUFFER(results, channel_occupation_sums, channel_occupation_sum_elements, nc);
  BUFFER(results, channel_density_traces, channel_density_trace_elements, nc);
  BUFFER(results, channel_weighted_density_traces, channel_weighted_density_trace_elements, nc);
  BUFFER(workspace, density_scratch, density_elements, sm);
  BUFFER(workspace, weighted_density_scratch, weighted_density_elements, sm);
  BUFFER(workspace, weights, weight_elements, sn);
  BUFFER(workspace, energy_weights, energy_weight_elements, sn);
  BUFFER(workspace, band_energy_scratch, band_energy_elements, systems);
  BUFFER(workspace, occupation_sum_scratch, occupation_sum_elements, systems);
  BUFFER(workspace, density_trace_scratch, density_trace_elements, systems);
  BUFFER(workspace, weighted_density_trace_scratch, weighted_density_trace_elements, systems);
  BUFFER(workspace, channel_band_energy_scratch, channel_band_energy_elements, nc);
  BUFFER(workspace, channel_occupation_sum_scratch, channel_occupation_sum_elements, nc);
  BUFFER(workspace, channel_density_trace_scratch, channel_density_trace_elements, nc);
  BUFFER(workspace, channel_weighted_density_trace_scratch, channel_weighted_density_trace_elements,
         nc);
#undef BUFFER
  workspace.sequence_active = storage.make<std::uint32_t>(1);
  workspace.sequence_active_elements = 1;
  auto* errors = storage.make<std::uint32_t>(systems);
  auto* error = storage.make<std::uint32_t>(1);
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  auto launch = [&]() {
    check(reset_gfn2_density_device_errors_cuda(systems, errors, error, stream));
    if (spin)
      check(evaluate_gfn2_spin_density_cuda(batch, layout, input, results, workspace, errors, error,
                                            stream));
    else
      check(evaluate_gfn2_restricted_density_cuda(batch, input, results, workspace, errors, error,
                                                  stream));
  };
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (capture) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
  constexpr double sentinel = -973.375;
  for (int scenario = 0; scenario < 5; ++scenario) {
    for (double* p : {results.density, results.energy_weighted_density})
      std::fill(p, p + sm, sentinel);
    for (double* p : {results.band_energies, results.occupation_sums, results.density_traces,
                      results.weighted_density_traces})
      std::fill(p, p + systems, sentinel);
    for (double* p : {results.channel_band_energies, results.channel_occupation_sums,
                      results.channel_density_traces, results.channel_weighted_density_traces})
      std::fill(p, p + nc, sentinel);
    active[2] = scenario == 1 ? 0 : 1;
    const I bad = layout.spin_matrix_offsets[2] - 1;
    const I bad_eigen = layout.spin_orbital_offsets[2] - 1;
    const I extent_bad = batch.orbital_offsets[2] - batch.orbital_offsets[1];
    const I bad_alpha = 2 * batch.orbital_offsets[1] + extent_bad - 1;
    const I bad_beta = bad_alpha + extent_bad;
    const double saved = c[bad], saved_eigen = e[bad_eigen];
    const double saved_alpha = occ[bad_alpha], saved_beta = occ[bad_beta];
    // All input entries stay finite. Scenario 2 fails P; scenario 3 permits P
    // but fails W in the late beta channel. Both physical channels must retain
    // every published sentinel. Scenario 4 proves recovery on the same graph.
    if (scenario == 2) c[bad] = 1e200;
    if (scenario == 3) {
      c[bad] = 1e100;
      e[bad_eigen] = 1e150;
      occ[bad_alpha] = occ[bad_beta] = 0.5;
    }
    const bool failed = scenario == 2 || scenario == 3;
    if (capture)
      check(cudaGraphLaunch(executable, stream));
    else
      launch();
    check(cudaStreamSynchronize(stream));
    if ((*error != 0) != failed) return 11;
    if (failed &&
        errors[1] != static_cast<std::uint32_t>(
                         scenario == 2
                             ? Gfn2DensityDeviceError::kNonfiniteDensityArithmetic
                             : Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic))
      return 16;
    for (I s = 0; s < systems; ++s) {
      const I extent = batch.orbital_offsets[s + 1] - batch.orbital_offsets[s];
      const I count = layout.spin_channels[s];
      const bool suppressed = (!active[s] || (failed && s == 1));
      if ((errors[s] != 0) != (failed && s == 1)) return 12;
      long double system_p = 0, system_w = 0, system_band = 0, system_occ = 0;
      for (I channel = 0; channel < count; ++channel) {
        const I start = layout.spin_matrix_offsets[s] + channel * extent * extent;
        const I eigen = layout.spin_orbital_offsets[s] + channel * extent;
        const I occupation = 2 * batch.orbital_offsets[s];
        long double trace_p = 0, trace_w = 0, band = 0, weight_sum = 0;
        for (I r = 0; r < extent; ++r)
          for (I col = 0; col < extent; ++col) {
            long double p = 0, w = 0;
            if (!suppressed)
              for (I k = 0; k < extent; ++k) {
                const long double weight =
                    count == 1 ? (long double)occ[occupation + k] + occ[occupation + extent + k]
                               : occ[occupation + channel * extent + k];
                const long double term =
                    (long double)c[start + r * extent + k] * weight * c[start + col * extent + k];
                p += term;
                w += term * e[eigen + k];
                if (r == 0 && col == 0) {
                  band += weight * e[eigen + k];
                  weight_sum += weight;
                }
              }
            const I index = start + r * extent + col;
            if (!std::isfinite(results.density[index]) ||
                !std::isfinite(results.energy_weighted_density[index]) ||
                std::abs(results.density[index] - (suppressed ? sentinel : double(p))) > 3e-13 ||
                std::abs(results.energy_weighted_density[index] -
                         (suppressed ? sentinel : double(w))) > 3e-13)
              return 13;
            if (r == col) {
              trace_p += p;
              trace_w += w;
            }
          }
        const I diagnostic = layout.spin_channel_offsets[s] + channel;
        if (spin) {
          const double expected[] = {double(band), double(weight_sum), double(trace_p),
                                     double(trace_w)};
          const double actual[] = {results.channel_band_energies[diagnostic],
                                   results.channel_occupation_sums[diagnostic],
                                   results.channel_density_traces[diagnostic],
                                   results.channel_weighted_density_traces[diagnostic]};
          for (int k = 0; k < 4; ++k)
            if (!std::isfinite(actual[k]) ||
                std::abs(actual[k] - (suppressed ? sentinel : expected[k])) > 3e-12)
              return 14;
        }
        system_p += trace_p;
        system_w += trace_w;
        system_band += band;
        system_occ += weight_sum;
      }
      const double expected[] = {double(system_band), double(system_occ), double(system_p),
                                 double(system_w)};
      const double actual[] = {results.band_energies[s], results.occupation_sums[s],
                               results.density_traces[s], results.weighted_density_traces[s]};
      for (int k = 0; k < 4; ++k)
        if (!std::isfinite(actual[k]) ||
            std::abs(actual[k] - (suppressed ? sentinel : expected[k])) > 3e-12)
          return 15;
    }
    c[bad] = saved;
    e[bad_eigen] = saved_eigen;
    occ[bad_alpha] = saved_alpha;
    occ[bad_beta] = saved_beta;
  }
  // A pre-existing sequence error gates the complete method before either
  // channel's scratch can be published. Deliberately do not call reset here.
  std::fill(results.density, results.density + sm, sentinel);
  std::fill(results.energy_weighted_density, results.energy_weighted_density + sm, sentinel);
  std::fill(errors, errors + systems, 0);
  *error = static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteDensityArithmetic);
  if (spin)
    check(evaluate_gfn2_spin_density_cuda(batch, layout, input, results, workspace, errors, error,
                                          stream));
  else
    check(evaluate_gfn2_restricted_density_cuda(batch, input, results, workspace, errors, error,
                                                stream));
  check(cudaStreamSynchronize(stream));
  for (I i = 0; i < sm; ++i)
    if (results.density[i] != sentinel || results.energy_weighted_density[i] != sentinel) return 17;
  if (*error != static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteDensityArithmetic))
    return 18;
  for (I s = 0; s < systems; ++s)
    if (errors[s] != 0) return 19;
  if (executable) check(cudaGraphExecDestroy(executable));
  if (graph) check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  return 0;
}

int run(bool spin, bool capture, I largest) {
  Storage storage;
  const std::vector<I> sizes{3, largest, 17};
  const I systems = sizes.size();
  std::vector<I> offsets{0}, matrix_offsets{0}, spin_offsets{0}, spin_matrices{0}, channels{0};
  for (I system = 0; system < systems; ++system) {
    const I n = sizes[system], count = spin && system == 1 ? 2 : 1;
    offsets.push_back(offsets.back() + n);
    matrix_offsets.push_back(matrix_offsets.back() + n * n);
    spin_offsets.push_back(spin_offsets.back() + count * n);
    spin_matrices.push_back(spin_matrices.back() + count * n * n);
    channels.push_back(channels.back() + count);
  }
  const I n = offsets.back(), m = matrix_offsets.back(), sn = spin_offsets.back();
  const I sm = spin_matrices.back();
  Gfn2HamiltonianDeviceBatch batch{};
  batch.batch_size = systems;
  batch.total_atoms = batch.total_shells = batch.total_orbitals = n;
  batch.total_matrix_elements = m;
  batch.plan_token = 1;
  batch.atom_offset_count = batch.batch_shell_offset_count = batch.batch_orbital_offset_count =
      batch.matrix_offset_count = systems + 1;
  batch.atom_shell_offset_count = batch.shell_orbital_offset_count = n + 1;
  batch.shell_to_atom_count = batch.orbital_to_shell_count = batch.orbital_to_atom_count = n;
  batch.atom_offsets = storage.offsets(offsets);
  batch.batch_shell_offsets = storage.offsets(offsets);
  batch.batch_orbital_offsets = storage.offsets(offsets);
  batch.matrix_offsets = storage.offsets(matrix_offsets);
  auto* atom_shell = storage.make<I>(n + 1);
  auto* shell_orbital = storage.make<I>(n + 1);
  auto* shell_atom = storage.make<I>(n);
  auto* orbital_shell = storage.make<I>(n);
  auto* orbital_atom = storage.make<I>(n);
  for (I i = 0; i <= n; ++i) atom_shell[i] = shell_orbital[i] = i;
  for (I i = 0; i < n; ++i) shell_atom[i] = orbital_shell[i] = orbital_atom[i] = i;
  batch.atom_shell_offsets = atom_shell;
  batch.shell_orbital_offsets = shell_orbital;
  batch.shell_to_atom = shell_atom;
  batch.orbital_to_shell = orbital_shell;
  batch.orbital_to_atom = orbital_atom;

  Gfn2WavefunctionLayoutView layout{};
  layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  layout.plan_token = 1;
  layout.batch_size = systems;
  layout.total_spin_channels = channels.back();
  layout.total_spin_orbitals = sn;
  layout.total_spin_matrix_elements = sm;
  layout.total_spin_shells = sn;
  layout.total_spin_atoms = sn;
  layout.spin_channel_count = systems;
  layout.spin_channel_offset_count = layout.spin_orbital_offset_count =
      layout.spin_matrix_offset_count = layout.spin_shell_offset_count =
          layout.spin_atom_offset_count = systems + 1;
  auto* spin_channels = storage.make<std::int32_t>(systems);
  for (I i = 0; i < systems; ++i) spin_channels[i] = channels[i + 1] - channels[i];
  layout.spin_channels = spin_channels;
  layout.spin_channel_offsets = storage.offsets(channels);
  layout.spin_orbital_offsets = storage.offsets(spin_offsets);
  layout.spin_shell_offsets = storage.offsets(spin_offsets);
  layout.spin_atom_offsets = storage.offsets(spin_offsets);
  layout.spin_matrix_offsets = storage.offsets(spin_matrices);

  auto* h0 = storage.make<double>(m);
  auto* overlap = storage.make<double>(m);
  auto* dipole = storage.make<double>(3 * m);
  auto* quad = storage.make<double>(6 * m);
  auto* scalar = storage.make<double>(sn);
  auto* dp = storage.make<double>(3 * sn);
  auto* qp = storage.make<double>(6 * sn);
  for (I i = 0; i < m; ++i) {
    h0[i] = 0.125 + (i % 11) * 0.003;
    overlap[i] = 0.2 + (i % 7) * 0.01;
  }
  for (I i = 0; i < 3 * m; ++i) dipole[i] = 0.003 * (i % 13) - 0.02;
  for (I i = 0; i < 6 * m; ++i) quad[i] = 0.002 * (i % 17) - 0.01;
  for (I i = 0; i < sn; ++i) scalar[i] = 0.01 * (i % 19) - 0.13;
  for (I i = 0; i < 3 * sn; ++i) dp[i] = 0.03 * (i % 5) - 0.06;
  for (I i = 0; i < 6 * sn; ++i) qp[i] = 0.02 * (i % 7) - 0.05;
  Gfn2HamiltonianDeviceInput input{h0,     m,  overlap, m,      dipole, 3 * m,  quad, 6 * m,
                                   scalar, sn, dp,      3 * sn, qp,     6 * sn, 1};
  auto* active = storage.make<std::uint8_t>(systems);
  std::fill(active, active + systems, 1);
  auto* result = storage.make<double>(sm);
  auto* scratch = storage.make<double>(sm);
  auto* sequence = storage.make<std::uint32_t>(1);
  auto* errors = storage.make<std::uint32_t>(systems);
  auto* error = storage.make<std::uint32_t>(1);
  Gfn2HamiltonianDeviceActivity activity{active, systems, 1};
  Gfn2HamiltonianDeviceOutput output{result, sm, 1};
  Gfn2HamiltonianDeviceWorkspace workspace{scratch, sm, sequence, 1, 1};
  constexpr double sentinel = -973.375;
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  const auto launch = [&]() {
    check(reset_gfn2_hamiltonian_device_errors_cuda(systems, errors, error, stream));
    if (spin)
      check(assemble_gfn2_spin_hamiltonian_cuda(batch, layout, input, activity, output, workspace,
                                                errors, error, stream));
    else
      check(assemble_gfn2_hamiltonian_cuda(batch, input, activity, output, workspace, errors, error,
                                           stream));
  };
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (capture) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
  // Replay valid, inactive, and late-tile-NaN inputs through the same graph.
  for (int scenario = 0; scenario < 3; ++scenario) {
    std::fill(result, result + sm, sentinel);
    active[2] = scenario == 1 ? 0 : 1;
    const I bad = matrix_offsets[2] - 1;
    const double saved = h0[bad];
    if (scenario == 2) h0[bad] = std::numeric_limits<double>::quiet_NaN();
    if (capture)
      check(cudaGraphLaunch(executable, stream));
    else
      launch();
    check(cudaStreamSynchronize(stream));
    if ((*error != 0) != (scenario == 2)) return 1;
    for (I s = 0; s < systems; ++s) {
      const I extent = sizes[s], off = matrix_offsets[s], begin = spin_offsets[s];
      if ((errors[s] != 0) != (scenario == 2 && s == 1)) return 2;
      for (I channel = 0; channel < spin_channels[s]; ++channel) {
        // Independent long-double scalar oracle. Each pair owns both directions;
        // spin conversion follows the documented charge/magnetization contract.
        auto potential = [&](const double* p, I width, I atom, I component) -> long double {
          const long double charge = p[(begin + atom) * width + component];
          if (spin_channels[s] == 1) return charge;
          const long double magnet = p[(begin + extent + atom) * width + component];
          return (charge + (channel == 0 ? magnet : -magnet)) * 0.5L;
        };
        for (I r = 0; r < extent; ++r)
          for (I c = r; c < extent; ++c) {
            const I f = off + r * extent + c, rev = off + c * extent + r;
            long double shift =
                -0.5L * overlap[f] * (potential(scalar, 1, r, 0) + potential(scalar, 1, c, 0));
            for (I k = 0; k < 3; ++k)
              shift -= 0.5L * (dipole[k * m + f] * potential(dp, 3, c, k) +
                               dipole[k * m + rev] * potential(dp, 3, r, k));
            for (I k = 0; k < 6; ++k)
              shift -= 0.5L * (quad[k * m + f] * potential(qp, 6, c, k) +
                               quad[k * m + rev] * potential(qp, 6, r, k));
            if (spin_channels[s] == 2) shift *= 2;
            for (I entry : {f, rev}) {
              const I index = spin_matrices[s] + channel * extent * extent + entry - off;
              const bool suppressed = (!active[s] || (scenario == 2 && s == 1));
              const double expected = suppressed ? sentinel : double(h0[entry] + shift);
              if (!std::isfinite(result[index]) || std::abs(result[index] - expected) > 3e-13) {
                std::fprintf(stderr, "spin=%d scenario=%d system=%ld entry=%ld: %.17g != %.17g\n",
                             spin, scenario, s, entry, result[index], expected);
                return 3;
              }
            }
          }
      }
    }
    h0[bad] = saved;
  }
  if (executable) check(cudaGraphExecDestroy(executable));
  if (graph) check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  return run_density(storage, batch, layout, spin, capture);
}

int main() {
  try {
    // One tile, multiple tiles, and the cap with additional grid-stride passes.
    for (I size : {7, 193, 389})
      for (bool spin : {false, true})
        for (bool graph : {false, true})
          if (int status = run(spin, graph, size)) return status;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 4;
  }
  std::puts(
      "Hamiltonian/density: ragged spin, graph replay, inactive and failed-tile publication "
      "passed");
}
