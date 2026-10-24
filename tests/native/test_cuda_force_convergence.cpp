/** Physical stationarity must survive tiny DIIS steps, localized spin errors,
 * nonfinite entries and inactive neighbors. Run only in a Slurm allocation. */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/direct_density_bounds.hpp"
#include "scf/cuda/scf_convergence_kernels.hpp"
#include "scf/cuda/scf_incremental_policy.cuh"
#include "scf/cuda/scf_matrix_kernels.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* data{};
  std::size_t size{};
  explicit Device(const std::vector<T>& values) : size(values.size()) {
    check(cudaMalloc(&data, size * sizeof(T)));
    check(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<T> read() const {
    std::vector<T> values(size);
    check(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

/** Compare the actual density-bound reduction with independent scalar shell
 * blocks. Tiny/partial warps, multi-iteration walks, inactive neighbors and
 * opposite-spin cancellation must all reach the shared-memory handoff safely. */
void check_density_bounds(unsigned spins, unsigned threads, bool masked, bool tiny) {
  using namespace generativeqc::scf::cuda_execution;
  const unsigned systems = tiny ? 1 : 3, n = tiny ? 2 : 35, matrix = n * n;
  const std::vector<unsigned> shell_sizes =
      tiny ? std::vector<unsigned>{1, 1} : std::vector<unsigned>{1, 3, 6, 10, 15};
  const unsigned shells = shell_sizes.size();
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> first, second, owners;
  for (unsigned system = 0; system < systems; ++system) {
    for (unsigned shell = 0; shell < shells; ++shell) {
      offsets.push_back(offsets.back() + shell_sizes[shell]);
      for (unsigned other = 0; other <= shell; ++other) {
        first.push_back(system * shells + shell);
        second.push_back(system * shells + other);
        owners.push_back(system);
      }
    }
  }
  std::vector<double> values(systems * spins * matrix);
  for (std::size_t element = 0; element < values.size(); ++element)
    values[element] = (static_cast<int>((element * 17) % 257) - 128) / 64.0;
  const std::vector<std::uint8_t> mask =
      tiny ? std::vector<std::uint8_t>{1} : std::vector<std::uint8_t>{1, 0, 1};
  Device<std::int64_t> device_offsets(offsets);
  Device<std::int32_t> device_first(first), device_second(second), device_owners(owners);
  Device<double> density(values);
  Device<std::uint8_t> active(mask);
  Device<ShellPairDensityBounds> bounds(std::vector<ShellPairDensityBounds>(first.size()));
  DeviceBatch batch{};
  batch.batch_size = systems;
  batch.direct_nbf = n;
  batch.total_shell_pairs = first.size();
  batch.shell_direct_ao_offsets = device_offsets.data;
  batch.shell_pair_first = device_first.data;
  batch.shell_pair_second = device_second.data;
  batch.shell_pair_systems = device_owners.data;
  // Match the production captured iteration as well as its tiny shell blocks.
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  launch_reduce_shell_pair_density_bounds_kernel(
      spins == 2, first.size(), threads, 3 * threads * sizeof(double), stream, batch, density.data,
      masked ? active.data : nullptr, bounds.data);
  check(cudaGetLastError());
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  check(cudaGraphLaunch(executable, stream));
  check(cudaStreamSynchronize(stream));
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  const auto actual = bounds.read();
  for (std::size_t pair = 0; pair < first.size(); ++pair) {
    const auto system = owners[pair];
    ShellPairDensityBounds expected{};
    if (!masked || mask[system]) {
      for (auto row = offsets[first[pair]]; row < offsets[first[pair] + 1]; ++row)
        for (auto column = offsets[second[pair]]; column < offsets[second[pair] + 1]; ++column) {
          const auto i = row - system * n, j = column - system * n;
          for (const auto element : {i * n + j, j * n + i}) {
            const double alpha = values[system * spins * matrix + element];
            const double beta = spins == 2 ? values[(system * spins + 1) * matrix + element] : 0;
            expected.coulomb = std::max(expected.coulomb, std::abs(alpha + beta));
            expected.exchange_alpha = std::max(expected.exchange_alpha, std::abs(alpha));
            expected.exchange_beta = std::max(expected.exchange_beta, std::abs(beta));
          }
        }
    }
    if (actual[pair].coulomb != expected.coulomb ||
        actual[pair].exchange_alpha != expected.exchange_alpha ||
        actual[pair].exchange_beta != expected.exchange_beta)
      throw std::runtime_error("density-bound reduction changed a scalar shell-block maximum");
  }
}

void check_convergence(unsigned spins, bool retain, bool require_physical, bool approximate,
                       double tolerance, double energy_tolerance = 1e-12,
                       double allowed_residual = 0.0, bool qualified_owner = false) {
  using namespace generativeqc::scf::cuda_execution;
  constexpr unsigned n = 8, batch = 6;
  const std::size_t size = spins * n * n;
  const double bound = allowed_residual > 0 ? allowed_residual : std::min(1e-8, tolerance);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> residual(batch * size, 0);
  const double peaks[]{bound / 2, bound, 3 * bound, nan, std::numeric_limits<double>::infinity(),
                       nan};
  for (unsigned i = 0; i < batch; ++i) residual[(i + 1) * size - 1] = peaks[i];
  Device<double> d_residual(residual), density(std::vector<double>(batch * size, 0.25)),
      next(std::vector<double>(batch * size, 0.25 + 1e-12)), energy(std::vector<double>(batch, -1)),
      previous(std::vector<double>(batch, -1)), change(std::vector<double>(batch, nan)),
      rms(std::vector<double>(batch, nan));
  Device<std::uint8_t> active(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0}),
      converged(std::vector<std::uint8_t>(batch, 0));
  Device<std::uint32_t> iterations(std::vector<std::uint32_t>(batch, 1));
  Device<std::uint32_t> census(std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0});
  const auto launch =
      spins == 1 ? launch_update_convergence_kernel : launch_update_uhf_convergence_kernel;
  launch(retain, batch, 32, 0, nullptr, batch, n, energy_tolerance, tolerance, false, energy.data,
         previous.data, next.data, density.data, active.data, converged.data, iterations.data,
         change.data, rms.data, require_physical ? d_residual.data : nullptr,
         approximate ? census.data : nullptr, qualified_owner, nullptr, nullptr);
  check(cudaGetLastError());
  const auto flags = converged.read();
  const auto counts = iterations.read();
  const auto density_out = density.read();
  for (unsigned i = 0; i < batch; ++i) {
    const bool expected = i < 5 && (!require_physical || i < 2 || (approximate && i < 4));
    if (flags[i] != expected || counts[i] != (i == 5 ? 1U : 2U))
      throw std::runtime_error("tiny density step bypassed the physical force criterion");
    const double expected_density = i == 5 || (retain && expected) ? 0.25 : 0.25 + 1e-12;
    if (density_out[i * size] != expected_density)
      throw std::runtime_error("physical convergence changed density-retention semantics");
  }

  if (require_physical && !approximate) {
    // Exercise the fitted owner independently with the same adversarial
    // residuals, including unsupported loose UHF and unqualified RHF owners.
    Device<double> df_alpha(std::vector<double>(batch * n * n, 0.25)),
        df_beta(std::vector<double>(batch * n * n, 0.25)),
        df_next(std::vector<double>(batch * n * n, 0.25 + 1e-12)),
        df_previous(std::vector<double>(batch, -1));
    Device<std::uint8_t> df_active(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0}),
        df_converged(std::vector<std::uint8_t>(batch, 0));
    Device<std::uint32_t> df_iterations(std::vector<std::uint32_t>(batch, 1));
    if (spins == 1)
      generativeqc::scf::cuda_df::launch_update_device_convergence_kernel(
          batch, 32, 0, nullptr, batch, n, energy_tolerance, tolerance, energy.data,
          df_previous.data, df_next.data, df_alpha.data, df_active.data, df_converged.data,
          df_iterations.data, change.data, rms.data, d_residual.data, qualified_owner);
    else
      generativeqc::scf::cuda_df::launch_update_device_uhf_convergence_kernel(
          batch, 32, 0, nullptr, batch, n, energy_tolerance, tolerance, energy.data,
          df_previous.data, df_next.data, df_next.data, df_alpha.data, df_beta.data, df_active.data,
          df_converged.data, df_iterations.data, change.data, rms.data, d_residual.data);
    check(cudaGetLastError());
    if (df_converged.read() != flags)
      throw std::runtime_error("direct and DF owner admission disagree");
  }

  // The rebuilt force determinant must be checked separately: iteration
  // acceptance says nothing about a later projection's physical residual.
  Device<std::uint8_t> final_active(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0}),
      final_converged(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0});
  Device<std::uint32_t> tested(std::vector<std::uint32_t>{0});
  // The kernel reports per-item participation separately from the aggregate
  // counter so callers can distinguish inactive items from rejected ones.
  Device<std::uint8_t> tested_items(std::vector<std::uint8_t>(batch, 0));
  launch_validate_force_residual_kernel(nullptr, batch, spins, n, tolerance, d_residual.data,
                                        final_active.data, final_converged.data, tested.data,
                                        tested_items.data, energy_tolerance, qualified_owner);
  check(cudaGetLastError());
  if (final_converged.read() != std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0} ||
      final_active.read() != std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0} || tested.read()[0] != 5 ||
      tested_items.read() != std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0})
    throw std::runtime_error("force validation missed a bad determinant or lost rejected work");
}
/** Independent fixtures from the 768-AO diagnosis plus adversarial states.
 * Both kernel families must reach the same verdict, including first-step
 * baseline admission, localized beta residuals and nonfinite observations. */
void check_direct_df_acceptance(unsigned spins) {
  constexpr unsigned n = 8, batch = 11;
  const std::size_t matrix = n * n, size = spins * matrix;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> energies(batch, -2431.0), baselines(batch, -2431.0),
      residual(batch * size, 0.0), density(batch * size, 0.25), next(batch * size, 0.25 + 1e-12);
  baselines[0] += 4.55e-12;  // Measured DF third-step scale: within the common guard.
  baselines[1] += 2.01e-10;  // Measured seed-to-canonical scale: still rejected.
  baselines[2] = infinity;   // An absent baseline is not a zero energy change.
  energies[3] = nan;
  for (std::size_t k = 4 * size; k < 5 * size; ++k) next[k] += 2e-10;
  residual[6 * size - 1] = 2e-10;  // UHF error lies only in beta.
  residual[7 * size - 1] = nan;
  energies[7] = infinity;
  next[9 * size - 1] = nan;
  // Item 9 has a valid baseline at iteration zero; item 10 is inactive and
  // must preserve its counters/flags even though every input is invalid.
  energies[10] = baselines[10] = nan;
  const std::vector<std::uint8_t> expected{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
  for (bool fitted : {false, true}) {
    Device<double> d_energy(energies), previous(baselines), d_residual(residual),
        change(std::vector<double>(batch, nan)), rms(std::vector<double>(batch, nan));
    std::vector<std::uint8_t> mask(batch, 1);
    mask.back() = 0;
    Device<std::uint8_t> active(mask), converged(std::vector<std::uint8_t>(batch, 0));
    Device<std::uint32_t> iterations(std::vector<std::uint32_t>(batch, 0));
    if (!fitted) {
      Device<double> current(density), proposal(next);
      const auto launch =
          spins == 1 ? generativeqc::scf::cuda_execution::launch_update_convergence_kernel
                     : generativeqc::scf::cuda_execution::launch_update_uhf_convergence_kernel;
      launch(false, batch, 32, 0, nullptr, batch, n, 1e-12, 1e-10, true, d_energy.data,
             previous.data, proposal.data, current.data, active.data, converged.data,
             iterations.data, change.data, rms.data, d_residual.data, nullptr, false, nullptr,
             nullptr);
      check(cudaGetLastError());
    } else {
      std::vector<double> alpha(batch * matrix), beta(batch * matrix), next_alpha(batch * matrix),
          next_beta(batch * matrix);
      for (unsigned item = 0; item < batch; ++item)
        for (std::size_t k = 0; k < matrix; ++k) {
          alpha[item * matrix + k] = density[item * size + k];
          next_alpha[item * matrix + k] = next[item * size + k];
          if (spins == 2) {
            beta[item * matrix + k] = density[item * size + matrix + k];
            next_beta[item * matrix + k] = next[item * size + matrix + k];
          }
        }
      Device<double> da(alpha), db(beta), na(next_alpha), nb(next_beta);
      if (spins == 1)
        generativeqc::scf::cuda_df::launch_update_device_convergence_kernel(
            batch, 32, 0, nullptr, batch, n, 1e-12, 1e-10, d_energy.data, previous.data, na.data,
            da.data, active.data, converged.data, iterations.data, change.data, rms.data,
            d_residual.data);
      else
        generativeqc::scf::cuda_df::launch_update_device_uhf_convergence_kernel(
            batch, 32, 0, nullptr, batch, n, 1e-12, 1e-10, d_energy.data, previous.data, na.data,
            nb.data, da.data, db.data, active.data, converged.data, iterations.data, change.data,
            rms.data, d_residual.data);
      check(cudaGetLastError());
    }
    if (converged.read() != expected)
      throw std::runtime_error("direct/DF RHF/UHF acceptance differs from independent fixtures");
    const auto changes = change.read();
    if (!(changes[0] > 1e-12 && changes[0] < 1e-11) || !std::isinf(changes[2]))
      throw std::runtime_error("roundoff policy hid the raw energy delta or missing baseline");
    const auto counts = iterations.read();
    if (counts.back() != 0 ||
        !std::all_of(counts.begin(), counts.end() - 1, [](auto value) { return value == 1; }))
      throw std::runtime_error("shared acceptance changed work counts or inactive isolation");
  }
}
/** Exercise the real multi-CTA controller and convergence kernels together.
 * A manufactured stationary tail must retire deltas, reject a reconstructed
 * baseline, and require consecutive full evaluations. Neighbors retain their
 * own warm/coarse/invalid/inactive state, for both spin and density lifetimes. */
void check_incremental_energy_refinement(unsigned spins, bool retain) {
  using namespace generativeqc::scf::cuda_execution;
  constexpr unsigned batch = 8, basis_size = 16;
  constexpr double tolerance = 1e-9;
  const std::size_t matrix_size = basis_size * basis_size;
  const std::size_t vector_size = spins * matrix_size;
  const std::size_t total = batch * vector_size;
  const unsigned blocks = static_cast<unsigned>((total + 127) / 128);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> energies(batch, -1200.0), baselines(batch, -1200.0);
  baselines[0] += 1e-8;
  energies[6] = nan;
  std::vector<double> residual(total, 0.0), proposal(total, 0.25 + 1e-12);
  residual[4 * vector_size - 1] = 2 * tolerance;
  residual[5 * vector_size - 1] = nan;
  proposal.back() += 1e-6;
  Device<double> energy(energies), previous(baselines), physical(residual), next(proposal),
      density(std::vector<double>(total, 0.25)), change(std::vector<double>(batch, nan)),
      rms(std::vector<double>(batch, nan));
  Device<std::uint8_t> active(std::vector<std::uint8_t>{1, 1, 1, 1, 0, 1, 1, 1}),
      converged(std::vector<std::uint8_t>(batch, 0)), full(std::vector<std::uint8_t>(batch, 0));
  Device<std::uint32_t> iterations(std::vector<std::uint32_t>(batch, 0)),
      updates(std::vector<std::uint32_t>{2, 2, 0, 2, 2, 2, 2, 2}),
      census(std::vector<std::uint32_t>{0, 0, 0, 0, 0, 1, 0, 0});
  Device<unsigned long long> full_count(std::vector<unsigned long long>(batch, 0)),
      delta_count(std::vector<unsigned long long>{2, 2, 0, 2, 2, 2, 2, 2});
  const auto launch =
      spins == 1 ? launch_update_convergence_kernel : launch_update_uhf_convergence_kernel;
  const auto converge = [&] {
    launch(retain, batch, 32, 0, nullptr, batch, basis_size, 1e-12, tolerance, true, energy.data,
           previous.data, next.data, density.data, active.data, converged.data, iterations.data,
           change.data, rms.data, physical.data, census.data, false, updates.data,
           delta_count.data);
    check(cudaGetLastError());
  };
  converge();
  auto states = updates.read();
  auto flags = converged.read();
  const auto baseline_after = previous.read();
  if (states[0] != kIncrementalDirectJkFullEnergyRefinement ||
      states[1] != kIncrementalDirectJkFullEnergyRefinement || flags[0] || flags[1] ||
      !std::isinf(baseline_after[0]) || !std::isinf(baseline_after[1]) ||
      !(change.read()[0] > 1e-9))
    throw std::runtime_error("stationary delta tail retained a reconstructed energy baseline");
  if (flags[2] != 1 || states[2] != 0 || states[3] != 2 || states[4] != 2 || states[5] != 2 ||
      states[6] != 2 || states[7] != 2 || flags[3] || flags[4] || flags[6] || flags[7] ||
      iterations.read()[4] != 0)
    throw std::runtime_error("full-energy fallback contaminated a neighboring item");

  const double full_provider_value = std::nextafter(1.0, 0.0);
  Device<double> hcore(std::vector<double>(batch * matrix_size, 0.5)),
      anchor_density(std::vector<double>(total, 97.0)),
      anchor_fock(std::vector<double>(total, 98.0)), delta(std::vector<double>(total, 99.0)),
      fock(std::vector<double>(total, full_provider_value)),
      maximum(std::vector<double>(batch, 0.0));
  for (unsigned full_iteration = 1; full_iteration <= 2; ++full_iteration) {
    launch_prepare_incremental_direct_jk_kernel(
        blocks, 128, 0, nullptr, batch, spins, basis_size, 0, 0.0, rms.data, density.data,
        hcore.data, active.data, anchor_density.data, anchor_fock.data, delta.data, updates.data,
        full.data, full_count.data, delta_count.data, maximum.data);
    check(cudaGetLastError());
    const auto inputs = delta.read();
    const auto densities = density.read();
    if (full.read()[0] != 1 || full.read()[1] != 1 ||
        !std::equal(inputs.begin(), inputs.begin() + 2 * vector_size, densities.begin()))
      throw std::runtime_error("full-only fallback fed a density delta to the provider");
    launch_finalize_incremental_direct_jk_kernel(
        blocks, 128, 0, nullptr, batch, spins, basis_size, true, density.data, hcore.data,
        active.data, anchor_density.data, anchor_fock.data, fock.data, updates.data, full.data);
    check(cudaGetLastError());
    states = updates.read();
    const auto provider_result = fock.read();
    if (!std::all_of(provider_result.begin(), provider_result.begin() + 2 * vector_size,
                     [=](double value) { return value == full_provider_value; }))
      throw std::runtime_error("full fallback changed the provider's physical Fock");
    const auto full_counts = full_count.read();
    const auto delta_counts = delta_count.read();
    if (states[0] != kIncrementalDirectJkFullEnergyRefinement ||
        states[1] != kIncrementalDirectJkFullEnergyRefinement || full_counts[0] != full_iteration ||
        full_counts[1] != full_iteration || delta_counts[0] != 2 || delta_counts[1] != 2 ||
        full_counts[4] != 0 || anchor_density.read()[4 * vector_size] != 97.0)
      throw std::runtime_error("full-energy fallback was not sticky or lost work accounting");
    converge();
    flags = converged.read();
    const bool expected_convergence = full_iteration == 2;
    if (flags[0] != expected_convergence || flags[1] != expected_convergence)
      throw std::runtime_error("fallback did not require two consecutive full Focks");
  }
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    check(cudaSetDevice(0));
    for (unsigned spins : {1U, 2U})
      for (unsigned threads : {32U, 128U, 256U})
        for (bool masked : {false, true})
          for (bool tiny : {false, true}) check_density_bounds(spins, threads, masked, tiny);
    for (unsigned spins : {1U, 2U})
      for (bool retain : {false, true}) check_incremental_energy_refinement(spins, retain);
    for (unsigned spins : {1U, 2U}) check_direct_df_acceptance(spins);
    for (unsigned spins : {1U, 2U})
      for (bool retain : {false, true})
        for (bool physical : {false, true})
          for (bool approximate : {false, true})
            for (double tolerance : {1e-10, 1e-6})
              check_convergence(spins, retain, physical, approximate, tolerance);
    // Standard admits a bounded residual; a tight energy request restores the
    // original ceiling. Check both iteration and rebuilt-force consumers,
    // including localized beta errors, NaNs and inactive neighbors.
    for (unsigned spins : {1U, 2U}) {
      check_convergence(spins, true, true, false, 1e-5, 1e-9, spins == 1 ? 1e-5 : 1e-8, true);
      // Explicit loose controls do not qualify ECP/unknown owners. UHF stays
      // strict even if a caller accidentally supplies restricted admission.
      check_convergence(spins, true, true, false, 1e-5, 1e-9, 1e-8, false);
      check_convergence(spins, true, true, false, 1e-5, 1e-12, 1e-8);
    }
    std::cout << "Physical force convergence and final-state rejection passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
