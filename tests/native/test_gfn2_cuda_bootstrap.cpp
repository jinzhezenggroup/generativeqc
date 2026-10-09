// Exercise the real molecular CUDA transaction, including first-call failures.
// Oracle values are supplied from the independent tblite fixture by the runner.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gfn2_cuda_execution.hpp"

using namespace generativeqc::xtb::detail;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
generativeqc_xtb_const_buffer_t input(const std::vector<T>& values) {
  return {values.data(), values.size() * sizeof(T), GENERATIVEQC_XTB_MEMORY_HOST, 0};
}
template <class T>
generativeqc_xtb_buffer_t output(std::vector<T>& values) {
  return {values.data(), values.size() * sizeof(T), GENERATIVEQC_XTB_MEMORY_HOST, 0};
}
struct Molecule {
  std::string name;
  std::vector<std::int32_t> numbers;
  std::vector<double> positions, forces;
  double energy;
};
struct Result {
  std::vector<double> energy{123.0}, forces;
  std::vector<std::int32_t> iterations{-42}, statuses{-43};
  std::vector<std::uint8_t> converged{0xa5};
  std::uint32_t flags = 0;
  generativeqc_xtb_status_t status = -1;
};

Result run(Gfn2CudaExecutionCache& cache, const Molecule& molecule, bool device_input,
           bool forces = true, int maximum_iterations = 300) {
  const auto count = static_cast<std::int64_t>(molecule.numbers.size());
  std::vector<std::int64_t> offsets{0, count};
  std::vector<double> charge{0};
  std::vector<std::int32_t> unpaired{0}, spin{1};
  generativeqc_xtb_batch_t batch{};
  batch.struct_size = sizeof(batch);
  batch.api_version = GENERATIVEQC_XTB_API_VERSION;
  batch.batch_size = 1;
  batch.total_atoms = count;
  batch.atom_offsets = input(offsets);
  batch.atomic_numbers = input(molecule.numbers);
  batch.positions = input(molecule.positions);
  batch.molecular_charges = input(charge);
  batch.unpaired_electrons = input(unpaired);
  batch.spin_channels = input(spin);
  struct DevicePositions {
    double* pointer = nullptr;
    ~DevicePositions() {
      if (pointer) cudaFree(pointer);
    }
  } device;
  if (device_input) {
    check(cudaMalloc(reinterpret_cast<void**>(&device.pointer),
                     molecule.positions.size() * sizeof(double)));
    check(cudaMemcpy(device.pointer, molecule.positions.data(),
                     molecule.positions.size() * sizeof(double), cudaMemcpyHostToDevice));
    batch.positions.data = device.pointer;
    batch.positions.memory_space = GENERATIVEQC_XTB_MEMORY_CUDA_DEVICE;
  }
  generativeqc_xtb_compute_options_t options{};
  options.struct_size = sizeof(options);
  options.api_version = GENERATIVEQC_XTB_API_VERSION;
  options.model = GENERATIVEQC_XTB_MODEL_GFN2_XTB;
  options.flags = GENERATIVEQC_XTB_COMPUTE_ENERGY | (forces ? GENERATIVEQC_XTB_COMPUTE_FORCES : 0);
  options.max_scc_iterations = maximum_iterations;
  options.energy_tolerance = 1e-12;
  options.charge_tolerance = 1e-10;
  options.electronic_temperature = GENERATIVEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.scc_start_mode = GENERATIVEQC_XTB_SCC_START_FRESH;
  options.scc_mixer = GENERATIVEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
  options.scc_mixer_history = 8;
  options.scc_mixer_damping = 0.4;
  Result result;
  if (forces) result.forces.assign(3 * count, -321.0);
  generativeqc_xtb_batch_result_t destination{};
  destination.struct_size = sizeof(destination);
  destination.api_version = GENERATIVEQC_XTB_API_VERSION;
  destination.energies = output(result.energy);
  if (forces) destination.forces = output(result.forces);
  destination.scc_iterations = output(result.iterations);
  destination.scc_converged = output(result.converged);
  destination.per_system_status = output(result.statuses);
  std::string error;
  result.status = execute_restricted_gfn2_cuda(cache, batch, options, destination, error);
  result.flags = destination.flags;
  return result;
}

void oracle(const Result& result, const Molecule& molecule) {
  require(result.status == GENERATIVEQC_XTB_STATUS_SUCCESS &&
              result.statuses[0] == GENERATIVEQC_XTB_STATUS_SUCCESS && result.converged[0] == 1,
          "healthy request did not converge");
  require(std::isfinite(result.energy[0]) && std::abs(result.energy[0] - molecule.energy) < 5e-7,
          "first or recovered energy differs from tblite oracle");
  for (std::size_t i = 0; i < result.forces.size(); ++i)
    require(
        std::isfinite(result.forces[i]) && std::abs(result.forces[i] - molecule.forces[i]) < 5e-7,
        "first or recovered force differs from tblite oracle");
}

void failed(const Result& result) {
  if (result.status != GENERATIVEQC_XTB_STATUS_SUCCESS) {
    // Synchronous admission/refresh rejection must not commit caller bytes.
    require(result.energy[0] == 123.0 && result.iterations[0] == -42 && result.statuses[0] == -43 &&
                result.converged[0] == 0xa5 && result.flags == 0,
            "rejected request changed output or diagnostic sentinels");
    for (double value : result.forces) require(value == -321.0, "rejected force was published");
  } else {
    // A documented terminal SCC failure commits diagnostics and NaN science.
    require(result.statuses[0] != GENERATIVEQC_XTB_STATUS_SUCCESS && result.converged[0] == 0,
            "invalid first request published success");
    require(std::isnan(result.energy[0]), "failed SCC published a finite energy");
    for (double value : result.forces) require(std::isnan(value), "failed SCC published a force");
  }
}

void verify_mixer_receipts(const Molecule& molecule, bool device_input) {
  Gfn2CudaExecutionCache baseline_cache(0, nullptr);
  Gfn2CudaExecutionCache measured_cache(0, nullptr);
  require(measured_cache.enable_mixer_diagnostics(), "could not enable mixer receipts");
  const auto baseline_start = std::chrono::steady_clock::now();
  const auto baseline = run(baseline_cache, molecule, device_input);
  const auto baseline_stop = std::chrono::steady_clock::now();
  const auto measured = run(measured_cache, molecule, device_input);
  const auto measured_stop = std::chrono::steady_clock::now();
  oracle(baseline, molecule);
  oracle(measured, molecule);
  const auto maximum_force_difference = [](const Result& first, const Result& second) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < first.forces.size(); ++index)
      maximum = std::max(maximum, std::abs(first.forces[index] - second.forces[index]));
    return maximum;
  };
  constexpr double kEndpointAgreement = 5e-14;
  const bool equivalent = std::abs(measured.energy[0] - baseline.energy[0]) <= kEndpointAgreement &&
                          maximum_force_difference(measured, baseline) <= kEndpointAgreement &&
                          measured.iterations == baseline.iterations &&
                          measured.statuses == baseline.statuses &&
                          measured.converged == baseline.converged;
  if (!equivalent) {
    Gfn2CudaExecutionCache control_cache(0, nullptr);
    const auto control = run(control_cache, molecule, device_input);
    oracle(control, molecule);
    std::fprintf(stderr,
                 "MIXER_AB_DIAGNOSTIC case=%s ingress=%s baseline_energy=%.17g "
                 "control_energy=%.17g measured_energy=%.17g "
                 "aa_force_max=%.17g ab_force_max=%.17g "
                 "iterations=%d,%d,%d statuses=%d,%d,%d\n",
                 molecule.name.c_str(), device_input ? "device" : "host", baseline.energy[0],
                 control.energy[0], measured.energy[0], maximum_force_difference(baseline, control),
                 maximum_force_difference(baseline, measured), baseline.iterations[0],
                 control.iterations[0], measured.iterations[0], baseline.statuses[0],
                 control.statuses[0], measured.statuses[0]);
  }
  require(equivalent, "mixer receipts changed the complete energy/host-force endpoint");

  Gfn2CudaMixerDiagnosticSnapshot snapshot;
  std::string error;
  require(measured_cache.read_mixer_diagnostics(snapshot, error),
          "completed mixer diagnostic readback failed");
  require(snapshot.graph_submitted && snapshot.endpoint_completed && snapshot.graph_family != 0 &&
              snapshot.plan_token != 0 && snapshot.device_id == 0 && snapshot.call_id == 1,
          "mixer receipts are not bound to a completed device graph call");
  require(snapshot.attempted_receipts == snapshot.receipts.size() && !snapshot.receipts.empty() &&
              snapshot.receipts.size() <= static_cast<std::size_t>(measured.iterations[0]),
          "mixer receipt count is missing or overflowed");
  std::uint64_t coefficient_elements = 0, gram_elements = 0, combination_elements = 0;
  std::uint64_t residual_cycles = 0, history_cycles = 0, solve_cycles = 0, combination_cycles = 0;
  for (std::size_t index = 0; index < snapshot.receipts.size(); ++index) {
    const auto& receipt = snapshot.receipts[index];
    coefficient_elements += receipt.coefficient_dot_elements;
    gram_elements += receipt.gram_dot_elements;
    combination_elements += receipt.combination_elements;
    residual_cycles += receipt.residual_elapsed_cycles;
    history_cycles += receipt.history_elapsed_cycles;
    solve_cycles += receipt.solve_elapsed_cycles;
    combination_cycles += receipt.combination_elapsed_cycles;
    const auto live = std::min<std::uint64_t>(8u, receipt.iteration_before);
    require(receipt.invocation == index && receipt.system == 0 &&
                receipt.iteration_before == index && receipt.restart_before == 0 &&
                receipt.vector_elements > 0 && receipt.status == GENERATIVEQC_XTB_STATUS_SUCCESS &&
                (receipt.completed_stages & cuda::kMixerCommitted) != 0,
            "mixer invocation identity or terminal status disagrees with fresh SCC");
    if (live == 0u) {
      require(receipt.coefficient_dot_elements == 0 && receipt.gram_dot_elements == 0 &&
                  receipt.combination_elements == 0 && receipt.live_history == 0,
              "first mixer invocation falsely reported history algebra");
      continue;
    }
    const auto length = static_cast<std::uint64_t>(receipt.vector_elements);
    require(receipt.live_history == static_cast<std::int64_t>(live) &&
                receipt.new_slot == static_cast<std::int64_t>((index - 1u) % 8u) &&
                receipt.coefficient_dot_elements == live * length &&
                receipt.gram_dot_elements == live * live * length &&
                receipt.combination_elements == live * length &&
                (receipt.completed_stages & (cuda::kMixerResidual | cuda::kMixerHistory |
                                             cuda::kMixerSolve | cuda::kMixerCombination)) ==
                    (cuda::kMixerResidual | cuda::kMixerHistory | cuda::kMixerSolve |
                     cuda::kMixerCombination),
            "actual mixer history work disagrees with independent ring trajectory");
  }
  std::printf(
      "MIXER_RECEIPT case=%s ingress=%s call=%llu graph=%u invocations=%llu "
      "coefficient_elements=%llu gram_elements=%llu combination_elements=%llu "
      "residual_cycles=%llu history_cycles=%llu solve_cycles=%llu "
      "combination_cycles=%llu\n",
      molecule.name.c_str(), device_input ? "device" : "host",
      static_cast<unsigned long long>(snapshot.call_id), snapshot.graph_family,
      static_cast<unsigned long long>(snapshot.attempted_receipts),
      static_cast<unsigned long long>(coefficient_elements),
      static_cast<unsigned long long>(gram_elements),
      static_cast<unsigned long long>(combination_elements),
      static_cast<unsigned long long>(residual_cycles),
      static_cast<unsigned long long>(history_cycles),
      static_cast<unsigned long long>(solve_cycles),
      static_cast<unsigned long long>(combination_cycles));

  const auto replay_start = std::chrono::steady_clock::now();
  const auto replay = run(measured_cache, molecule, device_input);
  const auto replay_stop = std::chrono::steady_clock::now();
  oracle(replay, molecule);
  require(std::abs(replay.energy[0] - measured.energy[0]) <= kEndpointAgreement &&
              maximum_force_difference(replay, measured) <= kEndpointAgreement &&
              replay.iterations == measured.iterations && replay.statuses == measured.statuses,
          "mixer graph replay changed the complete endpoint");
  Gfn2CudaMixerDiagnosticSnapshot replay_snapshot;
  require(measured_cache.read_mixer_diagnostics(replay_snapshot, error) &&
              replay_snapshot.call_id == 2 && replay_snapshot.plan_token == snapshot.plan_token &&
              replay_snapshot.graph_family == snapshot.graph_family &&
              replay_snapshot.graph_submitted && replay_snapshot.endpoint_completed &&
              replay_snapshot.attempted_receipts == snapshot.attempted_receipts &&
              replay_snapshot.receipts.size() == snapshot.receipts.size(),
          "mixer graph replay did not reset its call-owned receipts");
  for (std::size_t index = 0; index < snapshot.receipts.size(); ++index) {
    const auto& before = snapshot.receipts[index];
    const auto& after = replay_snapshot.receipts[index];
    require(after.invocation == before.invocation && after.system == before.system &&
                after.iteration_before == before.iteration_before &&
                after.restart_before == before.restart_before &&
                after.vector_elements == before.vector_elements &&
                after.live_history == before.live_history && after.new_slot == before.new_slot &&
                after.coefficient_dot_elements == before.coefficient_dot_elements &&
                after.gram_dot_elements == before.gram_dot_elements &&
                after.combination_elements == before.combination_elements &&
                after.completed_stages == before.completed_stages && after.status == before.status,
            "mixer graph replay changed semantic work or retained a stale receipt");
  }
  const auto seconds = [](auto first, auto last) {
    return std::chrono::duration<double>(last - first).count();
  };
  std::printf(
      "MIXER_ENDPOINT case=%s ingress=%s baseline_seconds=%.9f "
      "measured_seconds=%.9f replay_seconds=%.9f\n",
      molecule.name.c_str(), device_input ? "device" : "host",
      seconds(baseline_start, baseline_stop), seconds(baseline_stop, measured_stop),
      seconds(replay_start, replay_stop));
}

void verify_failed_mixer_receipts(const Molecule& water, bool device_input) {
  Gfn2CudaExecutionCache cache(0, nullptr);
  require(cache.enable_mixer_diagnostics(), "could not enable failure receipts");
  auto bad = water;
  bad.positions[0] = std::numeric_limits<double>::quiet_NaN();
  const auto first = run(cache, bad, device_input);
  failed(first);
  Gfn2CudaMixerDiagnosticSnapshot first_snapshot;
  std::string error;
  require(
      cache.read_mixer_diagnostics(first_snapshot, error) && first_snapshot.call_id == 1 &&
          first_snapshot.plan_token != 0 &&
          first_snapshot.endpoint_completed == (first.status == GENERATIVEQC_XTB_STATUS_SUCCESS),
      "failed first call lost its call-owned mixer diagnostics");
  require(first_snapshot.graph_submitted == (first_snapshot.graph_family != 0),
          "failed first call misreported graph submission");

  oracle(run(cache, water, device_input), water);
  Gfn2CudaMixerDiagnosticSnapshot recovered;
  require(cache.read_mixer_diagnostics(recovered, error) && recovered.call_id == 2 &&
              recovered.endpoint_completed && !recovered.receipts.empty(),
          "recovery retained stale failed-call diagnostics");
  require(!cache.enable_mixer_diagnostics(), "late enablement changed prepared graph bindings");

  // A different iteration limit requires a new Prepared even for the same
  // molecule. A failed candidate must not publish that topology or lose its
  // independent receipt identity while the prior Prepared remains reusable.
  const auto replacement = run(cache, bad, device_input, true, 301);
  failed(replacement);
  Gfn2CudaMixerDiagnosticSnapshot replaced;
  require(
      cache.read_mixer_diagnostics(replaced, error) && replaced.call_id == 3 &&
          replaced.plan_token != recovered.plan_token &&
          replaced.endpoint_completed == (replacement.status == GENERATIVEQC_XTB_STATUS_SUCCESS),
      "failed replacement read the prior topology's mixer diagnostics");
  oracle(run(cache, water, device_input), water);
  Gfn2CudaMixerDiagnosticSnapshot replay;
  require(cache.read_mixer_diagnostics(replay, error) && replay.call_id == 4 &&
              replay.endpoint_completed &&
              replay.attempted_receipts == recovered.attempted_receipts,
          "valid reuse after a failed replacement lost or accumulated receipts");
  if (replacement.status != GENERATIVEQC_XTB_STATUS_SUCCESS)
    require(replay.plan_token == recovered.plan_token, "failed replacement published its topology");

  // Invalid iteration limits are rejected before any receipt reset. The last
  // readable host snapshot must become unavailable rather than masquerading
  // as diagnostics for this new call.
  failed(run(cache, water, device_input, true, 0));
  require(!cache.read_mixer_diagnostics(replay, error),
          "pre-reset rejection exposed an earlier call's mixer diagnostics");
}

void verify_density_endpoint_timing(const Molecule& molecule, bool device_input) {
  if (molecule.name != "h2o" && molecule.name != "water-8" && molecule.name != "water-32") return;
  Gfn2CudaExecutionCache cache(0, nullptr);
#if defined(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
  require(cache.enable_density_diagnostics(), "could not enable timed density receipts");
  constexpr const char* arm = "on";
#else
  constexpr const char* arm = "off";
#endif
  oracle(run(cache, molecule, device_input), molecule);
  std::vector<double> seconds;
  for (int repeat = 0; repeat < 5; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    const auto result = run(cache, molecule, device_input);
#if defined(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
    Gfn2CudaDensityDiagnosticSnapshot snapshot;
    std::string error;
    require(cache.read_density_diagnostics(snapshot, error) && snapshot.graph_submitted &&
                snapshot.endpoint_completed &&
                snapshot.attempted_receipts <= snapshot.receipt_capacity &&
                snapshot.receipts.size() == snapshot.attempted_receipts,
            "timed density readback was incomplete");
#endif
    const auto end = std::chrono::steady_clock::now();
    oracle(result, molecule);
    seconds.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(seconds.begin(), seconds.end());
  std::printf(
      "DENSITY_ENDPOINT arm=%s case=%s ingress=%s repetitions=5 "
      "median_seconds=%.9f min_seconds=%.9f max_seconds=%.9f\n",
      arm, molecule.name.c_str(), device_input ? "device" : "host", seconds[seconds.size() / 2],
      seconds.front(), seconds.back());
}

#if defined(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
void append_density_receipts(const Gfn2CudaDensityDiagnosticSnapshot& snapshot,
                             const Molecule& molecule, bool device_input, const char* phase) {
  const char* const path = std::getenv("GENERATIVEQC_GFN2_DENSITY_RECEIPT_CSV");
  if (path == nullptr || *path == '\0') return;
  std::ofstream stream(path, std::ios::app);
  require(bool(stream), "could not open durable density receipt CSV");
  if (stream.tellp() == 0)
    stream << "case,ingress,phase,call_id,plan_token,device_id,graph_family,graph_submitted,"
              "endpoint_completed,attempted,capacity,arena_bytes,slot,system,spin_channels,"
              "channel,tile,orbital_count,pair_count,pairs_visited,plain_visits,"
              "plain_completed,weighted_visits,weighted_completed,failed_pairs,"
              "published_pairs,cta_cycles,status\n";
  if (snapshot.receipts.empty())
    stream << molecule.name << ',' << (device_input ? "device" : "host") << ',' << phase << ','
           << snapshot.call_id << ',' << snapshot.plan_token << ',' << snapshot.device_id << ','
           << snapshot.graph_family << ',' << snapshot.graph_submitted << ','
           << snapshot.endpoint_completed << ',' << snapshot.attempted_receipts << ','
           << snapshot.receipt_capacity << ',' << snapshot.arena_bytes
           << ",-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,no_receipt\n";
  for (const auto& receipt : snapshot.receipts) {
    stream << molecule.name << ',' << (device_input ? "device" : "host") << ',' << phase << ','
           << snapshot.call_id << ',' << snapshot.plan_token << ',' << snapshot.device_id << ','
           << snapshot.graph_family << ',' << snapshot.graph_submitted << ','
           << snapshot.endpoint_completed << ',' << snapshot.attempted_receipts << ','
           << snapshot.receipt_capacity << ',' << snapshot.arena_bytes << ',' << receipt.slot << ','
           << receipt.system << ',' << receipt.spin_channels << ',' << receipt.channel << ','
           << receipt.tile << ',' << receipt.orbital_count << ',' << receipt.pair_count << ','
           << receipt.pairs_visited << ',' << receipt.plain_visits << ',' << receipt.plain_completed
           << ',' << receipt.weighted_visits << ',' << receipt.weighted_completed << ','
           << receipt.failed_pairs << ',' << receipt.published_pairs << ',' << receipt.cta_cycles
           << ',' << receipt.status << '\n';
  }
  stream.flush();
  require(bool(stream), "density receipt CSV write failed");
}

void verify_density_receipts(const Molecule& water, bool device_input) {
  Gfn2CudaExecutionCache cache(0, nullptr);
  require(cache.enable_density_diagnostics(), "could not enable density receipts");
  const auto first = run(cache, water, device_input);
  oracle(first, water);
  Gfn2CudaDensityDiagnosticSnapshot snapshot;
  std::string error;
  require(cache.read_density_diagnostics(snapshot, error),
          "completed density diagnostic readback failed");
  require(snapshot.call_id == 1 && snapshot.plan_token != 0 && snapshot.device_id == 0 &&
              snapshot.batch_size == 1 && snapshot.graph_submitted && snapshot.endpoint_completed &&
              snapshot.graph_family != 0 && snapshot.grid_tiles > 0 &&
              snapshot.maximum_iterations == 300 &&
              snapshot.arena_bytes ==
                  sizeof(std::uint64_t) +
                      snapshot.receipt_capacity * sizeof(cuda::Gfn2DensityDeviceReceipt) &&
              snapshot.arena_bytes <= 64u * 1024u * 1024u,
          "density receipt identity or resource admission is incomplete");
  const auto slots_per_launch = 2u * snapshot.grid_tiles;
  require(snapshot.attempted_receipts == snapshot.receipts.size() &&
              snapshot.attempted_receipts >= slots_per_launch &&
              snapshot.attempted_receipts % slots_per_launch == 0 &&
              snapshot.attempted_receipts <= snapshot.receipt_capacity,
          "density receipts are missing or overflowed");
  const auto check_work = [&](const Gfn2CudaDensityDiagnosticSnapshot& current) {
    for (std::size_t begin = 0; begin < current.receipts.size(); begin += slots_per_launch) {
      std::uint64_t pairs = 0, plain = 0, weighted = 0, published = 0;
      std::int64_t orbitals = 0;
      std::uint32_t skipped_channels = 0;
      for (std::size_t index = begin; index < begin + slots_per_launch; ++index) {
        const auto& receipt = current.receipts[index];
        require(receipt.slot == index && receipt.system == 0 && receipt.tile < current.grid_tiles,
                "density CTA receipt lost its slot or topology identity");
        if (receipt.channel == 1u) {
          require(receipt.status == cuda::kDensityUnusedChannel && receipt.plain_visits == 0 &&
                      receipt.weighted_visits == 0,
                  "restricted density visited its inactive spin channel");
          ++skipped_channels;
          continue;
        }
        require(receipt.channel == 0u && receipt.spin_channels == 1u &&
                    receipt.status == cuda::kDensityContractCompleted &&
                    receipt.orbital_count > 0 && receipt.cta_cycles > 0 &&
                    receipt.failed_pairs == 0 && receipt.plain_visits == receipt.plain_completed &&
                    receipt.weighted_visits == receipt.weighted_completed,
                "healthy density CTA reported incomplete semantic work");
        if (orbitals == 0) orbitals = receipt.orbital_count;
        require(orbitals == receipt.orbital_count &&
                    receipt.pair_count == orbitals * (orbitals + 1) / 2,
                "density CTA dimensions disagree with the independent triangle");
        pairs += receipt.pairs_visited;
        plain += receipt.plain_visits;
        weighted += receipt.weighted_visits;
        published += receipt.published_pairs;
      }
      require(skipped_channels == current.grid_tiles && orbitals > 0 &&
                  pairs == static_cast<std::uint64_t>(orbitals * (orbitals + 1) / 2) &&
                  published == pairs && plain == pairs * static_cast<std::uint64_t>(orbitals) &&
                  weighted == plain,
              "actual density visits disagree with the independent healthy work oracle");
    }
  };
  check_work(snapshot);
  append_density_receipts(snapshot, water, device_input, "first");
  const auto replay = run(cache, water, device_input);
  oracle(replay, water);
  Gfn2CudaDensityDiagnosticSnapshot replay_snapshot;
  require(cache.read_density_diagnostics(replay_snapshot, error) && replay_snapshot.call_id == 2 &&
              replay_snapshot.plan_token == snapshot.plan_token &&
              replay_snapshot.graph_submitted && replay_snapshot.endpoint_completed &&
              replay_snapshot.attempted_receipts == snapshot.attempted_receipts,
          "density replay retained stale receipts or changed graph identity");
  check_work(replay_snapshot);
  append_density_receipts(replay_snapshot, water, device_input, "replay");
  require(!cache.enable_density_diagnostics(), "late density enablement changed graph binding");

  auto bad = water;
  bad.positions[0] = std::numeric_limits<double>::quiet_NaN();
  const auto replacement = run(cache, bad, device_input, true, 301);
  failed(replacement);
  Gfn2CudaDensityDiagnosticSnapshot replaced;
  require(
      cache.read_density_diagnostics(replaced, error) && replaced.call_id == 3 &&
          replaced.plan_token != snapshot.plan_token &&
          replaced.endpoint_completed == (replacement.status == GENERATIVEQC_XTB_STATUS_SUCCESS),
      "failed density candidate lost its independent receipt identity");
  append_density_receipts(replaced, water, device_input, "replacement_failed");
  oracle(run(cache, water, device_input), water);
  Gfn2CudaDensityDiagnosticSnapshot recovered;
  require(cache.read_density_diagnostics(recovered, error) && recovered.call_id == 4 &&
              recovered.plan_token == snapshot.plan_token && recovered.endpoint_completed,
          "failed density candidate published topology or broke old graph reuse");
  append_density_receipts(recovered, water, device_input, "recovered");
  failed(run(cache, water, device_input, true, 0));
  require(!cache.read_density_diagnostics(recovered, error),
          "pre-reset rejection exposed stale density receipts");
  std::uint64_t total_pairs = 0, total_plain = 0, total_weighted = 0;
  std::uint64_t maximum_cta_cycles = 0;
  for (const auto& receipt : snapshot.receipts) {
    total_pairs += receipt.pairs_visited;
    total_plain += receipt.plain_visits;
    total_weighted += receipt.weighted_visits;
    maximum_cta_cycles = std::max(maximum_cta_cycles, receipt.cta_cycles);
  }
  std::printf(
      "DENSITY_RECEIPT case=%s ingress=%s call=%llu graph=%u "
      "receipt_ctas=%llu capacity=%llu arena_bytes=%llu pairs=%llu "
      "plain_visits=%llu weighted_visits=%llu max_cta_cycles=%llu\n",
      water.name.c_str(), device_input ? "device" : "host",
      static_cast<unsigned long long>(snapshot.call_id), snapshot.graph_family,
      static_cast<unsigned long long>(snapshot.attempted_receipts),
      static_cast<unsigned long long>(snapshot.receipt_capacity),
      static_cast<unsigned long long>(snapshot.arena_bytes),
      static_cast<unsigned long long>(total_pairs), static_cast<unsigned long long>(total_plain),
      static_cast<unsigned long long>(total_weighted),
      static_cast<unsigned long long>(maximum_cta_cycles));

  Gfn2CudaExecutionCache failed_first_cache(0, nullptr);
  require(failed_first_cache.enable_density_diagnostics(),
          "could not enable first-failure density receipts");
  const auto failed_first = run(failed_first_cache, bad, device_input);
  failed(failed_first);
  Gfn2CudaDensityDiagnosticSnapshot failed_first_snapshot;
  require(failed_first_cache.read_density_diagnostics(failed_first_snapshot, error) &&
              failed_first_snapshot.call_id == 1 && failed_first_snapshot.plan_token != 0 &&
              failed_first_snapshot.endpoint_completed ==
                  (failed_first.status == GENERATIVEQC_XTB_STATUS_SUCCESS),
          "failed first density call lost its bounded snapshot");
  append_density_receipts(failed_first_snapshot, water, device_input, "first_failed");
}
#endif

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected independent fixture path");
    std::ifstream stream(argv[1]);
    int cases = 0;
    stream >> cases;
    require(cases > 0, "empty oracle fixture");
    std::vector<Molecule> molecules;
    for (int i = 0; i < cases; ++i) {
      Molecule molecule;
      int atoms = 0;
      stream >> molecule.name >> atoms >> molecule.energy;
      require(atoms > 0, "invalid oracle atom count");
      molecule.numbers.resize(atoms);
      molecule.positions.resize(3 * atoms);
      molecule.forces.resize(3 * atoms);
      for (int atom = 0; atom < atoms; ++atom) {
        stream >> molecule.numbers[atom];
        for (int k = 0; k < 3; ++k) stream >> molecule.positions[3 * atom + k];
        for (int k = 0; k < 3; ++k) stream >> molecule.forces[3 * atom + k];
      }
      require(bool(stream), "truncated oracle fixture");
      molecules.push_back(std::move(molecule));
    }
    check(cudaSetDevice(0));
    for (bool device_input : {false, true}) {
      for (const auto& molecule : molecules) verify_mixer_receipts(molecule, device_input);
      Gfn2CudaExecutionCache reused(0, nullptr);
      for (const auto& molecule : molecules) {
        Gfn2CudaExecutionCache cold(0, nullptr);
        const auto reference = run(cold, molecule, device_input);
        oracle(reference, molecule);
        const auto first = run(reused, molecule, device_input);
        oracle(first, molecule);
        require(first.iterations == reference.iterations, "rebuilt cache changed fresh SCC work");
        const auto repeated = run(reused, molecule, device_input);
        oracle(repeated, molecule);
        require(repeated.iterations == reference.iterations, "repeated call reused SCC state");
        oracle(run(reused, molecule, device_input, false), molecule);
        oracle(run(reused, molecule, device_input), molecule);
        verify_density_endpoint_timing(molecule, device_input);
#if defined(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
        if (molecule.name == "water-8" || molecule.name == "water-32")
          verify_density_receipts(molecule, device_input);
#endif
      }
      const auto found_water = std::find_if(molecules.begin(), molecules.end(),
                                            [](const Molecule& m) { return m.name == "h2o"; });
      require(found_water != molecules.end(), "missing water failure/recovery fixture");
      const auto& water = *found_water;
      verify_failed_mixer_receipts(water, device_input);
#if defined(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
      verify_density_receipts(water, device_input);
#endif
      for (int fault = 0; fault < 4; ++fault) {
        auto bad = water;
        if (fault == 0) bad.positions[0] = std::numeric_limits<double>::quiet_NaN();
        if (fault == 1) bad.positions[0] = 1e308;
        if (fault == 2)
          std::copy(bad.positions.begin(), bad.positions.begin() + 3, bad.positions.begin() + 3);
        if (fault == 3) bad.numbers[0] = 0;
        Gfn2CudaExecutionCache initially_bad(0, nullptr);
        failed(run(initially_bad, bad, device_input));
        oracle(run(initially_bad, water, device_input), water);
        // A failed replacement with another topology must also leave a prior
        // committed calculation usable. This is a different rollback path.
        oracle(run(reused, molecules.front(), device_input), molecules.front());
        failed(run(reused, bad, device_input));
        oracle(run(reused, molecules.front(), device_input), molecules.front());
        oracle(run(reused, water, device_input), water);
      }
      Gfn2CudaExecutionCache unconverged(0, nullptr);
      const auto terminal = run(unconverged, water, device_input, true, 1);
      require(terminal.status == GENERATIVEQC_XTB_STATUS_SUCCESS &&
                  terminal.statuses[0] == GENERATIVEQC_XTB_STATUS_SCC_NOT_CONVERGED,
              "first nonconverged call lost its terminal diagnostic");
      failed(terminal);
      oracle(run(unconverged, water, device_input), water);
    }
    std::puts(
        "CUDA bootstrap: first-call tblite, host/device ingress, failures and recovery passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA bootstrap qualification failed: %s\n", error.what());
    return 1;
  }
}
