// Test-only complete endpoint journal. Production bindings and arithmetic stay
// in execute_restricted_gfn2_cuda; readback happens after synchronous settlement.
#ifndef GENERATIVEQC_TEST_GFN2_MIXER_ENDPOINT_PROBE_HPP
#define GENERATIVEQC_TEST_GFN2_MIXER_ENDPOINT_PROBE_HPP

#include <iomanip>
#include <memory>
#include <sstream>

namespace mixer_probe_detail {
constexpr std::size_t kJournalLimit = 64u * 1024u * 1024u;

void number(std::ostream& out, double value) {
  if (std::isfinite(value))
    out << std::setprecision(17) << value;
  else
    out << "null";
}

void vector(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    number(out, values[i]);
  }
  out << ']';
}

void append(std::ofstream& out, std::size_t& bytes, const std::string& row) {
  require(row.size() + 1 <= kJournalLimit - bytes, "mixer probe journal exceeds 64 MiB");
  out << row << '\n';
  out.flush();  // Preserve settled rows even if a later call fails.
  require(bool(out), "could not retain mixer probe row");
  bytes += row.size() + 1;
}

Result sample(std::ofstream& out, std::size_t& bytes, Gfn2CudaExecutionCache& cache,
              const Molecule& molecule, std::size_t case_index, bool enabled, const char* phase,
              int repetition, int order, double construction_seconds,
              int maximum_iterations = 300) {
  const auto start = std::chrono::steady_clock::now();
  const auto result = run(cache, molecule, false, true, maximum_iterations);
  const auto stop = std::chrono::steady_clock::now();
  Gfn2CudaMixerDiagnosticSnapshot snapshot;
  std::string error;
  const bool available = enabled && cache.read_mixer_diagnostics(snapshot, error);
  const auto read_stop = std::chrono::steady_clock::now();
  const auto seconds = [](auto first, auto last) {
    return std::chrono::duration<double>(last - first).count();
  };
  std::ostringstream row;
  row << std::setprecision(17) << "{\"schema\":1,\"case_index\":" << case_index
      << ",\"enabled\":" << (enabled ? "true" : "false") << ",\"phase\":\"" << phase
      << "\",\"repetition\":" << repetition << ",\"order\":" << order
      << ",\"maximum_iterations\":" << maximum_iterations
      << ",\"construction_seconds\":" << construction_seconds
      << ",\"endpoint_seconds\":" << seconds(start, stop)
      << ",\"readback_seconds\":" << seconds(stop, read_stop) << ",\"status\":" << result.status
      << ",\"system_status\":" << result.statuses[0]
      << ",\"converged\":" << int(result.converged[0]) << ",\"iterations\":" << result.iterations[0]
      << ",\"flags\":" << result.flags << ",\"energy\":";
  number(row, result.energy[0]);
  row << ",\"forces\":";
  vector(row, result.forces);
  row << ",\"positions\":";
  vector(row, molecule.positions);
  row << ",\"numbers\":[";
  for (std::size_t atom = 0; atom < molecule.numbers.size(); ++atom) {
    if (atom) row << ',';
    row << molecule.numbers[atom];
  }
  row << ']';
  row << ",\"snapshot_available\":" << (available ? "true" : "false")
      << ",\"receipt_bytes\":" << sizeof(cuda::Gfn2SccMixerDeviceReceipt)
      << ",\"call_id\":" << snapshot.call_id << ",\"plan_token\":" << snapshot.plan_token
      << ",\"device_id\":" << snapshot.device_id << ",\"graph_family\":" << snapshot.graph_family
      << ",\"graph_submitted\":" << (snapshot.graph_submitted ? "true" : "false")
      << ",\"endpoint_completed\":" << (snapshot.endpoint_completed ? "true" : "false")
      << ",\"attempted_receipts\":" << snapshot.attempted_receipts << ",\"receipts\":[";
  for (std::size_t i = 0; i < snapshot.receipts.size(); ++i) {
    if (i) row << ',';
    const auto& r = snapshot.receipts[i];
    row << "{\"invocation\":" << r.invocation << ",\"system\":" << r.system
        << ",\"iteration_before\":" << r.iteration_before
        << ",\"restart_before\":" << r.restart_before
        << ",\"vector_elements\":" << r.vector_elements << ",\"live_history\":" << r.live_history
        << ",\"new_slot\":" << r.new_slot << ",\"completed_stages\":" << r.completed_stages
        << ",\"status\":" << r.status
        << ",\"coefficient_dot_elements\":" << r.coefficient_dot_elements
        << ",\"gram_dot_elements\":" << r.gram_dot_elements
        << ",\"combination_elements\":" << r.combination_elements
        << ",\"residual_elapsed_cycles\":" << r.residual_elapsed_cycles
        << ",\"history_elapsed_cycles\":" << r.history_elapsed_cycles
        << ",\"solve_elapsed_cycles\":" << r.solve_elapsed_cycles
        << ",\"combination_elapsed_cycles\":" << r.combination_elapsed_cycles << '}';
  }
  row << "]}";
  append(out, bytes, row.str());
  return result;
}
}  // namespace mixer_probe_detail

int mixer_endpoint_probe(const std::vector<Molecule>& molecules, const char* path,
                         const char* repetitions_text) {
  std::size_t consumed = 0;
  const int repetitions = std::stoi(repetitions_text, &consumed);
  require(consumed == std::string(repetitions_text).size() && repetitions >= 3 && repetitions <= 20,
          "probe repetitions must be between 3 and 20");
  require(molecules.size() <= 32 && molecules.size() % 2 == 0,
          "probe requires bounded original/moved fixture pairs");
  // nvidia-smi may list physical peers outside CUDA_VISIBLE_DEVICES. Bind this
  // probe to the device actually addressed by its cache, using the runtime API.
  cudaDeviceProp properties{};
  int visible_devices = 0, driver_version = 0, runtime_version = 0;
  check(cudaGetDeviceProperties(&properties, 0));
  check(cudaGetDeviceCount(&visible_devices));
  check(cudaDriverGetVersion(&driver_version));
  check(cudaRuntimeGetVersion(&runtime_version));
  std::ofstream device(std::string(path) + ".device.json");
  require(bool(device), "could not open probe device identity");
  device << "{\"selected_device\":0,\"visible_devices\":" << visible_devices
         << ",\"driver_version\":" << driver_version << ",\"runtime_version\":" << runtime_version
         << ",\"name\":" << std::quoted(properties.name)
         << ",\"compute_major\":" << properties.major << ",\"compute_minor\":" << properties.minor
         << ",\"total_global_memory\":" << properties.totalGlobalMem << ",\"uuid_hex\":\"";
  for (unsigned char byte : properties.uuid.bytes)
    device << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(byte);
  device << "\"}\n";
  device.close();
  require(bool(device), "could not retain probe device identity");
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  require(bool(out), "could not open mixer probe journal");
  std::size_t bytes = 0;
  for (std::size_t index = 0; index < molecules.size(); index += 2) {
    require(molecules[index].numbers == molecules[index + 1].numbers &&
                molecules[index].positions != molecules[index + 1].positions,
            "probe fixture pair must retain topology and move geometry");
    std::unique_ptr<Gfn2CudaExecutionCache> caches[2];
    double construction[2]{};
    for (int arm = 0; arm < 2; ++arm) {
      const auto start = std::chrono::steady_clock::now();
      caches[arm] = std::make_unique<Gfn2CudaExecutionCache>(0, nullptr);
      if (arm) require(caches[arm]->enable_mixer_diagnostics(), "probe mixer enable failed");
      construction[arm] =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    for (int geometry = 0; geometry < 2; ++geometry) {
      for (int repetition = 0; repetition <= repetitions; ++repetition) {
        const char* phase =
            geometry ? (repetition ? "moved_replay" : "moved") : (repetition ? "warm" : "cold");
        for (int order = 0; order < 2; ++order) {
          const int arm = (order + repetition + geometry + int(index / 2)) % 2;
          mixer_probe_detail::sample(out, bytes, *caches[arm], molecules[index + geometry],
                                     index + geometry, arm != 0, phase, repetition, order,
                                     geometry == 0 && repetition == 0 ? construction[arm] : 0.0);
        }
      }
    }
  }
  // Retain the existing transaction scenarios as raw readout, rather than
  // adding a different failure mechanism to the production owner.
  const auto water = std::find_if(molecules.begin(), molecules.end(),
                                  [](const Molecule& molecule) { return molecule.name == "h2o"; });
  require(water != molecules.end(), "missing water lifecycle fixture");
  const auto water_index = static_cast<std::size_t>(water - molecules.begin());
  std::ofstream failures(std::string(path) + ".failures.jsonl");
  require(bool(failures), "could not open lifecycle journal");
  Gfn2CudaExecutionCache lifecycle(0, nullptr);
  require(lifecycle.enable_mixer_diagnostics(), "could not enable lifecycle receipts");
  auto bad = *water;
  bad.positions[0] = std::numeric_limits<double>::quiet_NaN();
  failed(mixer_probe_detail::sample(failures, bytes, lifecycle, bad, water_index, true,
                                    "first_failed", 0, 0, 0.0));
  oracle(mixer_probe_detail::sample(failures, bytes, lifecycle, *water, water_index, true,
                                    "recovered", 0, 0, 0.0),
         *water);
  failed(mixer_probe_detail::sample(failures, bytes, lifecycle, bad, water_index, true,
                                    "replacement_failed", 0, 0, 0.0, 301));
  oracle(mixer_probe_detail::sample(failures, bytes, lifecycle, *water, water_index, true, "replay",
                                    0, 0, 0.0),
         *water);
  failed(mixer_probe_detail::sample(failures, bytes, lifecycle, *water, water_index, true,
                                    "rejected", 0, 0, 0.0, 0));
  oracle(mixer_probe_detail::sample(failures, bytes, lifecycle, *water, water_index, true,
                                    "after_rejection", 0, 0, 0.0),
         *water);
  return 0;
}
#endif
