#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "generativeqc/generativeqc.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Record>
Record versioned() {
  Record record{};
  record.struct_size = sizeof(record);
  record.abi_version = GENERATIVEQC_ABI_VERSION;
  return record;
}

using System = std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)>;
using Batch = std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)>;

bool seeded(const Batch& batch) {
  auto state = versioned<generativeqc_hf_warm_state>();
  require(generativeqc_batch_get_hf_warm_state(batch.get(), 0, &state) == 0, "seed query failed");
  return state.present;
}

generativeqc_batch_item_result_descriptor execute(const Batch& batch) {
  auto result = versioned<generativeqc_batch_item_result_descriptor>();
  require(generativeqc_batch_execute(batch.get(), nullptr, 0, &result, 1) == 0 && !result.status,
          "SDK HF solve failed");
  return result;
}

/** All three fleets deliberately share one context; transfers must not lock
 * the same mutex as two distinct owners or depend on Python metadata tags. */
void run(bool unrestricted) {
  auto descriptor = versioned<generativeqc_context_descriptor>();
  descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(descriptor);
  const generativeqc_atom atom{unrestricted ? 1 : 2, 0, 0, 0};
  auto model = versioned<generativeqc_system_descriptor>();
  model.atoms = &atom;
  model.atom_count = 1;
  model.multiplicity = unrestricted ? 2 : 1;
  model.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  const auto system = [&](const char* basis) {
    generativeqc_system* pointer{};
    require(
        generativeqc_system_create_bundled(context.get(), &model, basis, nullptr, &pointer) == 0,
        "SDK system preparation failed");
    return System(pointer, generativeqc_system_destroy);
  };
  auto source_model = system("sto-3g"), target_model = system("def2-svp");
  auto method = generativeqc::default_method_descriptor(unrestricted ? "uhf" : "rhf");
  const auto prepare = [&](const System& selected) {
    const generativeqc_system* pointer = selected.get();
    generativeqc_batch* batch{};
    require(generativeqc_batch_prepare(context.get(), &pointer, 1, &method,
                                       GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, &batch) == 0,
            "SDK fleet preparation failed");
    return Batch(batch, generativeqc_batch_destroy);
  };
  auto source = prepare(source_model), target = prepare(target_model), next = prepare(target_model);
  const auto source_result = execute(source);
  auto controls = versioned<generativeqc_hf_fleet_projection_controls_v1>();
  controls.strict = 1;
  controls.maximum_ao = 4096;
  controls.maximum_host_bytes = 1;
  controls.relative_threshold = 1e-10;
  controls.validation_tolerance = 1e-7;
  controls.maximum_residual = 0.5;
  auto capacity = versioned<generativeqc_hf_fleet_projection_capacity_v1>();
  require(generativeqc_hf_fleet_projection_query_v1(target.get(), source.get(), &controls, nullptr,
                                                    0, &capacity) == 0,
          "below-budget metadata query failed");
  require(capacity.numeric_peak_bytes > 1 && capacity.retained_bytes > 0,
          "native query omitted live numeric storage");
  auto output = versioned<generativeqc_hf_fleet_projection_item_v1>();
  output.accepted = 99;
  const auto preserved = output;
  const auto transfer = [&](const Batch& destination, const Batch& origin) {
    return generativeqc_hf_fleet_initialize_cpu_v1(destination.get(), origin.get(), &controls,
                                                   nullptr, 0, &output, 1);
  };
  controls.maximum_host_bytes = capacity.numeric_peak_bytes - 1;
  require(transfer(target, source) == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
          "below-capacity seed transfer was admitted");
  require(!seeded(target) && seeded(source) && !std::memcmp(&output, &preserved, sizeof(output)),
          "failed transfer partially published state or diagnostics");
  ++controls.maximum_host_bytes;
  require(!transfer(target, source) && output.accepted && seeded(target),
          "exact-capacity native transfer failed");
  require(
      output.source_energy == source_result.energy && output.spin_count == (unrestricted ? 2 : 1),
      "transfer lost source/spin provenance");
  require(std::abs(output.spins[0].electron_trace - (unrestricted ? 1.0 : 2.0)) < 1e-9,
          "native projection changed electron count");
  if (unrestricted)
    require(output.spins[1].occupied_rank == 0, "empty beta spin was not preserved");
  controls.maximum_host_bytes = 1u << 28;
  require(transfer(next, target) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !seeded(next),
          "unexecuted proposal was promoted to a successful source");
  const auto target_result = execute(target);
  require(target_result.warm_start_used, "installed projection did not seed target SCF");
  require(!transfer(next, target) && seeded(next), "executed target did not become eligible");
  const auto replay = execute(next);
  require(std::abs(replay.energy - target_result.energy) < 1e-10,
          "native transfer/replay changed the target equations");
}
}  // namespace

int main() {
  try {
    run(false);
    run(true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
