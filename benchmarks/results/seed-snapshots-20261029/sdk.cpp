// Installed public-header warm-seed lifecycle; no Python or private C++ headers.
#include <generativeqc/warm_seed.h>

#include <array>
#include <cmath>
#include <cstring>
#include <source_location>
#include <stdexcept>
#include <string>

void require_test(bool condition, std::source_location location = std::source_location::current()) {
  if (!condition)
    throw std::runtime_error(std::string("SDK contract failed: ") + location.file_name() + ":" +
                             std::to_string(location.line()));
}

int main(int argc, char**) {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = argc > 1 ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc_context* context{};
  require_test(generativeqc_context_create(&descriptor, &context) == 0);
  std::array<generativeqc_atom, 3> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}, {2, 0, 0, 0}}};
  std::array<generativeqc_system*, 2> systems{};
  for (unsigned i = 0; i != 2; ++i) {
    generativeqc_system_descriptor system{};
    system.struct_size = sizeof(system);
    system.abi_version = GENERATIVEQC_ABI_VERSION;
    system.atoms = atoms.data() + (i ? 2 : 0);
    system.atom_count = i ? 1 : 2;
    system.multiplicity = 1;
    require_test(
        generativeqc_system_create_bundled(context, &system, "sto-3g", nullptr, &systems[i]) == 0);
  }
  std::array<const generativeqc_system*, 2> inputs{systems[0], systems[1]};
  std::array<generativeqc_warm_seed_snapshot*, 4> snapshots{};
  std::array<double, 4> source_energies{};
  unsigned model_index = 0;
  for (const char* model : {"rhf", "uhf", "pbe-rks", "pbe-uks"}) {
    const bool unrestricted = std::strcmp(model, "uhf") == 0 || std::strcmp(model, "pbe-uks") == 0;
    generativeqc_method_descriptor method{
        sizeof(method),
        GENERATIVEQC_ABI_VERSION,
        unrestricted ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF,
        100,
        8,
        1e-10,
        1e-8,
        1e-12};
    const auto prepare = [&] {
      generativeqc_batch* batch{};
      if (model[0] == 'p') {
        method.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
        generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION, model,
                                               nullptr, 0};
        require_test(generativeqc_batch_prepare_named_ks_v1(
                         context, inputs.data(), inputs.size(), &method, &named,
                         GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, &batch) == 0);
      } else {
        require_test(generativeqc_batch_prepare(context, inputs.data(), inputs.size(), &method,
                                                GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                                &batch) == 0);
      }
      return batch;
    };
    const auto execute = [](generativeqc_batch* batch) {
      std::array<generativeqc_batch_item_result_descriptor, 2> result{};
      for (auto& item : result) {
        item.struct_size = sizeof(item);
        item.abi_version = GENERATIVEQC_ABI_VERSION;
      }
      require_test(generativeqc_batch_execute(batch, nullptr, 0, result.data(), result.size()) ==
                   0);
      for (const auto& item : result) require_test(item.status == 0 && item.converged);
      return result;
    };
    auto* source = prepare();
    generativeqc_warm_seed_capacity_v1 capacity{sizeof(capacity), GENERATIVEQC_ABI_VERSION};
    require_test(generativeqc_batch_query_warm_seed_capacity_v1(source, &capacity) == 0);
    generativeqc_warm_seed_snapshot_info_v1 info{sizeof(info), GENERATIVEQC_ABI_VERSION};
    require_test(generativeqc_batch_capture_warm_seeds_v1(source, capacity.total_bytes - 1, &info,
                                                          &snapshots[model_index]) ==
                 GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    require_test(!snapshots[model_index]);
    const auto expected = execute(source);
    source_energies[model_index] = expected[0].energy;
    require_test(generativeqc_batch_capture_warm_seeds_v1(source, capacity.total_bytes, &info,
                                                          &snapshots[model_index]) == 0);
    require_test(info.numeric_bytes == capacity.total_bytes && info.system_count == 2);
    generativeqc_batch_destroy(source);
    for (int replay = 0; replay != 2; ++replay) {
      auto* target = prepare();
      require_test(generativeqc_batch_restore_warm_seeds_v1(target, snapshots[model_index]) == 0);
      require_test(generativeqc_batch_set_warm_start_updates(target, 0) == 0);
      const auto actual = execute(target);
      for (unsigned i = 0; i != actual.size(); ++i)
        require_test(std::abs(actual[i].energy - expected[i].energy) < 2e-9);
      generativeqc_batch_destroy(target);
    }
    ++model_index;
  }
  for (auto* system : systems) generativeqc_system_destroy(system);
  generativeqc_context_destroy(context);
  // Captured arrays remain valid after every source/runtime owner is destroyed.
  for (unsigned i = 0; i != snapshots.size(); ++i) {
    generativeqc_warm_seed_view_v1 view{sizeof(view), GENERATIVEQC_ABI_VERSION};
    require_test(generativeqc_warm_seed_snapshot_read_v1(snapshots[i], 0, &view) == 0);
    require_test(view.present && view.energy == source_energies[i]);
    generativeqc_warm_seed_snapshot_destroy(snapshots[i]);
  }
}
