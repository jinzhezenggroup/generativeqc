// Installed public-header probe; basis/model expansion occurs inside the SDK.
#include <generativeqc/warm_seed.h>

#include <array>
#include <cassert>
#include <cstring>

int main(int argc, char**) {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = argc > 1 ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc_context* context{};
  assert(generativeqc_context_create(&descriptor, &context) == 0);
  std::array<generativeqc_atom, 3> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}, {2, 0, 0, 0}}};
  std::array<generativeqc_system*, 2> systems{};
  for (unsigned i = 0; i != 2; ++i) {
    generativeqc_system_descriptor system{};
    system.struct_size = sizeof(system);
    system.abi_version = GENERATIVEQC_ABI_VERSION;
    system.atoms = atoms.data() + (i ? 2 : 0);
    system.atom_count = i ? 1 : 2;
    system.multiplicity = 1;
    assert(generativeqc_system_create_bundled(context, &system, "sto-3g", nullptr, &systems[i]) ==
           0);
  }
  std::array<const generativeqc_system*, 2> inputs{systems[0], systems[1]};
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
    generativeqc_batch* batch{};
    if (model[0] == 'p') {
      method.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
      generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION, model,
                                             nullptr, 0};
      assert(generativeqc_batch_prepare_named_ks_v1(context, inputs.data(), inputs.size(), &method,
                                                    &named, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                                    &batch) == 0);
    } else {
      assert(generativeqc_batch_prepare(context, inputs.data(), inputs.size(), &method,
                                        GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, &batch) == 0);
    }
    generativeqc_warm_seed_capacity_v1 capacity{sizeof(capacity), GENERATIVEQC_ABI_VERSION};
    assert(generativeqc_batch_query_warm_seed_capacity_v1(batch, &capacity) == 0);
    const unsigned spins = unrestricted ? 2 : 1;
    assert(capacity.density_bytes == 40 * spins && capacity.coordinate_bytes == 72);
    assert(capacity.total_bytes == 40 * spins + 72 && capacity.largest_density_bytes == 32 * spins);
    generativeqc_batch_destroy(batch);
  }
  for (auto* system : systems) generativeqc_system_destroy(system);
  generativeqc_context_destroy(context);
}
