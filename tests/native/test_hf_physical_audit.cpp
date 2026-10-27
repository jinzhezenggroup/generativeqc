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

/** Installed SDK contract: no Python, scientific oracle or compiler is loaded. */
void audit_sdk() {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(descriptor);
  std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0.1, 0, 0.7}}};
  generativeqc_system_descriptor model{};
  model.struct_size = sizeof(model);
  model.abi_version = GENERATIVEQC_ABI_VERSION;
  model.atoms = atoms.data();
  model.atom_count = atoms.size();
  model.multiplicity = 1;
  model.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  generativeqc_system* raw{};
  require(generativeqc_system_create_bundled(context.get(), &model, "sto-3g", nullptr, &raw) == 0,
          "native bundled source creation failed");
  std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> system(
      raw, generativeqc_system_destroy);
  for (const auto fitted : {0u, 1u}) {
    for (const auto unrestricted : {0u, 1u}) {
      generativeqc_hf_physical_audit_controls_v1 controls{sizeof(controls),
                                                          GENERATIVEQC_ABI_VERSION,
                                                          unrestricted,
                                                          fitted,
                                                          8,
                                                          0,
                                                          1u << 28,
                                                          1e-14,
                                                          1e-10};
      generativeqc_hf_physical_audit_capacity_v1 capacity{};
      capacity.struct_size = sizeof(capacity);
      capacity.abi_version = GENERATIVEQC_ABI_VERSION;
      require(generativeqc_hf_physical_audit_cpu_query_v1(system.get(), nullptr, &controls,
                                                          &capacity) == 0,
              "native audit capacity query failed");
      require(capacity.required_host_bytes == capacity.provider_bytes + capacity.workspace_bytes,
              "inconsistent native numeric bound");
      std::vector<double> density(capacity.density_count, 0.1);
      std::vector<double> overlap(capacity.overlap_count, -19);
      generativeqc_hf_physical_audit_result_v1 result{};
      result.struct_size = sizeof(result);
      result.abi_version = GENERATIVEQC_ABI_VERSION;
      result.overlap = overlap.data();
      result.overlap_capacity = overlap.size();
      controls.maximum_host_bytes = capacity.required_host_bytes;
      require(
          generativeqc_hf_physical_audit_cpu_v1(context.get(), system.get(), nullptr, &controls,
                                                density.data(), density.size(), -1, &result) == 0,
          "exact-budget native audit failed");
      require(result.fock_builds == 1 && std::isfinite(result.energy) &&
                  std::isfinite(result.maximum_commutator) && std::isfinite(result.rms_commutator),
              "native audit published incomplete observations");
      std::array<unsigned char, sizeof(result)> before{};
      std::memcpy(before.data(), &result, sizeof(result));
      const auto saved_overlap = overlap;
      --controls.maximum_host_bytes;
      require(generativeqc_hf_physical_audit_cpu_v1(context.get(), system.get(), nullptr, &controls,
                                                    density.data(), density.size(), -1,
                                                    &result) == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
              "one-byte-short audit was not refused");
      require(std::memcmp(before.data(), &result, sizeof(result)) == 0 && overlap == saved_overlap,
              "refused audit modified detached output");
    }
  }
}
}  // namespace

int main() {
  try {
    audit_sdk();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
