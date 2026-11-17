#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>

#include "scf/aot_shell_registry.hpp"

#if GENERATIVEQC_TEST_AOT_SHELLS
#include "generativeqc_generated_shell_registry.hpp"
#endif

namespace generated = generativeqc::scf::generated;

namespace {

std::uint64_t metadata_mask(const generated::ShellKernelMetadata* kernels, std::size_t count) {
  std::uint64_t mask = 0;
  for (std::size_t index = 0; index < count; ++index) {
    mask |= std::uint64_t{1} << kernels[index].shell_class;
  }
  return mask;
}

}  // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 1;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 2;

  generated::select_profile_for_device(0, properties.major, properties.minor);
  const auto& profile = generated::selected_profile();
  const auto& force_policy = generated::selected_force_policy();
  const char* status = generated::selected_registry_status();
  const auto force_mask = generated::enabled_shell_class_mask();
  const auto fock_mask = generated::enabled_fock_shell_class_mask();
  std::cout << "sm_" << properties.major << properties.minor << ' ' << profile.name
            << " tuned=" << profile.tuned << " portable=" << profile.portable
            << " compatible=" << profile.compatible << " registry=" << status
            << " force_mask=" << force_mask << " fock_mask=" << fock_mask
            << " angular_queue=" << force_policy.small_work_angular_queue
            << " strict_coverage=" << force_policy.require_bounded_spd_coverage << '\n';

  const generated::ProfileInfo* expected = nullptr;
  std::uint64_t compiled_force_mask = 0;
  std::uint64_t compiled_fock_mask = 0;
#if GENERATIVEQC_TEST_AOT_SHELLS
  // The manifest records build intent, including forced portable schedules on
  // measured architectures. Device identity alone cannot recover that intent.
  for (const auto& compiled : generated::kCompiledProfiles) {
    if (compiled.compute_capability_major == properties.major &&
        compiled.compute_capability_minor == properties.minor) {
      if (expected != nullptr) return 3;
      expected = &compiled;
    }
  }
  if (expected != nullptr) {
    for (const auto& kernel : generated::kCompiledShellKernels) {
      if (std::strcmp(kernel.target_architecture, expected->target_architecture) != 0) continue;
      if (kernel.consumer_mask & 1U) compiled_fock_mask |= std::uint64_t{1} << kernel.shell_class;
      if (kernel.consumer_mask & 2U) compiled_force_mask |= std::uint64_t{1} << kernel.shell_class;
    }
  }
  const char* fallback_status = "no_matching_kernel_set";
#else
  const char* fallback_status = "aot_disabled";
#endif

  if (expected == nullptr) {
    if (std::strcmp(profile.name, "generic_cuda") != 0 || profile.tuned || !profile.portable ||
        profile.compatible || std::strcmp(status, fallback_status) != 0) {
      return 4;
    }
  } else if (std::strcmp(profile.name, expected->name) != 0 ||
             std::strcmp(profile.target_architecture, expected->target_architecture) != 0 ||
             profile.compute_capability_major != properties.major ||
             profile.compute_capability_minor != properties.minor ||
             profile.tuned != expected->tuned || profile.portable != expected->portable ||
             profile.compatible != expected->compatible || std::strcmp(status, "matched") != 0) {
    // This also rejects a tuned schedule borrowed from an incompatible SM.
    return 5;
  }

  std::size_t force_count = 0;
  std::size_t fock_count = 0;
  const auto* force_kernels = generated::selected_shell_kernels(force_count);
  const auto* fock_kernels = generated::selected_fock_shell_kernels(fock_count);
  if (metadata_mask(force_kernels, force_count) != compiled_force_mask ||
      metadata_mask(fock_kernels, fock_count) != compiled_fock_mask) {
    return 6;
  }
  // Environment selection may disable generated classes, but cannot invent
  // coverage absent from the selected compiled consumer inventory.
  if ((force_mask & ~compiled_force_mask) != 0 || (fock_mask & ~compiled_fock_mask) != 0) return 7;
#if defined(GENERATIVEQC_TEST_MOCK_DEVICES)
  // The host discovery stub provides an unmatched second device. Real CTest
  // jobs may expose only one GPU, or a second GPU with the same compiled SM.
  if (cudaSetDevice(1) != cudaSuccess) return 8;
  const auto& fallback_policy = generated::selected_force_policy();
  if (!fallback_policy.small_work_angular_queue || fallback_policy.require_bounded_spd_coverage)
    return 9;
  if (cudaSetDevice(0) != cudaSuccess) return 10;
  const auto& restored = generated::selected_force_policy();
  if (restored.small_work_angular_queue != force_policy.small_work_angular_queue ||
      restored.require_bounded_spd_coverage != force_policy.require_bounded_spd_coverage)
    return 11;
#endif
  return 0;
}
