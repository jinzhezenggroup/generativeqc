#include <cstdint>
#include <cstring>

#include "generativeqc/generativeqc.h"

// Host-only identity stub. It has no CUDA dependency and must never be used
// as numerical evidence; tests install it only into their temporary cache.
extern "C" generativeqc_status generativeqc_cuda_tuning_device(
    std::int32_t ordinal, generativeqc_cuda_tuning_device_descriptor* output) {
  if (!output || ordinal != 0 || output->struct_size != sizeof(*output) ||
      output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  generativeqc_cuda_tuning_device_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  std::strcpy(descriptor.name, "synthetic-only");
  std::strcpy(descriptor.official_profile, "synthetic_test_only");
  descriptor.major = 12;
  descriptor.runtime_version = 12090;
  descriptor.driver_version = 13000;
  descriptor.toolkit_version = 12090;
  descriptor.release_build = 1;
  *output = descriptor;
  return GENERATIVEQC_STATUS_SUCCESS;
}

extern "C" const char* generativeqc_get_source_identity() {
  return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
}

// Admit the synthetic small topology so pure DF scalar planning can be compared
// with the independent Python inventory, without any CUDA allocation/probe.
extern "C" int generativeqc_resource_small_hf_cuda_v2(std::size_t nbf, std::size_t, std::size_t,
                                                      std::size_t, const std::uint8_t*,
                                                      const std::size_t*, std::size_t, std::size_t,
                                                      int, double, double, std::uint64_t* output,
                                                      std::size_t count) {
  if (!output || count != 2 || nbf > 16) return 1;
  output[0] = 0;
  output[1] = 0;
  return 0;
}
