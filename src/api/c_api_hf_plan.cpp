#include "generativeqc/generativeqc.h"
#include "scf/progressive_plan.hpp"

extern "C" generativeqc_status generativeqc_hf_plan_query_v1(
    const generativeqc_hf_plan_descriptor_v1* descriptor, generativeqc_hf_plan_result_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!generativeqc::scf::progressive::valid_plan(*descriptor))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto failures = generativeqc::scf::progressive::plan_failures(*descriptor);
  // Publish only after consuming all input, also for overlapping caller storage.
  *output = {sizeof(*output), GENERATIVEQC_ABI_VERSION, failures, 0};
  return GENERATIVEQC_STATUS_SUCCESS;
}
