#include "generativeqc/generativeqc.h"
#include "scf/progressive_verification.hpp"

extern "C" generativeqc_status generativeqc_hf_verify_final_v1(
    const generativeqc_hf_verification_descriptor_v1* descriptor,
    generativeqc_hf_verification_result_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  using namespace generativeqc::scf::progressive;
  static_assert(fock_count_present == GENERATIVEQC_HF_VERIFY_FOCK_COUNT_PRESENT);
  const VerificationInput input{descriptor->flags,
                                descriptor->accuracy,
                                descriptor->iterations,
                                descriptor->iteration_limit,
                                descriptor->fock_builds,
                                descriptor->fock_limit,
                                descriptor->energy,
                                descriptor->energy_change,
                                descriptor->density_rms,
                                descriptor->residual_rms,
                                descriptor->audit_rms,
                                descriptor->audit_max,
                                descriptor->audit_energy_difference,
                                descriptor->energy_tolerance,
                                descriptor->density_tolerance,
                                descriptor->residual_tolerance};
  if (!valid(input)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto decision = verify(input);
  // Read all borrowed fields before publication, including for aliased storage.
  *output = {sizeof(*output),         GENERATIVEQC_ABI_VERSION, decision.status,
             decision.target_reasons, decision.budget_reasons,  decision.cleanup,
             decision.accuracy};
  return GENERATIVEQC_STATUS_SUCCESS;
}
