#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc_hf_verification_descriptor_v1 observed() {
  generativeqc_hf_verification_descriptor_v1 input{};
  input.struct_size = sizeof(input);
  input.abi_version = GENERATIVEQC_ABI_VERSION;
  input.flags =
      GENERATIVEQC_HF_VERIFY_MODEL_MATCHES | GENERATIVEQC_HF_VERIFY_PROVIDER_MATCHES |
      GENERATIVEQC_HF_VERIFY_HASHES_MATCH | GENERATIVEQC_HF_VERIFY_SUCCEEDED |
      GENERATIVEQC_HF_VERIFY_CONVERGED | GENERATIVEQC_HF_VERIFY_RESIDUAL_PRESENT |
      GENERATIVEQC_HF_VERIFY_OBSERVABLES_COMPLETE | GENERATIVEQC_HF_VERIFY_PRECISION_MATCHES |
      GENERATIVEQC_HF_VERIFY_FINAL_FP64 | GENERATIVEQC_HF_VERIFY_ACCURACY_PRESENT |
      GENERATIVEQC_HF_VERIFY_ACCURACY_TARGET_MATCHES | GENERATIVEQC_HF_VERIFY_FOCK_COUNT_PRESENT;
  input.accuracy = 4;
  input.iterations = input.iteration_limit = 120;
  input.energy = -1.1;
  input.energy_change = input.energy_tolerance = 1e-10;
  input.density_rms = input.density_tolerance = 1e-9;
  input.residual_rms = input.residual_tolerance = 1e-8;
  return input;
}

generativeqc_hf_verification_result_v1 decide(
    const generativeqc_hf_verification_descriptor_v1& input) {
  generativeqc_hf_verification_result_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  require(generativeqc_hf_verify_final_v1(&input, &result) == GENERATIVEQC_STATUS_SUCCESS,
          "valid observations were rejected");
  return result;
}

void run() {
  auto input = observed();
  auto output = decide(input);
  require(output.status == 1 && output.target_reasons == 0 && output.budget_reasons == 0 &&
              output.cleanup == 3 && output.accuracy == 4,
          "inclusive physical gates did not verify independent observed accuracy");
  for (auto accuracy : {0u, 5u, 6u}) {
    input.accuracy = accuracy;
    require(decide(input).status == 0, "an estimate or unknown accuracy was certified");
  }
  for (auto accuracy : {1u, 2u, 3u}) {
    input.accuracy = accuracy;
    output = decide(input);
    require(output.status == 2 && output.target_reasons == 0,
            "unmet accuracy erased the separately established physical target");
  }
  input = observed();
  input.flags |= GENERATIVEQC_HF_VERIFY_AUTOMATIC_PRECISION;
  output = decide(input);
  require(output.status == 2 && output.cleanup == 1 && output.target_reasons == (1u << 15),
          "automatic precision omitted strict closure");
  input.flags |= GENERATIVEQC_HF_VERIFY_STRICT_REFINEMENT;
  require(decide(input).status == 1 && decide(input).cleanup == 2,
          "strict closure was not accepted");
  input.flags |= GENERATIVEQC_HF_VERIFY_AUDIT_PRESENT | GENERATIVEQC_HF_VERIFY_AUDIT_TARGET_MATCHES;
  input.audit_rms = 1e-12;
  input.audit_max = 2e-8;
  output = decide(input);
  require(output.status == 2 && output.target_reasons == (1u << 11),
          "small audit RMS hid the failing maximum commutator");
  input.audit_max = input.residual_tolerance;
  input.residual_rms = std::numeric_limits<double>::quiet_NaN();
  require(decide(input).status == 1, "present physical audit did not supersede native RMS");
  input.audit_energy_difference = 2e-10;
  input.iteration_limit = 119;
  input.flags |=
      GENERATIVEQC_HF_VERIFY_AUDIT_BUDGET_EXHAUSTED | GENERATIVEQC_HF_VERIFY_FOCK_LIMIT_PRESENT;
  input.flags &= ~GENERATIVEQC_HF_VERIFY_FOCK_COUNT_PRESENT;
  output = decide(input);
  require(output.status == 3 && output.target_reasons == (1u << 8) && output.budget_reasons == 7,
          "budget precedence erased physical failures or missing work evidence");
  input = observed();
  input.energy = std::numeric_limits<double>::infinity();
  input.energy_change = std::numeric_limits<double>::quiet_NaN();
  input.density_rms = std::numeric_limits<double>::infinity();
  input.residual_rms = std::numeric_limits<double>::quiet_NaN();
  require(decide(input).target_reasons == ((1u << 4) | (1u << 5) | (1u << 6) | (1u << 10)),
          "nonfinite actual measurements were not retained as physical failures");
  input = observed();
  input.iterations = input.iteration_limit = std::numeric_limits<std::uint64_t>::max();
  require(decide(input).status == 1, "full-width exact work metadata was truncated");
  for (int error = 0; error < 4; ++error) {
    auto bad = observed();
    if (error == 0) bad.flags |= 1u << 31;
    if (error == 1) bad.accuracy = 7;
    if (error == 2) bad.residual_tolerance = 0;
    if (error == 3) bad.abi_version += 1;
    output = decide(observed());
    const auto preserved = output;
    const auto status = generativeqc_hf_verify_final_v1(&bad, &output);
    require(status == (error == 3 ? GENERATIVEQC_STATUS_ABI_MISMATCH
                                  : GENERATIVEQC_STATUS_INVALID_ARGUMENT),
            "malformed final-verification request was admitted");
    require(std::memcmp(&preserved, &output, sizeof(output)) == 0, "failed call published output");
  }
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
