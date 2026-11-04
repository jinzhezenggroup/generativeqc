#pragma once

#include <cstdint>
#include <utility>

#include "generativeqc/precision_admission.h"

namespace generativeqc::runtime {

/** Own the bounded candidate/auditor execution order, not numerical admission
 * or observable measurement. Callbacks preserve their prepared/candidate leases
 * and release transient auditor owners themselves, including on failure. */
inline generativeqc_status run_precision_execution(
    const generativeqc_precision_execution_context_v1& context,
    generativeqc_precision_execution_callback_v1 callback, void* state,
    generativeqc_precision_execution_receipt_v1& receipt) noexcept {
  receipt = {
      sizeof(receipt), GENERATIVEQC_ABI_VERSION, 1, 0, 0, GENERATIVEQC_PRECISION_EXECUTION_FAILED};
  generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
  auto invoke = [&](std::uint32_t event, bool decision) -> std::pair<int, std::uint32_t> {
    const auto bit = std::uint32_t{1} << event;
    receipt.attempted_events |= bit;
    std::uint32_t result{};
    int callback_status{};
    try {
      callback_status = callback(state, event, &result);
    } catch (...) {
      callback_status = GENERATIVEQC_PRECISION_CALLBACK_ERROR;
    }
    if (callback_status != GENERATIVEQC_PRECISION_CALLBACK_SUCCESS) {
      status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
      return {callback_status, 0};
    }
    if (result > (decision ? 1u : 0u)) {
      status = GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      return {GENERATIVEQC_PRECISION_CALLBACK_ERROR, 0};
    }
    receipt.completed_events |= bit;
    return {GENERATIVEQC_PRECISION_CALLBACK_SUCCESS, result};
  };

  const auto candidate = invoke(GENERATIVEQC_PRECISION_EXECUTE_CANDIDATE, false);
  if (candidate.first == GENERATIVEQC_PRECISION_CALLBACK_RETRY && context.retry_candidate) {
    status = GENERATIVEQC_STATUS_SUCCESS;
    invoke(GENERATIVEQC_PRECISION_EXECUTE_FP64_FALLBACK, false);
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto classification = invoke(GENERATIVEQC_PRECISION_CLASSIFY_CANDIDATE, true);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  if (classification.second) {
    receipt.outcome = GENERATIVEQC_PRECISION_EXECUTION_QUALIFIED;
    return status;
  }
  invoke(GENERATIVEQC_PRECISION_RELEASE_CANDIDATE, false);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto audit = invoke(GENERATIVEQC_PRECISION_EXECUTE_AUDIT, true);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  if (audit.second) {
    receipt.outcome = GENERATIVEQC_PRECISION_EXECUTION_AUDITED;
    return status;
  }
  const auto verification = invoke(GENERATIVEQC_PRECISION_VERIFY_FALLBACK, true);
  if (status == GENERATIVEQC_STATUS_SUCCESS)
    receipt.outcome = verification.second ? GENERATIVEQC_PRECISION_EXECUTION_VERIFIED
                                          : GENERATIVEQC_PRECISION_EXECUTION_REJECTED;
  return status;
}

}  // namespace generativeqc::runtime
