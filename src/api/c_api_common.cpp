#include <algorithm>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "methods/generated_method_manifest.hpp"
#include "methods/method.hpp"
#include "scf/progressive_hf.hpp"

extern "C" {

/** Private additive HF execution controller. The callback is synchronous and
 * borrowed for this call only; missing-symbol compatibility stays explicit. */
GENERATIVEQC_API generativeqc_status
generativeqc_progressive_hf_run_v1(generativeqc::scf::ProgressiveHFCallback callback, void* state,
                                   uint64_t* receipt, size_t count) {
  if (count != 2) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return generativeqc::scf::run_progressive_hf(callback, state, receipt);
}

uint32_t generativeqc_ks_options_version(void) { return 1; }

/** Private Python bridge: distinguish support for the optional size-tagged
 * precision extension from the unchanged scientific KS descriptor version. */
GENERATIVEQC_API uint32_t generativeqc_application_precision_version_v1(void) { return 1; }

/** Method-family-neutral budget transport, independent of SCF arithmetic. */
GENERATIVEQC_API uint32_t generativeqc_application_precision_version_v2(void) { return 2; }

/** Private capability: budget-selected full-range FP32 J/K and physical gates. */
GENERATIVEQC_API uint32_t generativeqc_application_fock_precision_version_v1(void) { return 1; }

GENERATIVEQC_API uint32_t generativeqc_application_fock_precision_version_v2(void) { return 2; }

/** Optional Python provenance transport, independent of the stable SCF ABI.
 * A missing/computation-only result must not inherit a previous force record. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_gradient_precision_v1(
    const generativeqc_calculation* calculation, uint64_t* output, size_t count) {
  if (!calculation || !output || count != 5) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(calculation->context->mutex);
  if (!calculation->precision_work || !calculation->precision_work->gradient[0])
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  if (calculation->precision_work->gradient_qualification > 1)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  std::copy_n(calculation->precision_work->gradient.begin(), count, output);
  return GENERATIVEQC_STATUS_SUCCESS;
}

GENERATIVEQC_API generativeqc_status generativeqc_batch_gradient_precision_v1(
    const generativeqc_batch* batch, uint32_t index, uint64_t* output, size_t count) {
  if (!batch || index >= batch->precision_work.size() || !output || count != 5)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  if (!batch->precision_work[index] || !batch->precision_work[index]->gradient[0])
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  if (batch->precision_work[index]->gradient_qualification > 1)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  std::copy_n(batch->precision_work[index]->gradient.begin(), count, output);
  return GENERATIVEQC_STATUS_SUCCESS;
}

/** V2 appends the model qualification ID. The v1 getter cannot misidentify a
 * newer model as its historical PBE0 qualification. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_gradient_precision_v2(
    const generativeqc_calculation* calculation, uint64_t* output, size_t count) {
  if (!calculation || !output || count != 6) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(calculation->context->mutex);
  if (!calculation->precision_work || !calculation->precision_work->gradient[0] ||
      !calculation->precision_work->gradient_qualification)
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  std::copy_n(calculation->precision_work->gradient.begin(), 5, output);
  output[5] = calculation->precision_work->gradient_qualification;
  return GENERATIVEQC_STATUS_SUCCESS;
}

GENERATIVEQC_API generativeqc_status generativeqc_batch_gradient_precision_v2(
    const generativeqc_batch* batch, uint32_t index, uint64_t* output, size_t count) {
  if (!batch || index >= batch->precision_work.size() || !output || count != 6)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  if (!batch->precision_work[index] || !batch->precision_work[index]->gradient[0] ||
      !batch->precision_work[index]->gradient_qualification)
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  std::copy_n(batch->precision_work[index]->gradient.begin(), 5, output);
  output[5] = batch->precision_work[index]->gradient_qualification;
  return GENERATIVEQC_STATUS_SUCCESS;
}

/** V3 adds a collection bit and per-item weighted task/source/retry counts.
 * Legacy model IDs retain their meaning; zero counts without the collection
 * bit are unavailable evidence, not proof that no weighted work executed. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_gradient_precision_v3(
    const generativeqc_calculation* calculation, uint64_t* output, size_t count) {
  if (!calculation || !output || count != 10) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(calculation->context->mutex);
  const auto status = generativeqc_calculation_gradient_precision_v2(calculation, output, 6);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto& work = *calculation->precision_work;
  output[6] = work.gradient_weighted_work_collected;
  std::copy(work.gradient_weighted_work.begin(), work.gradient_weighted_work.end(), output + 7);
  return GENERATIVEQC_STATUS_SUCCESS;
}

GENERATIVEQC_API generativeqc_status generativeqc_batch_gradient_precision_v3(
    const generativeqc_batch* batch, uint32_t index, uint64_t* output, size_t count) {
  if (!batch || !output || count != 10) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  const auto status = generativeqc_batch_gradient_precision_v2(batch, index, output, 6);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto& work = *batch->precision_work[index];
  output[6] = work.gradient_weighted_work_collected;
  std::copy(work.gradient_weighted_work.begin(), work.gradient_weighted_work.end(), output + 7);
  return GENERATIVEQC_STATUS_SUCCESS;
}

uint32_t generativeqc_get_abi_version(void) { return GENERATIVEQC_ABI_VERSION; }

const char* generativeqc_status_message(generativeqc_status status) {
  switch (status) {
    case GENERATIVEQC_STATUS_SUCCESS:
      return "success";
    case GENERATIVEQC_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case GENERATIVEQC_STATUS_ABI_MISMATCH:
      return "ABI mismatch";
    case GENERATIVEQC_STATUS_NOT_IMPLEMENTED:
      return "requested capability is not implemented";
    case GENERATIVEQC_STATUS_NOT_CONVERGED:
      return "SCF did not converge";
    case GENERATIVEQC_STATUS_NUMERICAL_FAILURE:
      return "numerical failure";
    case GENERATIVEQC_STATUS_CUDA_ERROR:
      return "CUDA runtime error";
    case GENERATIVEQC_STATUS_OUT_OF_MEMORY:
      return "out of memory";
    case GENERATIVEQC_STATUS_INTERNAL_ERROR:
      return "internal error";
    case GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE:
      return "precision provenance not yet populated";
  }
  return "unknown status";
}

generativeqc_status generativeqc_method_from_name(const char* canonical_name,
                                                  generativeqc_method* output) {
  if (canonical_name == nullptr || output == nullptr || *canonical_name == '\0')
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto* entry = generativeqc::methods::generated::find_method(canonical_name);
  if (entry == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *output = entry->method;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_method_get_name(generativeqc_method method, const char** output) {
  if (output == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto* entry = generativeqc::methods::generated::find_method(method);
  if (entry == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *output = entry->name.data();
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_method_available(generativeqc_method method, int32_t* available) {
  if (available == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const generativeqc::methods::Capabilities* capabilities =
      generativeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *available = capabilities->available ? 1 : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_method_get_capabilities(
    generativeqc_method method, generativeqc_method_capabilities_descriptor* output) {
  if (!generativeqc::api::valid_descriptor(output)) {
    return output == nullptr ? GENERATIVEQC_STATUS_INVALID_ARGUMENT
                             : GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  const generativeqc::methods::Capabilities* capabilities =
      generativeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  output->method = capabilities->method;
  output->family = capabilities->family;
  output->supported_properties = capabilities->supported_properties;
  output->available = capabilities->available ? 1 : 0;
  output->supports_batch = capabilities->supports_batch ? 1 : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // extern "C"
