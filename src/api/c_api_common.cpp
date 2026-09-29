#include "api/error.hpp"
#include "generativeqc/generativeqc.h"
#include "methods/method.hpp"

extern "C" {

uint32_t generativeqc_ks_options_version(void) { return 1; }

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
