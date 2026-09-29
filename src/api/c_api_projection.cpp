#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "integrals/s_integrals.hpp"

extern "C" generativeqc_status generativeqc_system_cross_overlap_cpu(
    generativeqc_context* context, const generativeqc_system* target,
    const generativeqc_system* source, double* output, size_t output_count) {
  if (context == nullptr || target == nullptr || source == nullptr || output == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    generativeqc::integrals::cross_overlap(target->data, source->data, {output, output_count});
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}
