#pragma once

#include "generativeqc/generativeqc.h"

extern "C" {
/** Prepared-item force admission; never infer capability from a method name.
 * This private synchronous query does no SCF/derivative work. Unqualified KS
 * items omit FORCES; a non-KS owner returns NOT_IMPLEMENTED.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_supported_properties_v1(
    const generativeqc_batch* batch, uint32_t index, generativeqc_property_flags* output);
}
