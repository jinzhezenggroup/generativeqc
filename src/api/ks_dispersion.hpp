#pragma once

#include "generativeqc/generativeqc.h"

extern "C" {
/** Private exact-identity discovery. NULL name means the library has no complete
 * matching program, not permission to substitute a similar functional. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_catalog_name_v1(const char* identity,
                                                                     generativeqc_backend backend,
                                                                     const char** name);
/** Read completed correction outputs from the last successful item transaction.
 * Optional buffers copy only existing work; a missing gradient is NOT_IMPLEMENTED.
 * D3 uses two_body_energy for its complete correction, including ATM if enabled. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_dispersion_result_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d4_batch_item_result_descriptor*);
GENERATIVEQC_API const char* generativeqc_ks_batch_dispersion_variant_v1(const generativeqc_batch*,
                                                                         uint32_t);
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_d3_diagnostic_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d3_runtime_diagnostic*);
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_d4_diagnostic_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d4_runtime_diagnostic*);
}
