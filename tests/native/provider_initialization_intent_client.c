#include <stddef.h>
#include <stdint.h>

#if __has_include("generativeqc/provider_initialization.h")
#include "generativeqc/provider_initialization.h"

_Static_assert(sizeof(generativeqc_provider_initialization_intent_v1) == 24, "intent v1 ABI size");
_Static_assert(offsetof(generativeqc_provider_initialization_intent_v1, struct_size) == 0 &&
                   offsetof(generativeqc_provider_initialization_intent_v1, abi_version) == 4 &&
                   offsetof(generativeqc_provider_initialization_intent_v1, mode) == 8 &&
                   offsetof(generativeqc_provider_initialization_intent_v1, reserved) == 12,
               "intent v1 prefix offsets");
_Static_assert(offsetof(generativeqc_provider_initialization_intent_v1, allowance_bytes) == 16,
               "intent v1 allowance offset");

int intent_c_header_available(void) { return 1; }
size_t intent_c_size(void) { return sizeof(generativeqc_provider_initialization_intent_v1); }
size_t intent_c_allowance_offset(void) {
  return offsetof(generativeqc_provider_initialization_intent_v1, allowance_bytes);
}
generativeqc_status intent_c_prepare(generativeqc_context* context,
                                     const generativeqc_system* system,
                                     const generativeqc_method_descriptor* method,
                                     const generativeqc_provider_initialization_intent_v1* intent,
                                     generativeqc_calculation** calculation) {
  return generativeqc_calculation_prepare_with_provider_initialization_v1(context, system, method,
                                                                          intent, calculation);
}
#else
int intent_c_header_available(void) { return 0; }
#endif
