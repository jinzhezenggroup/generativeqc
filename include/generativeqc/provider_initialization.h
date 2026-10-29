#ifndef GENERATIVEQC_PROVIDER_INITIALIZATION_H
#define GENERATIVEQC_PROVIDER_INITIALIZATION_H

#include "generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  GENERATIVEQC_PROVIDER_INITIALIZATION_LEGACY_DEFAULT = 0,
  GENERATIVEQC_PROVIDER_INITIALIZATION_PENDING = 1
};

enum {
  GENERATIVEQC_PROVIDER_INITIALIZATION_MP2_CUDA = 1u << 0,
  GENERATIVEQC_PROVIDER_INITIALIZATION_RCCSD_CUDA = 1u << 1
};

/** Immutable preparation intent, copied by value during prepare. Version 1
 * carries only a finite allowance: no caller identity or qualification fields.
 * Larger v1 records may append ignored bytes; reserved must be zero.
 * LEGACY_DEFAULT requires allowance_bytes=0 and preserves the ordinary path.
 * PENDING requires 0 < allowance_bytes <= INT64_MAX, representable by size_t.
 * A is a reservation, not a measured peak, allocation or admission receipt. */
typedef struct generativeqc_provider_initialization_intent_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t mode;
  uint32_t reserved;
  uint64_t allowance_bytes;
} generativeqc_provider_initialization_intent_v1;

/** @native-contract generativeqc_provider_initialization_intent_version
 * @behavior Report the supported standalone intent ABI version.
 * @inputs None.
 * @outputs Returns 1.
 * @errors None.
 * @lifetime Retains no caller storage.
 * @execution Synchronous host metadata; no device or provider observation.
 * @units Dimensionless version. */
GENERATIVEQC_API uint32_t generativeqc_provider_initialization_intent_version(void);

/** @native-contract generativeqc_provider_initialization_capabilities_v1
 * @behavior Report compiled software transport support, never qualification.
 * @inputs None.
 * @outputs Returns supported scalar conventional CUDA MP2/RCCSD bits.
 * @errors Unsupported builds return zero.
 * @lifetime Retains no caller storage.
 * @execution Synchronous host metadata; no context creation or CUDA query.
 * @units Dimensionless bit mask. */
GENERATIVEQC_API uint32_t generativeqc_provider_initialization_capabilities_v1(void);

/** @native-contract generativeqc_calculation_prepare_with_provider_initialization_v1
 * @behavior Snapshot pending intent and delegate to the existing private
 * candidate/publisher. NULL or a valid LEGACY_DEFAULT intent uses ordinary
 * preparation. PENDING is limited to scalar restricted conventional CUDA
 * MP2/RCCSD physical references; unsupported domains reject before publication.
 * Actual identity is captured later by the native owner in its established
 * context, never supplied or certified by this descriptor.
 * @inputs Valid context/system/method/output plus an optional complete v1 intent.
 * @outputs SUCCESS publishes one calculation through the existing publisher.
 * With otherwise valid pointer arguments, failure leaves *calculation NULL.
 * @errors Bad intent header/version returns ABI_MISMATCH; malformed controls
 * return INVALID_ARGUMENT; unsupported domain/build returns NOT_IMPLEMENTED.
 * Existing preparation errors and accounting-basis refusals propagate.
 * @lifetime Intent storage need only outlive this call; plans own copied values.
 * The complete intent is snapshotted before modifying calculation output.
 * @execution Synchronous preparation; this transport performs no identity probe,
 * context create/reset/switch, numerical solve or independent publication.
 * @units allowance_bytes is a byte reservation, charged by the native owner. */
GENERATIVEQC_API generativeqc_status
generativeqc_calculation_prepare_with_provider_initialization_v1(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* method,
    const generativeqc_provider_initialization_intent_v1* intent,
    generativeqc_calculation** calculation);

/** @native-contract generativeqc_context_close_provider_initialization_v1
 * @behavior Close a retained explicit provider owner before deleting SDK handles.
 * A context with no explicit owner succeeds without a provider observation.
 * @inputs A valid live context; no caller identity or qualification metadata.
 * @outputs SUCCESS confirms owner closure but does not delete a published
 * calculation or context. It reclaims an unpublished failed candidate only
 * after its complete close is confirmed.
 * @errors NULL returns INVALID_ARGUMENT. Unavailable or failed owner close
 * preserves its native status/detail and retains the original owner and budget.
 * @lifetime Failed close retains the context, plan, pins and original allowance.
 * The caller keeps these handles alive until confirmation; void destroy refuses
 * deletion of unconfirmed explicit owners.
 * @execution Synchronous native close in the original ambient context. Never
 * creates, selects, retains or resets a context. An attempted uncertain physical
 * close is latched by the physical owner and cannot be retried as cleanup.
 * @units Dimensionless native status; no allocation or peak is inferred. */
GENERATIVEQC_API generativeqc_status
generativeqc_context_close_provider_initialization_v1(generativeqc_context* context);

#ifdef __cplusplus
}
#endif

#endif
