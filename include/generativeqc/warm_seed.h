#ifndef GENERATIVEQC_WARM_SEED_H
#define GENERATIVEQC_WARM_SEED_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** All-slot numeric seed envelope, independent of current seed presence.
 * This excludes descriptors, allocator overhead, validation scratch and provider
 * storage; it is not a complete host/device resource plan. */
typedef struct generativeqc_warm_seed_capacity_v1 {
  uint32_t struct_size, abi_version;
  uint64_t density_bytes, coordinate_bytes, total_bytes, largest_density_bytes;
} generativeqc_warm_seed_capacity_v1;

/** @native-contract generativeqc_batch_query_warm_seed_capacity_v1
 * @behavior Query the prepared owner's numeric warm-seed staging envelope.
 * @inputs A live batch and initialized versioned output descriptor.
 * @outputs Transactionally writes all-slot density/coordinate/total bytes and the
 * largest density buffer. Includes cold/missing slots and both spins where needed.
 * @errors INVALID_ARGUMENT for null inputs; ABI_MISMATCH for descriptor headers;
 * NOT_IMPLEMENTED for owners without warm seeds; OUT_OF_MEMORY for byte overflow.
 * Failures leave output unchanged. Other owner errors propagate without fallback.
 * @lifetime The batch/context must remain alive. No pointer or seed is retained.
 * @execution Synchronous locked topology query. It does not inspect/materialize a
 * seed, transfer device data, execute science or allocate density buffers. Serialize
 * with execution/destruction as for other operations on the same batch/context.
 * @units All sizes are bytes of binary64 densities and Bohr coordinates.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_query_warm_seed_capacity_v1(
    const generativeqc_batch* batch, generativeqc_warm_seed_capacity_v1* output);

#ifdef __cplusplus
}
#endif

#endif
