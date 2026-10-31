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

typedef struct generativeqc_warm_seed_snapshot generativeqc_warm_seed_snapshot;

/** Actual retained numeric storage, excluding metadata and restore scratch. */
typedef struct generativeqc_warm_seed_snapshot_info_v1 {
  uint32_t struct_size, abi_version, system_count, reserved;
  uint64_t numeric_bytes, largest_density_bytes;
} generativeqc_warm_seed_snapshot_info_v1;

/** Immutable borrowed arrays. A missing slot has null pointers and zero counts. */
typedef struct generativeqc_warm_seed_view_v1 {
  uint32_t struct_size, abi_version;
  const double* density;
  uint64_t density_count;
  const double* coordinates;
  uint64_t coordinate_count;
  double energy, energy_change, density_rms;
  int32_t iterations, present;
} generativeqc_warm_seed_view_v1;

/** @native-contract generativeqc_batch_capture_warm_seeds_v1
 * @behavior Copy the supplying solver's scientific warm seeds into native host storage.
 * @inputs A live warm-enabled batch, numeric byte limit, initialized info and output handle.
 * @outputs On success writes a new immutable snapshot and its actual numeric storage.
 * @errors INVALID_ARGUMENT for null inputs or a non-warm batch; ABI_MISMATCH for info;
 * NOT_IMPLEMENTED for absent seed support; OUT_OF_MEMORY for overflow, a limit below
 * the all-slot envelope, or allocation failure. Both outputs are unchanged on failure.
 * @lifetime Caller destroys the snapshot; it may outlive the source batch and context.
 * @execution Checks the full envelope before materializing any seed. Existing owner
 * export may perform D2H copies. No integrals or SCF execute; source seeds are unchanged.
 * Serialize source execution/destruction with this call. Metadata/allocator overhead
 * and scientific validation scratch are outside the numeric limit.
 * @units The limit and receipt are bytes; stored coordinates are Bohr and energy is Eh.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_capture_warm_seeds_v1(
    const generativeqc_batch* batch, uint64_t maximum_numeric_bytes,
    generativeqc_warm_seed_snapshot_info_v1* info, generativeqc_warm_seed_snapshot** output);

/** @native-contract generativeqc_batch_restore_warm_seeds_v1
 * @behavior Restore immutable captured seeds through the target's existing scientific validator.
 * @inputs A live warm-enabled target batch and snapshot from the same ordered model/topology.
 * The caller must preserve model, basis, charge and spin identity, as with the descriptor
 * restore API. This handle is not a cross-model projection or capability token.
 * @outputs Present slots replace target seeds transactionally; absent slots are unchanged.
 * @errors INVALID_ARGUMENT for null inputs, shape or physical validation failures;
 * NOT_IMPLEMENTED for absent seed support; allocation/provider errors propagate.
 * @lifetime The snapshot remains reusable and independent of both batches/contexts.
 * @execution Synchronous target lock and retained resource-ledger binding; existing
 * overlap/occupation validation and provider-scratch admission are retained.
 * Does not execute target SCF or change the target's warm-update policy. Serialize target
 * execution/destruction and keep the snapshot alive throughout the call.
 * @units Native binary64 seed conventions; Bohr coordinates and Eh source diagnostics.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_restore_warm_seeds_v1(
    generativeqc_batch* batch, const generativeqc_warm_seed_snapshot* snapshot);

/** @native-contract generativeqc_warm_seed_snapshot_read_v1
 * @behavior Inspect one immutable snapshot slot without copying numeric arrays.
 * @inputs A live snapshot, original input index and initialized versioned view.
 * @outputs Transactionally publishes borrowed read-only arrays and source diagnostics.
 * @errors INVALID_ARGUMENT for null inputs or invalid index; ABI_MISMATCH for view.
 * @lifetime Borrowed pointers expire when the snapshot is destroyed; never free or mutate them.
 * @execution Allocation-free host metadata read. Concurrent reads are safe; destruction
 * must be externally serialized. No source batch, context or device is accessed.
 * @units Density is row-major RHF total or UHF alpha then beta; coordinates Bohr, energy Eh.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_warm_seed_snapshot_read_v1(const generativeqc_warm_seed_snapshot* snapshot,
                                        uint32_t index, generativeqc_warm_seed_view_v1* view);

/** @native-contract generativeqc_warm_seed_snapshot_destroy
 * @behavior Release a captured seed snapshot and its native host arrays.
 * @inputs A snapshot returned by capture, or NULL.
 * @outputs None; all borrowed views become invalid.
 * @errors No error return; NULL is a no-op.
 * @lifetime Call exactly once for a non-NULL owned handle, after all readers finish.
 * @execution Synchronous host-only destruction; requires no live context or GPU.
 * @units Not applicable.
 */
GENERATIVEQC_API void generativeqc_warm_seed_snapshot_destroy(
    generativeqc_warm_seed_snapshot* snapshot);

#ifdef __cplusplus
}
#endif

#endif
