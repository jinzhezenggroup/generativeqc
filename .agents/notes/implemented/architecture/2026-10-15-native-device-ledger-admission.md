# Decision: bind selected device capacity to its native ledger

Status: implemented (host metadata/lifecycle qualification; real-GPU gate separate)
Date: 2026-10-15

## Problem

Global native resource composition did not remove the second Python capacity
decision in `NativeDeviceLedger`: choosing one visible device, filtering opaque
runtime allowances, then sweeping numeric lifetimes before creating a native
ledger. A correctly admitted plan could therefore still depend on a separate
portable implementation immediately before persistent allocation enforcement.

Runtime allowances must not become capacity for explicit tracked buffers. In
particular, `peak(all) - peak(allowances)` is not `peak(numeric)` when their
lifetimes differ. The explicit numeric limit needs its own sweep. An
allowance-only inventory legitimately admits zero numeric bytes. A large valid
portable device label must not silently wrap through a C `int` during execution.

## Decision

`runtime/resource_ledger_plan.*` derives both full declared and numeric peaks
using the existing native sparse accounting owner. Device labels are mapped to
small metadata slots, not used as allocation extents. The single-device
prepared-owner contract remains explicit; missing/multiple devices do not
invent a device, and ordinals must fit the actual native device parameter.

The additive private `generativeqc_resource_ledger_create_from_buffers_v1`
validates immutable, disjoint capacity records and publishes a v1-compatible
shared ledger plus a three-word receipt only after all work succeeds. It creates
no CUDA context, scientific provider or numeric buffer. Existing observation
binding, provider reservation, allocation journal and retained shared lifetime
remain the enforcement owners; this is not a competing allocator framework.

The Python adapter marshals the already-selected inventory, validates the
native receipt and only then publishes the handle. Malformed receipts release
unpublished metadata. Native rejection cannot invoke legacy creation or portable
peak derivation. Missing symbols retain the explicit old-library path, including
the same zero-numeric and device-representability safety rules.

## Boundaries and evidence

The caller must already have admitted the global plan. This bridge does not
authenticate model/topology identity, reserve a whole global budget, change
provider lease generations, or establish complete host/device/RSS receipts.
Opaque runtime/library allocations remain unobserved allowances. No new CUDA
scientific capability or numerical promotion is implied.

Native host-only tests cover noncoincident peaks, sparse maximum phases,
allowance-only and multi-device domains, integer overflow, alias/adjacent
buffers, transactional failure, compatible observation binding, exact/below-cap
provider reservations, handle closure and final release on another thread.
Python compares deterministic ragged inventories against the independent
portable model and checks explicit choices, compatibility, rejection and cleanup.
The production allocation wrapper also runs against host CUDA allocation doubles
with the new constructor: retained/warm/reset, async/free, wrong-device,
allocation/registry failure and exact shortage retain the existing counters.
Those doubles are not real-GPU allocation, sanitizer or endpoint timing evidence.

The real-GPU resource suite retains its explicit finite Slurm admission and adds
native-capacity receipt checks. It must run on an allowed available GPU before
claiming the corresponding CUDA endpoint gate. Node n3 is maintenance-only;
absence of another local scheduler resource is not permission to bypass Slurm.

### Source-bound comparison protocol (2026-10-16)

The complete portable plan identity includes the selected native library's
SHA256 in each request's schedule. Independently built baseline and candidate
libraries therefore need not have equal opaque plan identities. A cross-build
comparison must first verify each full plan's canonical identity and its binding
to the actually loaded library, then normalize only that verified binary hash.
Profile identity, scientific identity, inventories, selected providers, limits,
phases and all other schedule fields remain strict comparison inputs. Ignoring
the whole identity would hide real semantic differences.

Some authorized Slurm hosts use no task-affinity plugin. The process's inherited
affinity can then include CPUs outside its allocation; its minimum CPU is not a
valid placement rule. Use the job's recorded `CPU_IDs`, intersect with allowed
affinity, and record both allocation and actual pinning. Keep failed sequential
measurements rather than relabelling them as passes. A complete per-case ABBA
comparison, with identical warmup/sample counts and unchanged per-endpoint
acceptance gates, is a separate qualification, not proof that placement caused
an earlier timing failure or a causal speedup claim.

## Revisit when

Global native plans gain authenticated prepared leases and complete allocation
journals. Preserve allowance versus numeric separation and the single scientific
owner's retained lifetime rather than inferring physical bytes from declared caps.

## References

- #2227 R2/R6; #933 / #934
- `2026-10-15-native-global-resource-composition.md`
- `docs/maintainer/resource_planning.md`
