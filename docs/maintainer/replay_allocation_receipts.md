# Prepared replay allocation receipts

`tools/audit_replay_allocations.py` consumes runtime allocation journals and
checks explicit zero-new-allocation assertions. It installs no allocator,
changes no production execution policy, and does not upgrade the
[advisory source-work audit](source_work_audit.md) into runtime evidence.

```sh
python tools/audit_replay_allocations.py receipt.json --expected expected.json
python tools/audit_replay_allocations.py ledger.json --native-ledger-v1
python tools/audit_replay_allocations.py ledger.json --native-ledger-v2
```

Exit zero means the strict receipt passed for its **declared ownership domains**.
`FAIL` and `INCOMPLETE` both exit one. PASS is not a process-wide allocation
guarantee, numerical acceptance, performance promotion, or CUDA qualification.
The consumer checks internal consistency and equality to a separately pinned
contract; it cannot authenticate a producer's claimed observation coverage or
source/build identity. The evidence reviewer must establish those claims from
the source-matched build, capture mechanism and complete endpoint execution.
Never derive the expected contract by copying the unreviewed receipt.

## Existing prepared endpoint and measurement owner

The first target is direct RHF
`Calculator.prepare_batch(...).execute(properties=("energy",), strict=True)`:
prepare the request, execute the cold energy route, then execute the same warm
route. Geometry, workload, schedule and prepared owner must stay fixed between
cold and warm executions. Publication and any geometry rebuild need separate
windows; their allocations cannot be charged to, or hidden inside, warm replay.

Enable resource observation with an explicit `ResourceBudget` or accepted
resource plan on the CUDA calculator/batch. The existing owner is `NativeDeviceLedger` in
`python/generativeqc/resources_native.py`, bound by `CpuResourceObservation`.
It exports `batch.resource_diagnostics["observation"]["device_ledger"]`.
Its native owner is `src/runtime/resource_ledger.hpp`, with allocation bookkeeping
in `src/runtime/resource_cuda.cuh` and C ABI reads in
`src/api/c_api_resources.cpp`. Binding resets allocation count and peak to
retained live ownership. These are successful owned CUDA buffer allocations;
driver, graph, pool and library internals are excluded. The peak includes
retained storage and is not the sum of newly requested bytes.

`--native-ledger-v1` accepts the exact four-counter `NativeDeviceLedger.to_dict()` shape.
It preserves allocation count, live/peak bytes, owner, visible device and
rejected allocations. It reports requested bytes as `null` and returns
`INCOMPLETE`, including when allocation count is zero. V1 lacks a cumulative
requested-byte counter, allocation/release journal, phase/source/build identity
and complete host observation. CPU boundary samples are also incomplete and
cannot fill those fields. Missing data is never inferred from live/peak bytes.

Libraries exporting `generativeqc_resource_ledger_read_v2` additionally report
`requested_bytes`: the cumulative bytes of successful registered owned-buffer
requests in the current observation binding. Frees do not subtract from this
counter; budget/driver rejection and failed registry insertion do not add to it.
Each binding resets it, along with allocation/rejection counts, while preserving
retained live storage and resetting peak to that storage. Thus requested bytes
can exceed both peak and the resource limit when storage is reused over time.
The v2 read takes one locked five-counter snapshot and rejects byte-counter
overflow without changing numerical allocation policy. V1's four-counter ABI
and exact legacy Python export remain available for older native libraries.

Use `--native-ledger-v2` for the exact extended export. The verifier preserves
the real requested bytes but still returns `INCOMPLETE`: there is no event
journal, pinned phase/source/build identity or complete host observation in
either native snapshot. A zero count and zero requested bytes prove no new
successful allocation only within the owned-device ledger's observation scope,
not a complete host/device zero-allocation endpoint.

`tests/python/test_hf_resources_cuda.py` already contains a warm owned-device
allocation-count assertion and cold/warm numerical comparisons. The GPU test
also checks zero newly requested owned-device bytes on warm energy/force replay.
The host-double regression in `tests/python/test_device_ledger_requested_bytes.py`
compiles the actual ledger C API and allocation wrappers without a GPU; it covers
retained ownership, synchronous/asynchronous reuse, failure accounting, v1 ABI
compatibility, binding reset and overflow. These are bookkeeping gates, not
real-GPU qualification. The receipt verifier
does not claim to have executed that GPU test or to have qualified a complete
host/device zero-allocation pathway. Completing that gate requires a capture
adapter integrated with the existing measurement owner, not a second allocator.

## Native bounded journals and scoped v2 assertions

`NativeDeviceJournal` in `python/generativeqc/resources_native.py` optionally
attaches to an existing `NativeDeviceLedger` between synchronous observation
scopes. It snapshots retained allocation generations as initial live owners
and reserves a finite number of event slots before replay. Successful registry
insertion and successful retirement/free append genuine allocation/release
events under the same native registry lock. Address reuse preserves generation
identity; delayed old-generation frees cannot retire a new allocation.
Recording continues across ledger-handle closure while native buffers remain
owned. Closing the journal stops observation, not scientific allocation.

Capacity is explicitly bounded to 1–1,048,576 events. Slot exhaustion increments
`dropped_events` without growing storage, adding CUDA calls or changing the
allocation policy. A dropped event prevents complete-coverage certification.
Journal metadata and snapshot buffers are profiling overhead, not a new
scientific resource-plan allowance or a claim that host replay is allocation-free.

`tools/replay_allocation_capture.py` supplies bounded journal/window adapters.
`JournalWindows` carries actual owners and contiguous event cursors through
every declared phase and rejects overwritten history, missing events and lost
coverage. The caller must establish synchronized capture boundaries and actual
source-matched execution independently; the adapter cannot authenticate those
claims. Counter snapshots alone remain `INCOMPLETE`.

The v2 schema, `generativeqc.replay-allocations.v2`, preserves v1 identity,
event, coverage, phase and metric checks. Its expected window contract replaces
the global `zero_new_allocations` boolean with an explicit list:

```json
{
  "id": "warm-energy",
  "phase": "replay",
  "zero_new_allocation_domains": [
    {"owner": "hf", "space": "device:0", "counter": "native-device-journal.v1"}
  ]
}
```

Every listed domain must be admitted in the independently pinned contract and
appear exactly once. At least one hot window must assert a zero-allocation
domain. Other observed domains still require complete, continuous journals and
consistent count/requested-byte/peak/live metrics, but need not be allocation-free.
Thus a known device-zero route can report nonzero host allocations without
either hiding host evidence or imposing a blanket allocation ban. V1's global
assertion semantics and exact field shapes remain unchanged.

### Optional native host-heap collector

The host adapter is qualified against **Memray 1.20.0**, installed only in a
separate optional profiling environment. It is not a compiler/runtime dependency
or a replacement scientific allocator. Use one continuous
`Tracker(..., file_format=FileFormat.ALL_ALLOCATIONS,
trace_python_allocators=False)` across the declared lifecycle and close it before
reading the raw file. The adapter requires finalized, nonaggregated data,
checks the event/footer count and enforces an explicit finite event limit.

Its named owner is `memray-malloc-family`: malloc/calloc/realloc and supported
aligned-allocation requests visible at the profiler's intercepted native call
sites, begun after tracker activation. Pre-tracker frees are outside that owner;
their bytes are never fabricated. Mapped-memory and Python object-suballocator
events are separate ownership layers and explicitly reported as exclusions.
This is not a process-wide heap/RSS or physical-memory claim. In particular,
direct ctypes/dlsym libc calls can bypass the intercepted call sites.

Compile `tools/allocation_audit_marker.cpp` through the verified compiler cache
when placing host phase boundaries. Its ordinary bounded PLT malloc/free calls
are observable; direct libc function pointers are not a substitute. Marker
address/size matches must be observed and unique, otherwise the adapter rejects
the capture instead of guessing a phase boundary from pointer reuse. Account
for observer/marker payloads in the appropriate setup/publication window.

The optional actual-heap tests in `tests/python/test_replay_allocation_capture.py`
check marker visibility, retained/pre-tracker ownership, requested-byte semantics
and the existing production dense-response-oracle buffer hoist against a
per-iteration reference and an independent diagonal-matrix answer. Ordinary
protocol tests alone do not qualify host interception or a complete coordinated
host/device prepared capture.

### Coordinated prepared capture

`tools/capture_prepared_allocations.py` is a reusable qualification command for
the public direct FP64 RHF/UHF `Calculator.prepare_batch` lifecycle. It executes
a ragged H2/water/H2 batch through preparation, first/warm energy, first/warm
energy-plus-force, changed geometry, closing publication, and final observer
cleanup. One continuous host tracker and the prepared owner's native device
journal cover all declared phases. Setup/publication payloads remain accounted;
only the owned-device domain asserts zero new requests in warm replay.

Install `memray==1.20.0` in a separate optional profiling venv. Compile the marker
with the verified compiler cache, compiling an object before linking so cache
use is real:

```bash
mkdir -p .artifacts/joint-allocation-audit
ccache --version
ccache c++ -fPIC -O2 -c tools/allocation_audit_marker.cpp \
  -o .artifacts/joint-allocation-audit/marker.o
c++ -shared .artifacts/joint-allocation-audit/marker.o \
  -o .artifacts/joint-allocation-audit/marker.so
python tools/capture_prepared_allocations.py --freeze-source \
  .artifacts/joint-allocation-audit/source.json
```

Freezing uses a temporary Git index, exports file digests, modes and symlink
targets, and verifies the exact working-tree object without committing or
changing the real index. This manifest may accompany a remote source copy
without `.git`: pin/capture reconstruct and check the Git tree against actual
bytes. The library must be built from that source's native inputs; retain build
configuration and compiler/cache evidence independently. Hash equality alone
does not establish that an arbitrary supplied binary was built from the source.

Pinning and collection must run inside a finite Slurm allocation with exactly
one visible GPU. Preserve `CUDA_VISIBLE_DEVICES`; do not run on a maintenance
node. For example, inside the scheduled shell with the optional profiling venv
activated and `PYTHONPATH="$PWD/python:$PWD"`:

```bash
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
python tools/capture_prepared_allocations.py --pin --method rhf \
  --source-manifest .artifacts/joint-allocation-audit/source.json \
  --library "$GENERATIVEQC_LIBRARY" \
  --marker .artifacts/joint-allocation-audit/marker.so \
  --toolchain '<verified compiler/build flags/cache and optional profiler identity>' \
  --expected .artifacts/joint-allocation-audit/rhf-expected.json
```

Review the independently pinned contract before collection. Repeat the same
arguments with `--capture` instead of `--pin`, adding
`--output .artifacts/joint-allocation-audit/rhf`. Repeat for `uhf`. Existing
contracts and output directories are never overwritten. Typical scheduling is
`srun --partition=main --gres=gpu:5090:1 --nodelist=node1 --nodes=1 --ntasks=1
--time=00:10:00 --input=none bash -lc '<pin/capture commands>'`. A different
physical GPU changes the pinned UUID and requires a separately reviewed contract.

Each capture retains `host.bin`, `receipt.json`, `verification.json` and
`diagnostics.json`. CUDA work is explicitly fenced. Endpoint markers begin the
following publication window; setup/publication markers end their own window.
The final window consumes finalized host EOF. Missing/ambiguous markers, altered
source/artifacts, event loss, mismatched boundaries, device leaks or numerical
failures prevent a passing receipt. The default host parsing bound is 1,048,576
raw events, with a configurable lower bound via `--max-host-events`.

Matched ordinary prepared execution, including changed geometry, runs outside
the monitored lifecycle. Gates are `1e-10` Eh energy and `1e-9` Eh/bohr force;
real iteration counts are recorded. This is an instrumentation non-regression
comparison, not independent scientific qualification of a new method. The
reference warms CUDA context/library state before collection; the host owner
starts at tracker activation, not process startup. Retained result/observer
objects may still own host bytes at EOF. No timing, speedup, blanket heap-zero,
CUDA-driver-allocation or process-wide physical-memory claim is made.

## Strict v1 contract

All objects reject missing and unknown fields; integer metrics are unsigned
64-bit integers (booleans, floats and null are rejected). JSON duplicate keys
are rejected. The schema identifier is `generativeqc.replay-allocations.v1`.

The separately reviewed `expected.json` contains:

| Field | Required meaning |
| --- | --- |
| `schema` | Exact v1 schema identifier |
| `identity` | Exact source/build/device/workload/endpoint identity described below |
| `domains` | Ordered, nonempty list of `{owner, space, counter}`; no duplicate owner/space |
| `windows` | Ordered, nonempty list of `{id, phase, zero_new_allocations}` |

`identity` requires `source_commit` and `source_tree` (40 lowercase hex digits),
`library_sha256`, `artifact_sha256` and `workload_sha256` (64 lowercase hex
digits), plus nonempty `toolchain`, `device` and `endpoint` strings. Pin the
actual installed library and endpoint artifact, not a nearby checkout. Workload
identity must cover inputs, properties, configuration and prepared schedule.
Device identity should include the physical device UUID and visible ordinal;
CPU identity must describe the actual execution host. The counter identifier
names the reviewed production measurement mechanism/version, not a static site.
Spaces are `host` or `device:N` with a nonnegative visible device ordinal.
Host coverage must be explicitly included before making a host/device claim.

Phases are `setup`, `geometry_rebuild`, `endpoint`, `replay`, `iteration`, `tile`
and `publication`. IDs are unique. At least one hot window (`replay`,
`iteration`, or `tile`) must explicitly set `zero_new_allocations: true`.
All windows in the contract must be observed in that order. A phase/window
cannot be silently omitted or relabelled to exempt it from the assertion.

The receipt contains `schema`, `identity`, `execution` and `windows`:

- Identity must match the independent contract exactly.
- `execution` is `{kind: "runtime", completed: true, source_matched: true}`.
  Static, synthetic, unexecuted or incomplete endpoint evidence is rejected.
- Each window is `{id, phase, observations}`. Every contracted ownership domain
  must have exactly one observation, in contract order, even when it allocated
  nothing. Per-owner journals are disjoint accounting domains; overlapping
  counters must not be presented as independent owners.

Each observation contains:

| Field | Required meaning |
| --- | --- |
| `domain` | Exact `{owner, space, counter}` from the contract |
| `coverage` | `complete`, `started_before_window`, `ended_after_window`, `synchronized` all true; `dropped_events` zero |
| `initial_live` | Map from live allocation ID/generation to its originally requested bytes |
| `event_begin`, `event_end` | Half-open contiguous per-domain event cursor range |
| `events` | Every observed allocation/release, in cursor order |
| `metrics` | Independently exported `allocation_count`, `requested_bytes`, `peak_live_bytes`, `live_bytes` for this window |

Events have exactly `sequence`, `kind`, `allocation_id`, `requested_bytes`.
Kinds are `allocate` or `release`; release must match a live allocation's ID
and size. Reuse of an address must retain enough generation identity to
distinguish owners. The journal reconstructs count, sum of successful requested
bytes, peak simultaneous owned requested bytes (including initial live storage),
and final live bytes. Each must equal its respective exported metric. For
example, two separate allocate/free pairs of 16 bytes contribute count 2,
requested bytes 32, and only 16 bytes of additional peak ownership. A zero-byte
allocation still increments count and fails a zero-new-allocation assertion.

Live owners and event cursors must connect exactly between consecutive windows.
Counter resets must be normalized by the capture adapter into one continuous
journal; a fresh native ledger bind alone does not establish this continuity.
Unclassified gaps must be explicit windows. Incomplete traces, failed allocation
attempts without a supported event representation, and resize/capacity uncertainty
fail closed. A resize source site cannot be reported as definitely allocating
without an observed underlying allocation event. Static absence of findings
provides no runtime zeros.

Metrics stay separated by owner, memory space and phase. The checker does not
sum peaks across phases, derive bytes from counts, or use unchanged final live
ownership to rule out transient allocate/free pairs.

## Verification scope

`tests/python/test_replay_allocation_audit.py` uses deliberately constructed
fixtures to validate the consumer's positive and adversarial branches. It runs
with std-lib `unittest` or normal pytest collection and needs no native build:

```sh
python -m unittest discover -s tests/python -p test_replay_allocation_audit.py -v
```

Fixtures are not runtime evidence, even where they exercise the runtime-labelled
branch. No new runtime hook or real CUDA receipt is supplied by this slice.
Remaining #1630 gates include complete source-matched production capture,
reviewed CUDA zero-allocation execution, and a measured iteration-to-preparation
allocation reduction with independent numerical validation.
