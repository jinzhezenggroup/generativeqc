# CUDA residency receipts

`tools/audit_residency_receipts.py` consumes declared transfer and synchronization
evidence for one CUDA endpoint. It does not measure CUDA operations. The native
`cuda_component_trace` JSONL and KS transport diagnostics remain the owners of
their existing measurements. The `#1668` source-work scanner remains an advisory
static inventory; neither its sites nor GPU-related filenames prove residency.

## Contract

The contract and receipt use `generativeqc.residency_receipt` version 1. Both
bind source commit, native source identity, build/library SHA-256, backend,
device, endpoint, problem, resource, owner, and dependency domain. A contract
declares every covered region with one role: `prepare`, `replay`, `iteration`,
`tile`, `publication`, `oracle`, or `compatibility`. The first three hot roles
(`replay`, `iteration`, `tile`) have explicit uint64 ratchets for H2D, D2H, D2D
bytes, stream/device synchronizations, event waits, and paired round trips.
Other roles are reported by role and cannot acquire a path-based exemption.
Publication can therefore be legal yet visible beside a replay sync regression.

A complete receipt requires `completed: true`, `execution: stream`, full region,
transfer, synchronization and payload-link coverage, an event count matching the
ordered `seq` stream, and every event's declared role/owner/domain. A transfer
records direction, byte count, payload identity, and dependency domain. An H2D
claiming a round trip must cite a preceding D2H sequence with the same payload
and dependency domain. Equal or summed byte counts alone never prove a pair.
Unknown, duplicated or reordered events, duplicate JSON members, integer decode
overflow, non-finite JSON numbers, excessive nesting, capture-only execution and
partial coverage return `INCOMPLETE`. A complete stream exceeding a hot ratchet
returns `FAIL`; only a complete stream within every ratchet returns `PASS`.
Receipts are assertions from a producer and must be evaluated with that
producer's coverage and provenance, not inferred from absence of events.

```console
python tools/audit_residency_receipts.py --contract expected.json --receipt observed.json
```

Exit status is 0 for `PASS`, 1 for `FAIL`, and 2 for `INCOMPLETE`. The output
separates `candidate_replay_syncs` from the `publication` role totals.

## Existing DF Trace Adapter

The historical adapter reads real `vibeqc.df_trace` JSONL with its retained
manifest. It checks the manifest schema, source/build identity,
contract-selected retained record, and LF-normalized SHA-256; rejects invalid,
dropped, or capture records; and aggregates only explicitly named owner counters.
`profiler_event_count` counts diagnostic CUDA events, not
production synchronization. The trace itself synchronizes its final event at
operation teardown, and optional progress tracing adds diagnostic region fences.
Those waits are separate from `explicit_synchronizations`,
`stream_synchronizations`, and `raw_panel_event_synchronizations` production
counters. `final_synchronization_ms` is a duration and cannot be converted to a
production wait count.

```console
python tools/audit_residency_receipts.py \
  --contract benchmarks/results/residency-receipts-1629/historical-oh-df-contract.json \
  --df-trace benchmarks/results/issue206-practical-auxiliary/diagnosis/control-diagnosis-v1/oh-def2-svp-spherical-uhf-auto.jsonl \
  --trace-manifest benchmarks/results/issue206-practical-auxiliary/manifest.json
```

The adapter always returns `INCOMPLETE`: the component trace has operation-wide
aggregates, no complete H2D/D2H/D2D or wait enumeration, no per-event payload
links, and no proof that every declared hot region is covered. It is useful for
auditing present counters and identifying precisely what the owner must supply
before a zero-unexpected-transfer claim can pass. The retained OH case is
historical evidence, not a current source or GPU run. See
`benchmarks/results/residency-receipts-1629/README.md` for its exact identity and
independent totals.

The current `generativeqc.df_trace` writer has the same construction/capture
distinction and diagnostic event synchronization. KS `KsTransportDiagnostic`
exposes cumulative setup/density H2D, scalar/matrix D2H, synchronization and
iteration counts; its public ABI does not expose D2D, per-event identity or
complete hot-region coverage. The richer `_ks_snapshot.py` owner diagnostics
cover specific action/preparation paths, not a universal transfer ledger. They
cannot be converted into a complete receipt by subtracting unrelated cumulative
snapshots. A future owner-specific instrumented producer must explicitly close
these coverage gaps before this consumer can pass a resident pathway.

## Optional CUPTI Activity Collection

`tools/cupti_residency_capture.cpp` and `tools/cupti_residency_capture.py` collect
actual activity using the separately installed **CUPTI API 28 (CUDA 12.9 Update
1)**. Neither CUPTI nor the collector is a production/compiler dependency. Use
one fresh qualification process per capture, without another CUPTI profiler.
The native decoder refuses other header/runtime versions rather than assuming
compatible record layouts.

The collector preallocates a finite record array and sixteen 64-KiB activity
buffers before enabling activity. It retains actual memcpy/peer-copy,
synchronization, runtime/driver/internal-API and external-correlation records.
Record overflow, buffer starvation, parser failures, outstanding buffers and
CUPTI-reported drops are explicit invalidation signals. Record storage remains
process-owned after stop to avoid late-callback use-after-free; a second capture
requires a fresh process. This is bounded optional observer storage, not a
scientific allocator or a memory-use recommendation for production.

Only **executed activity** contributes transfer bytes; enclosing runtime and
driver API records are retained as provenance, not counted again. Graph memcpy
activity includes graph/node IDs; graph construction is not itself a transfer.
External correlation assigns the submitting thread's public lifecycle phase.
Unassigned activities are reported, never guessed from timestamps or byte totals.
Explicit observer completion fences have a separate phase and remain visible.
Stop flushes activity but does not synchronize CUDA itself: the caller must first
fence work, and unfinished timestamps prevent an intact-stream claim.

Synchronization record types alone do not distinguish blocking waits from
nonblocking event/stream queries. The adapter joins API names by correlation ID,
reports ready/not-ready queries separately (including the normal `600` status),
and refuses to count an ambiguous event/stream record as a wait. This avoids
turning polling into either a spurious wait count or a false failed-sync report.

Compile the optional shared collector through the verified cache, compiling an
object before linking. For a verified CUDA/CUPTI installation:

```bash
mkdir -p .artifacts/residency-audit
ccache --version
ccache c++ -std=c++17 -fPIC -O2 -I"$CUPTI_ROOT/include" \
  -I"$CUDA_HOME/include" -c tools/cupti_residency_capture.cpp \
  -o .artifacts/residency-audit/collector.o
ccache c++ -std=c++17 -fPIC -O2 -I"$CUPTI_ROOT/include" \
  -I"$CUDA_HOME/include" -c tools/cupti_graph_inventory.cpp \
  -o .artifacts/residency-audit/graph-inventory.o
ccache c++ -std=c++17 -fPIC -O2 -I"$CUPTI_ROOT/include" \
  -I"$CUDA_HOME/include" -c tools/cupti_source_capture.cpp \
  -o .artifacts/residency-audit/source-boundaries.o
c++ -shared .artifacts/residency-audit/{collector,graph-inventory,source-boundaries}.o \
  -L"$CUPTI_ROOT/lib64" -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$CUPTI_ROOT/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -lcupti -lcudart -pthread \
  -o .artifacts/residency-audit/collector.so
python tools/capture_prepared_residency.py --freeze-source \
  .artifacts/residency-audit/source.json
```

The source exporter shares the independently checked temporary-index and remote
tree reconstruction used by the [allocation capture](replay_allocation_receipts.md).
Retain matching native build inputs/configuration and compiler/cache evidence;
an arbitrary binary's hash alone does not establish build/source correspondence.
Inside a finite Slurm GPU job, preserving the assigned visibility, pin before
collection:

```bash
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
python tools/capture_prepared_residency.py --pin --method rhf \
  --source-manifest .artifacts/residency-audit/source.json \
  --library "$GENERATIVEQC_LIBRARY" \
  --collector .artifacts/residency-audit/collector.so \
  --cupti-library "$CUPTI_ROOT/lib64/libcupti.so" \
  --toolchain '<verified compiler/build/cache identity>' \
  --expected .artifacts/residency-audit/rhf-expected.json
```

Review the contract, then repeat with `--capture` instead of `--pin`, adding
`--output .artifacts/residency-audit/rhf`. Repeat in a fresh process for `uhf`.
The case covers ragged direct FP64 H2/water/H2 preparation, first/warm energy,
first/warm forces, geometry change and closing publication. Installed source,
native library, collector, loaded CUPTI library and assigned GPU identity are
checked, including post-execution source/artifact checks. Matched ordinary
prepared history runs outside capture; energy/force acceptance and real iteration
counts are recorded without a timing/speedup claim. Matched ordinary CUDA
histories must retain identical iteration counts. A separate native-CPU FP64
direct history runs outside capture and checks independent-backend energy/force
errors, retaining but not equating different backend convergence histories.
`numerical.json` labels both reference kinds explicitly; ordinary-CUDA agreement
is observation noninterference, not an independent CPU reference. Output retains `activity.json`,
`diagnostics.json`, `graphs.json`, `graph-diagnostics.json`,
`source-boundaries.json`, `source-diagnostics.json` and `numerical.json`;
contracts and output directories cannot be overwritten. The default record bound
is 262,144, configurable up to 1,048,576. Graph traversal additionally limits the
total nodes per definition (`--graph-node-limit`, at most 4096) and nesting depth
(`--graph-depth-limit`, at most 16); both bounds are independently pinned.

### Observed replay work ratchet

The optional `--work-ratchet` selects independently reviewed limits for the
complete declared workload. Supply the
same file for `--pin` and `--capture`. The whole policy SHA-256 and selected
profile are pinned outside capture and rechecked after execution; pinning never
learns or raises limits from the run being checked. Matching uses the complete
declared workload, including method, precision, systems, endpoint properties and
numerical gates. The retained RHF/UHF profiles cover the fixed direct-FP64
H2/water/H2 warm energy/force histories, not arbitrary systems or other owners.

The retained `manifests/residency_work_ratchets/hf_prepared_direct_fp64.v1.json`
profiles use the historical rigid-translation workload. Current producers pin
and execute a non-rigid final-atom displacement; the old profiles therefore
fail the exact-workload gate and must not be relabeled with the new workload
hash. Capture without an observed-work assertion until a matching independently
reviewed policy has fresh source-matched evidence. Omitting this optional gate
does not establish work boundedness or complete residency acceptance.

`source-work-ratchet.json` recomputes source/API/activity joins from the bounded
raw journals. It checks each selected public hot region by actual source owner
and role, keeping successful transfer API calls, zero-byte calls, source stream
fences, source event fences, executed transfer bytes and observed sync/event-wait
records distinct. Policy schema v2 requires an explicit `source_event_fences`
limit for each listed owner/role pair. Frozen v1 policies remain accepted with
an implicit zero event-fence allowance; their original bytes and reviewed
allowances are not rewritten or enlarged.
Unlisted owner/role pairs have zero allowance. Preparation, publication, oracle
or compatibility labels inside replay cannot borrow another pair's allowance;
explicit limits can admit legal work in any recognized source role. Unowned
blocking work in a selected region requires review. Observer completion fences
and production setup outside selected replay remain separate. Nonblocking
queries, including normal not-ready results, are retained separately rather
than invented as fences or implicit-wait proof.

An intact stream within limits reports `WITHIN_OBSERVED_RATCHET`; exceeding a
limit or adding unowned blocking work reports `VIOLATION`. Loss, missing phase
API provenance or source correlation faults cannot authorize the observed gate.
With a configured policy, capture exits 1 when the gate is not within limits,
retaining available raw histories and numerical diagnostics for review. A run
within the observed gate still exits 2 and remains a full-residency `INCOMPLETE`,
never `PASS` or a zero-round-trip assertion. CI/qualification must require the
named gate explicitly, not interpret `INCOMPLETE` or absence of a report as
approval. Changing limits requires reviewed source-matched qualification, not
automatic baseline regeneration or file-path exemptions.

This gate cannot certify scientific producer/host-transform/consumer links,
implicit blocking, dynamic/device-tail graph execution or a same-count/byte
host fallback. It is not an allocation audit or endpoint timing benchmark.
Those remain separate obligations of the full residency contract.

### Source-owned graph inventory

The optional private `generativeqc_residency_observer_*_v1` ABI binds a synchronous
callback only on the submitting thread. It adds no CUPTI dependency to production
and never chooses an execution route. The RHF/UHF graph owner emits definitions,
host-launch begin/end and destruction at safe source boundaries. The callback
inspects the real retained graph and recursively enumerates child graph nodes;
it runs outside CUDA/CUPTI API callbacks. Process-local lifetime generations and
CUPTI graph/exec/node IDs are recorded, not retained or matched handle addresses.
Reconstruction, failed launches, callback exceptions, unknown node types, bound
exhaustion and missing definitions/lifetime closure stay visible.

An independent external-correlation domain binds the actual host `cudaGraphLaunch`
and driver API records to source submissions without replacing public lifecycle
phase correlation. `graph-diagnostics.json` reports declared event/semaphore wait
nodes and accepted host submissions separately. It never calls their product an
observed wait count: graph waits can be absent from CUPTI synchronization records,
and device-tail launches, conditional graphs and opaque host callbacks are not
qualified by static node inspection. Unknown instantiation flags are incomplete.
The inventory covers only these source-owned HF graphs; other graph owners,
graph updates, managed-memory migration and implicit CUDA blocking remain outside
qualified coverage. It cannot issue a zero-unexpected-round-trip assertion.

### Source-owned transfer and fence roles

The same optional submitting-thread callback multiplexes graph layout 1 and
boundary layout 2. `src/runtime/residency_boundaries.hpp` owns the immutable native
owner/role/site/payload tags; names are read from the actual production library
with `generativeqc_residency_boundary_name_v1`, not a duplicated Python registry.
The HF bucket driver declares individual result fields and forces as final
publication, host active/refinement/final-Fock controls as iteration, input
validation and explicit topology/geometry/warm-seed uploads as preparation,
and explicit profiling/structural diagnostics as
oracle work. These are source declarations, not exclusions for an entire file
or classifications inferred from a public `execute` interval. Graph capture and
upload drains belong to the separate `hf_graph_setup` owner and preparation
role. Bucket/eigensolver release fences belong to their actual resource owners.
The shared `device_resource_ledger` owns the existing tracked-async-release drain
before logical reservation retirement and the async-allocation metadata-rollback
drain. These scopes observe exactly the original conditional calls; untracked
async release remains asynchronous, failed frees do not create a fence, and
rollback still ignores the fence return when determining allocation status.
Neither the enclosing HF resource owner nor the ledger's allocation generations
establish a scientific payload/dependency link. All these fences use the
conservative `lifetime` role. Resource cleanup may occur inside an
endpoint, so a lifetime declaration is never an automatic publication/setup
exemption and remains higher risk even in a public close interval.

The `posthf_df_source` owner observes the explicit generated-DF host-tile
adapter: each nonempty raw read declares two existing `cudaEventSynchronize`
calls and one D2H copy in the compatibility role. Generation and publication
event fences have separate sites; they are not stream synchronizations or
device-side stream-wait-event submissions. The adapter's existing release drain
uses the lifetime role and its one-way J/K handoff drain uses preparation.
Host-resident metric reads and empty raw reads emit no transfer/fence boundary.
Underlying metric/source setup, implicit waits, and the separate J/K consumer
remain outside this adapter's coverage. Native payload generations do not link
this source to a downstream scientific consumer. `CudaDFSource` and
`DFProvider` therefore report unobserved `subsequent_h2d_bytes` as `null`, not
as measured zero; complete transfer attribution requires a qualified consumer
and source-owned scientific lineage.

Each source invocation and boundary submission has a scalar generation. Named
payload instances identify the source field for that invocation/submission,
not pointer identity or a complete scientific producer/consumer dependency.
The raw native wrappers execute the original CUDA copy/fence exactly once,
preserving its arguments, stream, zero-byte calls, return status and execution
order. The input-upload adapter instead preserves the existing HF helper's empty
upload early return: a skipped helper makes no CUDA call and emits no transfer
boundary. Borrowed scalar upload descriptors name each field without adding
retained numeric storage. Input density/energy seeds have distinct native tags
from downloaded density/energy results; those names do not assert scientific
identity or a producer/consumer link. The wrappers never add a completion fence.
Observer exceptions, recursion, exhausted IDs,
incomplete scopes and bounded-record loss remain explicit observation failures.

A third external-correlation domain joins source submissions to runtime/driver
API records and actual memcpy/synchronization activity. `source-diagnostics.json`
counts observed transfer bytes once, separately from verified successful source
API calls and CUPTI synchronization records. `by_role` and `by_owner` are separate
projections of the same matched work; `by_phase_owner_role` is another independent
projection for contextual replay limits. Do not add their totals together. Runtime
CUDA return codes must agree exactly, not merely agree that both are failures.
Zero-byte calls are visible but do
not invent transfer events. Event/stream queries cannot verify a declared source
fence. A source event fence requires the actual blocking event runtime API;
observed synchronization subtypes must agree with the declared fence kind.
The raw CUPTI event-synchronize and stream-wait-event subtypes stay available
as `sync_subtype` even in their shared event-wait activity projection.
Iteration/tile boundaries are higher-risk than explicit non-hot roles;
compatibility/oracle labels are legal only when actually declared by the source.

Unannotated production activity is retained, including work in other owners and
graph/resource helpers. Matching annotations are not proof of complete CUDA
coverage. Equal-sized downloads/uploads or field names cannot create a host
transform dependency; nonzero dependency IDs without qualified transform/link
evidence are incomplete. Full payload dependencies, graph execution and implicit
blocking still prevent a residency PASS. The advisory static source-work scanner
recognizes these explicitly qualified wrapper names so observation cannot hide
loop transfer/fence sites; its source findings remain review candidates, not
runtime counts or a scientific classification proof.

The graph-only observer ignores the recognized independent boundary layout 2;
the combined observer validates both journals. Unknown layouts still fail closed.
Annotating every observed transfer/explicit synchronization in a sampled HF
history does not establish coverage of implicit CUDA/library blocking or actual
graph wait execution. Dependency/wait coverage flags and the zero-round-trip
assertion remain incomplete until their independent contracts are qualified.

**Successful collection still exits 2 (`INCOMPLETE`)**, never `PASS` for the strict
residency auditor. Intact activity is not a complete residency receipt. Public
`execute` phase boundaries do not identify internal final publication versus
iteration/tile roles; opaque graph/correlation IDs do not identify scientific
payloads or host-transform dependencies. Equal bytes, addresses or adjacent
D2H/H2D records cannot create a round-trip link. Managed-memory migrations,
all implicit blocking and graph wait-node coverage are not qualified by this
adapter. It therefore leaves payload/dependency identities and zero-round-trip
assertions unset in the activity-only diagnostics. The separate source diagnostics
add named instances without inventing dependency links. Coverage for other owners,
full source-owned links and complete relevant wait
coverage remain required before an owner-specific residency ratchet can pass.

Protocol tests are `test_cupti_residency_capture.py` and
`test_cupti_graph_inventory.py` and `test_cupti_source_capture.py`; optional CPU-double native
ABI tests use real headers but never link/run CUPTI or access a GPU. Their header
export is selected with `GENERATIVEQC_CUPTI_TEST_HEADERS`, containing `cupti/` and
`cuda/` include directories. Neither kind of test substitutes for scheduled real
activity calibration, graph replay or production endpoint qualification.
