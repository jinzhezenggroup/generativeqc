# Prepared replay allocation receipts

`tools/audit_replay_allocations.py` consumes runtime allocation journals and
checks explicit zero-new-allocation assertions. It installs no allocator,
changes no production execution policy, and does not upgrade the
[advisory source-work audit](source_work_audit.md) into runtime evidence.

```sh
python tools/audit_replay_allocations.py receipt.json --expected expected.json
python tools/audit_replay_allocations.py ledger.json --native-ledger-v1
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

`--native-ledger-v1` accepts the exact `NativeDeviceLedger.to_dict()` shape.
It preserves allocation count, live/peak bytes, owner, visible device and
rejected allocations. It reports requested bytes as `null` and returns
`INCOMPLETE`, including when allocation count is zero. V1 lacks a cumulative
requested-byte counter, allocation/release journal, phase/source/build identity
and complete host observation. CPU boundary samples are also incomplete and
cannot fill those fields. Missing data is never inferred from live/peak bytes.

`tests/python/test_hf_resources_cuda.py` already contains a warm owned-device
allocation-count assertion and cold/warm numerical comparisons. This verifier
does not claim to have executed that GPU test or to have qualified a complete
host/device zero-allocation pathway. Completing that gate requires a capture
adapter integrated with the existing measurement owner, not a second allocator.

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
