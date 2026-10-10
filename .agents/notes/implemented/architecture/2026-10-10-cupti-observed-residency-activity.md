# Decision: collect bounded CUDA activity without fabricating residency coverage

Status: implemented (activity collector, not complete residency certification)
Date: 2026-10-10

## Problem

The #1629 advisory scanner and strict receipt consumer already distinguish
legal preparation/publication from hot residency regressions. Existing DF/KS
component aggregates do not enumerate every transfer, synchronization or payload
dependency. A caller-supplied zero or absence of events must not certify a path.

## Decision

Add an optional process-owned CUPTI activity collector, separately compiled and
linked against verified CUPTI API 28 / CUDA 12.9 Update 1. Production/compiler
dependencies, numerical kernels, schedules and allocators are unchanged. Require
an exact header/runtime version match before enabling the decoder, finite record
and callback-buffer storage, explicit loss/error reporting, and a fresh process
per capture. Retain native storage for process lifetime so late callbacks cannot
dereference a destroyed handle. This bounded observer storage is not part of a
scientific workspace budget or a production memory/performance recommendation.

Collect actual memcpy/peer-copy and synchronization activity, alongside
runtime/driver/internal API and external-correlation provenance. Count executed
copies once, not both nesting layers of API calls. Resolve phase correlations
after reading the complete unordered buffer stream. Never guess missing phases
from time or bytes. Graph copies retain graph/node IDs; construction is not replay.
Explicit observer CUDA fences have a separate visible region. Flush is not a
CUDA completion fence, and loss/unfinished records prevent an intact-stream claim.

The reusable command `tools/capture_prepared_residency.py` pins source tree,
library, collector, CUPTI dependency, assigned physical GPU, workload and bounds
before collection; verifies copied source and loaded libraries; and repeats
source/artifact checks after execution. It executes the actual public prepared
direct FP64 RHF/UHF lifecycle and compares matching ordinary prepared history
outside capture. Output is raw activity plus diagnostics and numerical/work-count
evidence, not a fabricated strict receipt. Successful collection deliberately
returns `INCOMPLETE` / exit 2.

## Important negative results

### Synchronization types are not sufficient

A pending-event calibration exposed CUPTI `SYNCHRONIZATION` records with event
synchronize type for **`cudaEventQuery`**, including ordinary not-ready status
`600`. A type-only adapter misclassified these as failed waits. The correction
joins the actual API name by correlation ID, counts ready/not-ready event/stream
queries separately, and refuses ambiguous event/stream records without an API
identity. Query status 600 is not a failed blocking synchronization. Both ready
and pending queries, runtime/driver names and per-thread-stream name variants
have explicit protocol gates. Raw API results remain retained.

### Executed graph waits are not enumerated by these counters

Slurm job **6903**, node1/main, one RTX 5090, finite five-minute limit, executed
two replays of a graph containing an external event-wait node. Independent
`cudaEventQuery` checks confirmed both producer events were pending. A producer
host callback delayed each event; the corresponding consumer stream waits
actually blocked for about 0.25 seconds each. Yet the collector observed **zero
graph event-wait records**. An ordinary `cudaStreamWaitEvent` in the same process
was observed correctly. The delay is only an observability control, not an
endpoint latency or speedup benchmark. Job 6901 also retained the already-ready
event control; it cannot by itself prove the pending-event case.

Therefore `event_waits == 0` is not proof that graph replay has no waits. Source
owners must supply graph node/launch semantics and complete relevant wait
coverage. Additional activity kinds, API timestamps or a larger buffer alone do
not justify upgrading the current diagnostic into a complete receipt.

## Real calibration and prepared evidence

Job **6896**, node1/main, one RTX 5090, finite five-minute limit, calibrated actual
128-byte H2D/D2D/D2H copies, their independent exact 16-double output, graph
construction and two graph replays. Ordinary copies contributed 128 bytes per
direction; construction contributed no transfers; replays contributed 256 bytes
per direction and six graph-copy records. Production stream syncs and an observer
device fence were separately attributed. Native drop, starvation, parse, unknown,
pending-buffer and finish-error counters were all zero. This qualifies these
activity patterns, not universal CUDA movement or wait coverage.

Job **6906**, node1/main, one RTX 5090, finite ten-minute limit, collected complete
observed activity for the actual ragged H2/water/H2 prepared RHF and UHF cases:
prepare, first/warm energy, first/warm forces, changed geometry and close. The
frozen working-tree object is `7d3c058dd91cf8f73e6ffc37c20b70c39adcd6af`, over baseline
`f87d51ab616cc9aa774bd344a15e268cef7eaf05`. Native inputs were unchanged from the
qualified resource-journal build. Library SHA-256:
`4898b3aa36f92f1d4e8ad1815ae3e4c0a295f292588dc3b106097ac1f188c858`;
collector SHA-256:
`61f91eecae64509fd14711ac15073a3c05ac0f25307014556e774426afeef7dc`;
CUPTI SHA-256:
`2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a`.
Build configuration/cache evidence remains retained; qualification AOT flags are
not promoted as production defaults.

RHF/UHF produced 2580/2632 raw records with no drops, unassigned work or capture
errors. Warm energy observed 24 H2D bytes and 557/1013 D2H bytes, with four sync
records. Warm forces observed 40 H2D bytes and 714/1170 D2H bytes, with two sync
records. Seven observer device fences remained separate; closing publication had
six observed sync records. These public execution intervals still contain legal
internal publication: their totals are not unexpected-round-trip regressions.

Maximum matched energy error was below `3e-14` Eh and force error below `9.3e-15`
Eh/bohr, within the declared `1e-10`/`1e-9` gates. Actual/reference iteration counts
agree: `[2,8,2]`, `[1,1,1]`, `[2,2,2]`, `[1,1,1]`, `[2,2,2]` across the five
executions. This is instrumentation non-regression, not independent qualification
of a new scientific method. Raw records, contracts, source manifests, numerical
evidence, current-adapter rechecks and logs are retained in ignored
`.artifacts/issue1629-cupti/`; the frozen Git tree/archive owns the measurement.
Later note/editorial changes do not rewrite its identity. The earlier job 6900
capture is retained separately.

CPU-double probes compiled against actual SDK layouts check exact version
rejection, finite capacity, overflow, all callback buffers exhausted, dropped and
unknown records, parse/stop failures and no-write short-buffer ABI behavior.
They link doubles, never CUPTI, and do not access a GPU. Protocol tests check
source order independence, conflicting correlations, query classification,
observer exclusion, malformed fields and that empty/zero observations cannot
issue a residency PASS. Neither set substitutes for the scheduled calibration.

## Rejected shortcuts and remaining acceptance

- Runtime memcpy API counts alone miss graph replay and can count construction.
- Summing runtime and driver copies double-counts the same operation.
- Pairing equal bytes/addresses or adjacent D2H/H2D invents payload dependencies.
- Treating query records as waits creates spurious failures and sync counts.
- Treating observed graph wait zero as complete coverage creates false PASS.
- Treating public `execute` boundaries as internal publication/iteration roles
  misclassifies legitimate result downloads; source-owned roles remain required.

#1629 remains open and incomplete: full source-owned payload/dependency links,
publication/iteration/tile role annotation, graph/implicit-wait coverage and a
qualified zero-unexpected-round-trip resident ratchet remain outstanding. The
collector directly advances runtime enumeration, but does not replace those
acceptance requirements with a narrower claim. Do not promote the existing
partial DF trace adapter or these diagnostics to PASS by changing only flags.

## Revisit when

The follow-up [source-owned graph inventory](2026-10-10-source-owned-cuda-graph-inventory.md)
adds bounded inspection and correlated host submissions for the retained HF graph
owner. It does not supersede the graph-wait blind-spot evidence, establish dynamic
execution counts or change this historical qualification's source/library identity.

A production owner can bind actual graph nodes/launches and memory lifetimes to
source-owned roles and payload dependencies, a different CUPTI version is
explicitly qualified, or additional movement/wait layers become part of the
measurement contract. Current-state usage lives in
`docs/maintainer/residency_receipts.md`.
