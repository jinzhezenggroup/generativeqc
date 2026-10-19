# Decision: explicitly opt in to CC source phase observations

Status: implemented
Date: 2026-10-19

## Problem

The resident CC route prepares host pointer/index metadata and submits GPU factor
production and contractions. Whole calls to generated CUDA work cannot represent
CPU numerical tile production. An inclusive progress log also cannot identify
which primal/replay batch survived actual preparation/allocation fallbacks.

## Decision

Keep the existing canonical residency callback/registration ABI and old scalar
layouts. A separate private CC control validates an explicit versioned contract,
the live callback/context owner and its canonical binding epoch. Only this
explicit permit enables the new CC records. A legacy observer alone receives no
new layouts. This is not a second callback-registration service.

The native control keeps integers only: binding epoch, permit, current parent,
plan identity, active depth and sticky errors. It does not retain borrowed
callback/context/code pointers across calls. Every source dispatch uses the
current native binding. A changed epoch suppresses delivery to a replacement
binding and makes the original capture incomplete. Finish refuses an open scope
and reports both lifecycle and canonical callback errors. Permits and source IDs
do not wrap; exhaustion prevents a complete new capture.

Source scopes are native, same-thread, nested stack objects. External clients
include the scalar contract only; transferring implementation-scope objects
between threads or instantiating another inline native state is not admitted.
The collector must still validate observed thread, nesting and closure rather
than treating a successful control call as a complete-capture certificate.

Twelve-word phase records distinguish inclusive calls, host-only pointer/state
preparation, GPU production submissions and GPU accumulation submissions. Scope
identity remains readable after close so the accumulation can declare the source
production call whose returned output it consumes. This is program dataflow,
not independent proof of device completion or a hideable stall. Exception
unwinding closes scopes with a nonzero status. Closed scopes are not emitted
twice.

A separate fourteen-word admission record snapshots the actual prepared native
Owner after its existing constructor, contraction preparation and allocation
fallbacks have finished. It records the admitted primal/replay batch, route,
budget and existing high-water capacity diagnostics; it does not select a plan, probe memory, increase a
budget, add a CUDA query or synchronize. The admission identity is a snapshot
tag, not an allocation/stream lifetime proof. Numeric/device high-water fields
come from existing max/peak accounting; numeric also includes provider peaks.
They are not exact live allocations or automatically comparable to the local
Owner maximum budget. Final batch/route facts are distinct from those peaks.
Nested scopes restore their
parent/plan tags on exit.

## Current coverage

The DF method attempt has an inclusive source root. Native CC solve/setup/
cleanup, primal/replay calls, both Q loops, host metadata, GPU production and
accumulation have explicit boundaries. The existing CUDA calls, branch
conditions, Q traversal, accumulation order, mathematical expressions, precision
and budgets remain unchanged. Cleanup callbacks begin before its allocation
measurement lock and end after that lock is released.

The mixed branch is explicitly inclusive rather than labelled pure production.
It is outside the proposed strict-FP64 capture admission. An inclusive root does
not prove detailed upstream RHF/DF or complete triples producer coverage. Those
source/work censuses and an actual collector remain separate work. No profiler
backend or default capture is enabled by these source hooks.

## Validation and proof limits

CPU tests compile a separate observer bridge and source emitter into a shared
library. The external client includes only the scalar contract and cannot
instantiate another observer TLS instance. The cases cover legacy-disabled
delivery, explicit version/live-owner checks, null context, busy finish,
canonical ABA/replacement, thread domain, scientific/callback exceptions,
binding/source-ID exhaustion, synthetic prepared-plan fallback and dependency
identity after close. Synthetic plan facts do not independently qualify a real
CUDA resource admission.

The affected CUDA solve and method translation units also compile through the
verified existing cache/toolchain against an immutable candidate snapshot. That
is offline object/type validation, not an installed DSO, device execution,
scientific acceptance or a complete endpoint attribution result.

The disabled path adds real TLS/branch/stack footprint. It performs no added
explicit heap allocation, clock read, I/O or CUDA operation in the source helper;
this is not a whole-library zero-overhead claim. Old inline or mixed old/new DSO
binding routes remain outside the binding-epoch proof domain. Actual capture
must establish the new native, canonical registration and collector identities.

Historical source/library/profile evidence keeps its original identity and
INCOMPLETE limitations. The original #1765 goal is not completed by these hooks,
host tests, successful compilation or CI.

## Detailed contract v2

The original opt-in preserves its Q-only emission view. A separately negotiated
v2 permit enables DF raw generation, projections, packing, publication/validation
and complete triples tile/panel/moment/epilogue/reduction/wait sites. A typed
seventeen-word triples admission record reports the selected lowering,
precision, panel capacity and reserved layout after preparation/fallback.
Reserved layout bytes are distinct from live allocation or peak measurements.
The CLI exposes existing DF/triples work counters so a source census need not
derive its expected inventory from the observer callback itself.

The triples algorithm's identities/ages/cache decisions remain unchanged. Its
separate three-entry diagnostic PanelSources ledger intentionally uses 48 bytes
of zero-initialized stack storage, including while capture is disabled. A late
v2 enable followed by an already-existing panel cache hit therefore returns the
explicit unknown identity zero. A different permit cannot inherit a cached
producer identity. Invalid slots fail closed without out-of-bounds access.
CPU controls exercise these paths directly, not by asserting source spelling.

Optional paired moment submission retains the original short-circuit and call.
Its observer scope records bounded decline status2 when that existing optional
call returns false; this is separate from scientific exception status1. The
collector must interpret the typed site/status and independent admitted route,
not silently treat every declined optional attempt as completed GPU production.
These source additions still do not establish collector losslessness, installed
build qualification, device execution, independent scientific acceptance or
whole-endpoint attribution.

The Owner's optional replay-batch storage uses zero as a scalar-fallback
sentinel. Admission reports effective width through the same replay tile rule
used by the real loop: positive remaining Q with no prepared batch means width1,
no Q means width0, and prepared batches retain their bounded/tail width. Tests
connect this actual projection to literal scalar/batched/ragged/no-Q walks and
prepared/allocation fallback doubles; they do not replace the emitted sentinel
with a hardcoded synthetic width while leaving production reporting unchanged.

Triples census keeps three layers distinct: logical W products, issued vendor
GEMM-family calls (including batched calls), and actual device kernel records.
For nine occupied orbitals the independent combinatorial domain has165
triangular tiles,729 ordered physical tuples and1458 two-product W terms. A
successful pair can represent four logical products through two batched vendor
calls plus helper kernels. None of these numbers alone fixes device kernel
count. Admitted lowering, pointer/panel capacity and accepted/declined pairing
remain required inputs to the mapping.

## Standalone activity and source journals

The optional benchmark collector uses CUPTI API28 concurrent-kernel, memcpy,
memset, synchronization, external-correlation and API activity records. It does
not enable serialized KERNEL activity. CUSTOM0 belongs to the CC phase stack;
the existing graph/source observers retain CUSTOM1/CUSTOM2. One canonical
native observer binding multiplexes all three views. The standalone module is
pinned for process lifetime because CUPTI can retain callback addresses.
Partially constructed sessions explicitly release their permit, binding and
activity registration before propagating an initialization refusal.

Binary v1 records have explicit little-endian headers and fixed widths:20
activity words,19 journal words containing up to17 original source words,
12 graph words and14 legacy-source words. Existing source layouts1/2 are not
reinterpreted. Metadata retains overflow, unknown/parse records, pending buffers,
clock failures and incomplete lifetimes. Missing queued/completed timestamps
remain UNAVAILABLE. Atomic publication cannot replace an existing artifact.

At the benchmark's selected capacities, row storage alone is bounded by320MiB
of activity,38MiB of CC journal,24MiB of graph and28MiB of legacy source records,
plus the fixed activity buffers, graph-query workspace and at most28MiB of
temporary legacy export rows. Activity export does not allocate a second full
activity vector.
These are observer host allocations, separate from scientific/device budgets.
An unknown or dropped record is a failed completeness gate, not permission to
increase the production numeric allowance or suppress evidence.

The reader binds opened regular-file identities and hashes; it rejects aliases,
symlinks, partial records and unsupported versions. Independent geometry,
prepared plans, existing scientific work counters and mathematical triples
enumeration constrain the source census. Q cardinality is compared with the
bounded record inventory before expanding a corrupt huge domain. Tests include
joint omitted/extra phase subtrees, stale producers and a huge-Q fast refusal.

The caller's normalized CUPTI clock envelope, source-root interval and supervised
monotonic child wall time are distinct boundaries. They are not subtracted from
each other. The observer-only completion fence occurs after scientific resource
destruction/timing. Only its exact correlation identity and actual successful
wait/API role can identify it; arbitrary high-bit identities and kernels/copies
cannot inherit that classification. Runtime/driver kinds come from compiled
CUPTI enums, not guessed numeric names.

A separate1024-integer CUDA transport control uses a hand oracle, fixed operation
and byte census, actual visible-device UUID and a bounded parent-ready handshake
for module/PID attestation. It is not one of the scientific campaign samples.
CPU doubles, compilation, schema queries, queue admission and that small device
control cannot qualify the installed CC endpoint or the large source census.
The raw-ledger report remains INCOMPLETE until independent producer/science and
tail evidence is validated; editable receipt booleans cannot manufacture PASS.
Historical Nsight warnings and scientific sample identities remain unchanged.

## Source-matched installed run and offline audit

The source7045237 installed native (SHA256
`b1ee3eacf7bd21bb8a55bfd1f29a2704f0549ccf8775207fc54124d94fc07f97`)
was exercised on H200/driver595.58.03 using two fresh clean processes and one
fresh CUPTI diagnostic process. The original230-AO/o9/v221/Q488 input and
independent committed PySCF oracle, strictFP64, Q8/DIIS8, numerical thresholds,
provider allowance and scientific budgets were retained. All three native exits
were0, original E/(T)/physicalR1/R2 gates passed, numerical bits and25 declared
work counters matched. Complete supervised child walls were27.148620069,
27.048108431 and27.499532839seconds. These three points are not statistical
profitability evidence and are separate from the older129d/Nsight campaign.

The diagnostic retained647895 activity records and15542 source journal records,
plus122 graph and158 legacy source records. Raw activity SHA256 is
`9281fb43ad755033ee1bc32b181ed0ae2863131da165b87855e4ce4206f97ffa`;
source journal SHA256 is
`217d2d4601ec09cc2e41a472c18887e1788b2cf30858621a97037bf56d152661`.
The independent geometry/plan/work census accounts for1281 Q tiles,165 triples
tiles,1458 logical W products and908 vendor GEMM-family calls. The selected
positive-domain audit requires actual kernel activity for kernel submission
sites, real copy bytes for DF publication and the12-byte triples energy/error
download, and both successful host synchronization API and device-wait joins
for every selected explicit wait. Async completion may exceed its submitting
host scope, but cannot precede submission or extend beyond the real final fence.

The complete normalized caller window is27.499687842seconds. Its mutually
exclusive partition retains5.588393579seconds unattributed. SourceRoot is a
smaller scientific component, not that full window. Host metadata intervals,
device waits and host API blocking have distinct meanings; no residual is
reclassified as CPU tensor production, GPU idle or causally hideable work.
The offline consumer accepts reviewed external spec/worker hashes and verifies
closed receipts, actual endpoint bytes, pre-exec witnesses, frozen module
bindings, independent numerical gates, source census, activity families and tail.
Frozen producer loader observations are not an independent remote DSO rehash.

The historical run had per-native1800-second supervision but lacked an outer
bound covering the whole worker before launch. Its late guard attached after
the scientific processes ended and saw an empty session; this is not a proof
of execution-time termination/reaping. That protocol gap remains part of the
run's provenance and is not repaired by another scientific sample. The separate
execution-guard implementation seals PID/start/session ownership and waits for
an independent finite guard to acknowledge the held child before releasing exec.
CPU controls exercise normal drain, timeout with a native descendant, unrelated
session preservation, reused leader rejection and postspawn identity failure.
That repaired protocol is a different version from the actual historical run.
