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
