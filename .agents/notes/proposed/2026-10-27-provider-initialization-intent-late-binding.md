# Decision: transport finite provider intent and bind identity in the native owner

Status: proposed
Date: 2026-10-27

## Problem

The native foundation accepts a qualified internal request, but an ordinary SDK
caller cannot supply trustworthy current-context or loaded-provider evidence.
The SDK must carry a reservation request without converting caller metadata into
permission. Prepared reuse must authenticate the actual owner generation even
when the caller's finite intent has not changed.

## Decision

Use a separately included size/version v1 C descriptor containing mode, a zero
reserved field and finite allowance A. Its 24-byte public ABI has no context ID,
UUID, build ID or qualification bit. Snapshot the descriptor to a value before
entering the private preparation port. A Python frozen value and optional adapter
use the already loaded library; absent intent calls the old preparation entry
without resolving the new symbols. An explicit request cannot fall back silently.

Initially admit only scalar restricted conventional CUDA MP2/RCCSD physical
references. Native-owned backend/system/method metadata and active-ledger basis
must pass before selecting the existing factory. Keep the existing registry's
option, system and preliminary-policy validation and its single candidate /
positive publisher. Never stage intent in mutable context globals or attach it
after publication. Preserve the existing B_ref/B_endpoint admission helpers and
one native retained reservation; this layer introduces no second sizing formula,
planner voucher or default promotion.

The physical-reference owner must bind after its normal context establishment.
An intent value only selects A. Actual current context, selected provider family
and strong loaded-object leases supply the private key. A context/key/loaded-owner
mismatch retires replay, preserves the original key and never overwrites it with
a new observation. A fresh generation requires explicit retirement of the old
owner and its actual lifetime boundary.

## Retirement is not successful teardown

The generation helper is an identity/lease contract, not a CUDA resource
destructor. It has no context create/select/retain/reset operation. A foreign or
unavailable current owner refuses the cleanup callback and reports unavailable
status, preserving leases. For a matching owner, only the actual cleanup callback
can report returned API values. An uncalled operation has no status; a status-less
host operation is distinct from a CUDA/vendor call returning zero. Failed cleanup
does not certify retirement or release the binding's leases. Raw observation
error domain/status propagate separately from those teardown results.

An unavailable boundary retains the whole concrete owner, loaded-object pins and
its original budget. A raw nonempty list of successful operations cannot certify
this boundary. The report must match the concrete owner's full resource schema,
including uncreated slots, its before/after member inventories, and the ordered
drain/release/fence operations. Missing or duplicate slots/operations, fabricated
not-created/status-less results, or remaining live resources refuse confirmation.

The bounded explicit-close port holds the whole owner before first creation. Its
manual holder cannot start the owner's automatic member destruction on a refused
close. An observation-only mismatch calls no hardware cleanup; restoring the
original ambient context may admit its first close. Once actual cleanup has been
attempted without confirmation, the holder preserves the original raw result and
refuses repeated cleanup, replacement and replay: a failed vendor destroy or
async release may already have had side effects. There is no global quarantine,
context switch, retain or reset.

The actual CUDA resource implementation has a private-copy proposal for an
explicit retirement protocol and complete 13-slot member inventory. Detailed
eigen-handle retirement preserves raw vendor statuses and does not clear a handle
on a reported failure. The explicit destructor invariant is only a final check
after closure; returning early from a destructor is not a retention mechanism.
Legacy destruction is unchanged in the proposal.

Production integration remains incomplete. The old native catch-delete and void
destroy paths must consult the owner before deletion starts and keep the handle
reachable on refusal. Python's scalar finally block currently discards local
handles after void destroy; it additionally needs a status-bearing close decision
and persistent bounded ownership before clearing any dependent handle. The
budget port currently retains the original A/B_ref/B_endpoint labels, not a new
live reservation or proof of once-only budget charging. These interfaces and the
real first-reference owner must be wired before optional support is advertised.

NVIDIA's [cuBLAS 12.9 context contract](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cublas-context)
requires the associated device to be current when using its handle and ties the
handle's lifetime to the CUDA context. A changed or destroyed context therefore
cannot be treated as evidence of successful cleanup. This is a source/lifetime
constraint, not a new GPU result.

## Rejected alternatives

- Caller-supplied context/build identities or a qualified flag: not authoritative.
- A second prepare publisher or a post-publication intent setter: breaks the
  existing atomic publication and immutable prepared controls.
- Intent equality alone as a reuse key: misses actual context/loaded-owner changes.
- Implicit cleanup context selection/retention/reset: changes ambient state and
  can hide an unavailable or failed teardown behind a fabricated success value.
- Advertising optional support before native owner wiring: creates a reachable
  path whose physical budget/lifetime contract is not yet implemented.

## Evidence and current limits

Compiled C11/C++ ABI probes demonstrated feature-absence RED, then transport,
invalid-input, legacy, by-value mutation and private-port failure behavior.
Compiled generation probes cover actual-key capture, same-owner reuse, changed
context/loaded owner, unknown ledger basis, typed API observation failures and
lease retention across unconfirmed cleanup. A retirement-query error was first
reproduced as swallowed and then corrected to preserve its original domain/status.
Pure Python probes cover frozen values, rejected inputs, no new symbol resolution
for absent intent, optional transport refusal and an independent finite copy.

A private-copy proposal also compiled the actual existing C API preparation and
prepared lifecycle translation units, with metadata-only method factories. The
baseline lacked the new shared port at runtime; the proposed shared publisher
passed by-value, failed-candidate and legacy lifetime checks. That proposal is not
applied to occupied production paths and is not ordinary SDK reachability.

Review follow-up reproduced all three actual Python batch/companion entry bypasses
with downstream work trapped, then verified early refusal and absent-intent
dispatch. New compiled owner tests reproduced partial-success confirmation,
duplicate-schema omission and uncertain-close retry, then verified fail-closed
retention. Real CudaResources/EigenHandles private-copy translation units compiled
against CUDA 12.9 headers and linked only fake CUDA/vendor functions. The original
delete path demonstrated foreign-context switching/cleanup; the proposed port
covered complete, foreign, raw vendor/buffer failure, partial-first-candidate and
unbound-empty outcomes. A failed host staging attempt omitted a header and is
retained as a compile prerequisite error, not a behavioral RED.

No full SDK build, deployment, real CUDA/vendor library execution, geometry, SCF,
solver or scientific measurement has been performed for this slice. The earlier
H100 initialization record remains tied to its own source/header/resource epoch.
The native implementation below adds physical-owner wiring, while device
qualification remains a separate checkpoint.

## Native whole-owner implementation checkpoint

The opt-in native implementation now carries copied intent and an inert shared
reference owner through the real MP2/RCCSD preparation factories and ScfOptions.
Both factories retain their incumbent validation/capacity helpers; actual context
binding remains deferred to the first ordinary physical reference. A separate
close interface avoids changing the generic method vtable. Absent intent retains
the old method, provider allowance and resource paths.

The owner retains the whole bucket before creation. Its inventory includes owning
stream, vendor resources, numeric/host buffers, both graph/executable pairs,
consumer events, source access lease and retained host payload. Foreign-context
close does not call CUDA. Failed attempted close latches the original outcome and
retains the owner/pins/original envelopes. Host-only partial creation can close
without identity queries or fabricated API zeroes. The resource close entry owns
the existing measurement mutex; graph/event extensions never reacquire it.

Source/cache reservation transfer follows logical ownership: the cache slot is
empty while a source charges its complete retained bucket, and source retirement
restores that same slot. Optional-memory retry cannot discard an unconfirmed
owner. This preserves one selected A in the existing reference and downstream
capacity helpers rather than adding a planner, voucher or second reservation.

The initial opt-in implementation uses the incumbent bounded Fock producer instead
of a separate transient resident-values owner. A forced transient policy rejects
explicitly. Its nonresident interaction source borrows the same immutable arena
fields used by the incumbent compact source and forwards to the same exact
orbit/rectangle ERI producers, avoiding an additional compact CUDA plan/stream.
This changes opt-in ownership/work, requiring future source-matched endpoint and
independent numerical/device acceptance before enabling software capabilities.

Actual native objects and whole-owner failure/identity/budget contracts have
bounded cached HOST evidence with fake CUDA/observer operations. Source
publication/reclaim now lives in the actual HF entry translation unit: a source
borrows the retained owner, and retirement returns that same bucket to its cache
slot. This integration preserves the upstream UHF out-of-memory exception path.
The registry, public ABI preparation/close ports and Python Calculator adapter
remain ignored source proposals. The preparation proposal retains the shared
native device-resource owner and its single candidate publisher; pending intent
still rejects an unknown active ledger accounting basis. Capabilities stay zero
and optional ABI/lifetime translation units remain unregistered. These proofs do
not establish an installed SDK, complete endpoint, device or scientific
qualification. Historical H100 initialization evidence remains attached to its
original source/resource epoch and is never transferred to this code.

## Remaining admission

The in-place MP2/RCCSD factories and first-reference source/lifetime wiring are
implemented in existing native translation units. B_ref/B_endpoint charging and
the concrete generation retirement protocol have executable HOST failure and
lifetime coverage. The ignored publisher/registry/close proposals must still be
reviewed and integrated before the ordinary SDK can reach this path. Software
capability enablement and device/scientific acceptance remain held. Any device
experiment needs a separately bounded source-pinned plan.

## Registered transport and H100 close qualification

The preceding unregistered-publisher descriptions record intermediate proposal
checkpoints. This change registers both optional ABI translation units in the
ordinary API target and connects the single scalar candidate publisher, registry,
context retention and Python transaction. Explicit intent remains restricted to
scalar conventional CUDA MP2/RCCSD. The feature capability result remains zero
until installed-SDK and numerical endpoint qualification; this is not a newly
advertised scientific backend.

The higher-level whole-bucket owner lives in `cuda/rhf_provider_owner.cpp`.
`CudaResources` keeps the lower-level member inventory and close operations;
it does not acquire the bucket or its source. The dependency audit retains that
boundary. Both owners use the existing neutral allocation-measurement mutex.

The current-source model-free H100 run is retained in
`benchmarks/results/h100-provider-owner-20261027/`. With a requested 512-MiB
initialization reservation and two 2-GiB envelopes it observed a 358-MiB
synchronized free-memory difference, retained the reservation once, and confirmed
the complete resource close. It does not change the legacy default. The record
separates privacy-redacted exports from unavailable original boot-ID bytes and
does not claim installed-SDK, molecular, solver, energy or force acceptance.

Current executable HOST coverage includes the actual C API publisher with the
native resource ledger, unknown-ledger refusal, metadata-handle retention and
device mismatch; Python coverage preserves native close status/detail, primary
errors, dependent handle order, cross-library retention and shared-context life.
The 36-case contract run is HOST evidence, separate from the real H100 resource
exercise and all future scientific endpoints.

## Shared historical storage after queue composition

The first final-head retention check covered the `1b755bf6d` base. Adding the
preceding queued KTBM change produced a different tree with 84,043,159 result
bytes, exceeding the unchanged 80-MiB cap by 157,079 bytes. A passing branch
check therefore cannot stand in for the actual queued composition.

The bounded repair changes only this owned engineering evidence family. V4/V6
published receipt JSON shares one TAR/XZ, and the older V4 source is a delta
against the unchanged V6 source archive. Replays and the explicit restore tool
reconstruct all three original compressed archives and verify their original
SHA-256 and lengths. No status, source, failure frame or prior acceptance is
relabelled. The V6 source remains byte-identical because V7 depends on it.

Reject raising the cap, editing another queued PR's payload, dropping telemetry
or treating privacy-redacted exports as authenticated remote originals. This
storage representation introduces no device/scientific run or acceptance. Any
later change to the anchor or compression recovery must retain exact recovery
and the unchanged V4/V6/V7 outcome checks.

## References

- Native foundation: PR #2815, merge `11b7bda884901e94c132433691106100e54b1f55`.
- Parent scope: #2809, #2727; ordinary SDK transport and science are not completed
  by the initial boundary/protocol components.
