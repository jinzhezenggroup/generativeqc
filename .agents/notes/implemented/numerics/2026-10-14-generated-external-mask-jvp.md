# Decision: explicit generated forward JVP for fixed external Boolean masks

Status: implemented
Date: 2026-10-14

## Problem

Reference JVP/VJP already admits fixed external masks, and materializing native
CPU already executes finite generic FP64 `where`. Generated `linearize` still
rejects every Boolean ancestor. Merely adding a selection rule would silently
discard eager primal or derivative errors when an input direction or output is
not requested. Retaining definitions is insufficient because execution and DCE
use output reachability.

## Decision

Add `allow_external_mask_where=False` to canonical `linearize`. Explicit
selection reuses the reference external-mask classification, existing JVP
algebra and canonical `where`. Boolean input/constant selectors and their
existing type-preserving views carry no tangent storage. Computed comparisons
remain unsupported even when independent of a requested direction.

The new domain is first-order, generic finite FP64, with materializing native
CPU qualification only. Reject Boolean outputs, scientific/control domains,
FP32/mixed values, packed/precision policies and higher-order generation.
Default rejection, non-selection JVP equations and generation version 3 remain
unchanged. Reference external-mask JVP/VJP, `AD_PRIMITIVES`, generated transpose,
public compiled AD, CUDA and portable/scalar owners are not broadened.

Generate explicit zero-like directions for missing real seeds, including
constants. Retain each original live primal output and each corresponding
forward result through zero-coefficient `add`, full reduction and broadcast
edges into the published derivatives. This applies even when the requested
output itself is a pure non-selection value in a graph with a live `where`.
Zero each element before reducing: reducing large finite values first could
introduce an overflow absent from the original endpoint. Native byte/node/work
budgets include these ordinary graph operations.

## Rejected alternatives

- Numeric Boolean seeds or float masks violate the typed Boolean contract.
- Adding `where` globally to `AD_PRIMITIVES` would advertise unqualified modes.
- Keeping only the selected tangent, or retaining unused definitions, hides
  unselected primal poison and unreturned forward-domain failures.
- Keeping only primal validation misses `sqrt(0)` in an unrequested real branch:
  the existing reference JVP rejects this singular derivative with a zero seed.
- A new validity primitive/schema or lazy C++ branch evaluation would introduce
  another owner or change the eager finite-domain contract.
- A global generation-version increment or permanent generated-source hash
  tests would unnecessarily relabel/block legacy consumers.

## Evidence

Baseline `f1a4e0559b7c287cbc1dc9a64d3901066fefe31b` was exported with
`git -c core.autocrlf=false archive`; 10,786 file bytes matched committed Git
blobs. Windows reference baseline: 127 passed. Initial Linux NumPy 1.26.4 was
below the declared dependency floor and retained two FP32 reference failures;
the isolated NumPy 2.5.4 environment passed all 194 baseline cases.

Old code fails the new opt-in API tests. Initial candidate: 223 passed on Linux.
After adding the pure-output-subset guard and further negative controls, the
next run retained 232 passed and one fixture failure: GCC parsed the deliberately
invalid `-gqc-invalid-qualification-option` as a debug-level argument, so its
error did not match the test's overly specific regex. Correcting only that
regex passed the actual compiler-failure/recovery, raw ABI and owner test.
The production input bytes and generated equations did not change in that
fixture repair. Local reference/domain/discovery coverage: 143 passed;
compiler dependency audit: 538 modules, zero errors.
The subsequent normal fast-forward to `a80d04b00782e4dcfd2ac6ed7efc3c9f9e81c8cd`
adds unrelated VV10 code; qualified TensorAD inputs and the native CPU header
remain unchanged. The current dependency audit checks 539 modules with zero
errors. Linux public-contract/Ruff checking passes; Windows checking retains
its path-separator export-binding failure.

The tests execute generated shared libraries against per-index analytical
NumPy derivatives, fixed-mask finite differences and the existing runtime JVP.
They cover scalar/empty/broadcast/multi-output inputs, Boolean views, repeated
gather, name collisions, immutable detached native publication, poisoned
unselected arithmetic, unreturned derivative overflow, zero-direction
`sqrt(0)`, large finite guards, replay/optimization, typed ABI poison/alignment/
overlap, charged budgets, real cached-driver failure recovery and closed owners.

One-time old/new non-selection JVP and Boolean controls preserve serialized
programs, math/derivative/source/execution identities, resources, outputs and
actual library bytes. These comparisons remain qualification records rather
than permanent production-body fingerprints. The representative selection JVP
has generated source SHA256
`01bb1a7c145ebd2268e86b28f3e7c4fd8ea5e3672665a91a1f06c5b3e4d00f3a`
and executed library SHA256
`51a2b5989777839119a8d5f3042d2590c737d5eaac925a73a02db6f507c4480c`.
Its typed input/output extents are 104/24 bytes, required storage 720 bytes and
92 scalar work units. These are one probe's resource counts, not an endpoint
performance claim.

Raw logs, actual wrapped compiler commands, cache/version evidence, input
snapshots and libraries are retained in ignored local
`.artifacts/jvp-qualification/` and the scope-owned persistent path
`/inspire/ssd/project/chemicalreaction/czxs25220150/codex/i2137-generated-jvp-1014/`.
The broad shared `sccache --show-stats` snapshots do not measure private JIT
workers and are not attributed as this slice's cache hits.
An additional analytical selection JVP with an explicit 1.125 real coefficient
records before/after statistics while each private worker is still alive:
the actual wrapped object compile has one cache miss, and the wrapped link
has no cache hit. Both exit zero, with `SCCACHE_NO_DAEMON=1` and the exclusive
lease descriptor inherited. No cache was cleared, identity forged or hit
claimed from the shared daemon's unrelated zero counters.

After #2459 actually merged as `8ce6913f3690014e452cf73a843d279da62db0d3`,
the branch normally fast-forwarded to
`8f72a935a6cefc45473806df287066c8795bc039`. The shared inventory's delivered
mode/blob was `100644` / `a3ccee1ae78b9abfaa032414238973c7b3da02da`; no other
open PR or registered worktree owned the affected targets. The independent
entry was then added to the current schema rather than replacing the prior
proposal over upstream changes. Current inventory checking registers 61
entries and 110 controls. Full inventory/discovery tests: 124 passed; full
legacy/reference AD and selection tests: 127 passed. A separate combined host
run retained 253 passed / 34 deselected; its exclusions are not counted as
executed tests. Current source queries reproduce the old probe's three
math/source/execution/resource descriptions exactly. Qualified TensorAD,
Array API, compiler-common and CPU-header inputs have no upstream delta, so
no new CPU allocation or whole-library qualification claim was needed.

## Consequences and revisit conditions

Explicit selection pays for eager primal and forward validation rather than
pruning away domain failures. It makes no GPU, performance, new scientific
method or complete Array API conformance claim. Broader backend/dtype/mode
admission requires separate independent source-matched qualification. Refs
#2137 and #2111; this slice does not complete either parent.
