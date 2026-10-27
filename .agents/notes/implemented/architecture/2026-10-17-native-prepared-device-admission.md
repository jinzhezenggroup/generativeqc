# Decision: prepared native owners bind their own device admission

Status: implemented
Date: 2026-10-17

## Problem

Selected-inventory admission already used the native sparse lifetime sweep and
actual native buffer/provider owners, but Python's observation scope still
controlled ledger binding. An installed C++ SDK or CLI could not independently
hold that same admission contract through preparation and execution. Merely
removing the observer would also lose its serialized host-workspace policy.

## Decision

Share the existing inventory admission factory between the private metadata ABI
and a typed public resource handle. Single calculations and batches retain that
shared ledger before the scientific plan member, so plan/provider/buffer
destruction releases final charges before the ledger owner. Native RAII scopes
bind preparation and execution; metadata handles may close independently.

Same-owner nesting borrows counters; foreign or concurrent bindings refuse.
Optional frontend observation is not authority. Native scope depth enforces the
same serialized workspace even after an optional observer ends. An otherwise
unobserved operation owns an isolated native sampling scope and saves the samples
in its diagnostic. Existing outer observations are borrowed without claiming an
isolated native peak. Missing samples and requested-byte overflow stay unavailable.

An unbind request made while a native scope borrows an outer binding is deferred
until the outermost native borrower returns, including exception unwinding. The
request must not simply be discarded: if the original observer ends first, a
borrowed scope does not own the binding and would otherwise leave it permanently
active. An explicit thread-local pending request distinguishes that case from a
still-live outer owner. Observer activity alone cannot establish ownership, since
explicit native bindings can exist without an observer.

The Python adapter selects the public path once. Only a missing preparation
symbol permits old-library compatibility; rejection, partial ABI or malformed
publication cannot trigger private creation or unbudgeted preparation. CLI
`run --owned-device-bytes` exposes only numeric/provider-reservation admission,
not the misleading promise of a global device/RSS cap.

## Rejected alternatives

- Keep binding in Python: does not satisfy installed SDK/CLI runtime ownership.
- Copy inventory arithmetic into another planner: creates competing authority.
- Reuse frontend observers as a mandatory native scope: retains the original
  dependency and can disable serialized workspace when optional observation ends.
- Name the CLI option a total device budget: runtime/library internals, physical
  coverage and global/model validation are not supplied by this ledger.

## Invariants

No scientific equations or capability bits change. Retained numeric charges and
provider reservations remain distinct, survive warm calls and metadata closure,
and follow actual buffer release, including cross-thread destruction. Failed
admission is not an alternative backend or an excuse to omit a budget.

## Evidence

Behavioral coverage lives in `tests/native/test_resource_ledger_plan.cpp`,
`tests/native/test_device_resources.cpp`,
`tests/python/test_native_prepared_resources.py`, and the installed-CLI controls
in `tests/native/test_cli_runtime.py`. Host metadata and synthetic observation
tests do not establish real-GPU numerical or endpoint performance acceptance;
source-matched scheduled CUDA qualification remains a separate gate.

Failure injection must constrain the selected constructor before native
preparation retains its owner. Replacing only the Python metadata handle after
preparation intentionally cannot alter that retained authority. Public-native,
private inventory and legacy constructor tests therefore select their paths
explicitly; compatibility injections must first disable the public prepared
entry point. This preserves old-library coverage without mistaking an inactive
private symbol for a tested public admission boundary.

Offline generated CUDA providers also reach the ledger through relative tensor
headers, without CMake's source-root include directory. The ledger's observation
header therefore uses a sibling-relative include. A standalone C++17/C++20 probe
without `-I src` protects this boundary; a full CMake build alone hid the failure
later exposed by the independent XC CUDA compile job.

## Observation replacement during native execution

Fixing deferred device unbinding alone does not establish CPU sampling ownership.
A synchronous callback may successfully end a native-first CPU observation and
begin a replacement before returning to the same native scope. Restoring a saved
snapshot solely because the scope originally created an observation erases the
replacement, attributes its samples to the wrong native interval, and makes its
later private-bridge end fail. Rejecting such a replacement would change the
existing active/inactive observation contract rather than preserve ownership.

CPU observation begins therefore reserve nonzero thread-local generations from
a counter that is independent of restored snapshots. Native scopes publish and
restore only their own generation. A replacement remains live with its own CPU
and CUDA-arena samples; native diagnostics leave that replaced interval
unavailable instead of claiming the replacement's work. Retained-host-capacity
scopes obey the same generation boundary when restoring their predecessor's
bytes. Generation exhaustion refuses entry before binding a native owner and
never wraps into an old interval. This changes neither public ABI signatures nor
scientific equations, and the native serialized workspace policy continues to
depend on native depth rather than optional observer activity.

`test_resource_ledger_plan.cpp` exercises native-first/observer-first,
direct/synchronous-callback, return/unwind and both replacement worker policies,
plus retained-capacity replacement and isolated generation exhaustion. These
host lifecycle controls do not transfer any prior source's real-GPU acceptance
to the changed runtime; immutable older qualification inputs remain historical.

## Installed compiler dependency closure

The independent current-head Python CI exposed a second packaging boundary:
the ledger's new sibling-relative observation header was absent from the wheel
payload. Restoring a source-root include would merely hide the missing installed
input and break standalone consumers again. Package the canonical native header
and include it in every tensor, AO/grid, XC, integral-derivative and stationary
consumer's explicit compiler dependency inventory instead. This changes compiler
artifact provenance, not the
scientific IR or equations; old cache entries must not survive an observation
header edit.

The wheel-only local-include test reproduces the missing-header failure. After
only adding the payload, executable tensor/XC controls still reproduce stale
artifact identities when that installed header changes. The complete fix must
pass both controls, as well as the compiler structure checks and wheel-only
grid compile/link test. These tests compare behavior after a dependency edit,
not a permanent expected source fingerprint. Existing immutable source builds
and qualification inputs remain attached to their original revisions.

The stationary post-link AOT manifest has a separate CMake dependency filter.
It must include the same observation header as the compiler's compatibility
identity; otherwise changing that header leaves an incrementally built manifest
stale even though a clean build hashes it correctly. The real CMake regeneration
control mutates this header, verifies that the manifest identity changes, and
checks that unrelated edits and subsequent no-op builds do not regenerate it.

## Typed refusal at the single HF boundary

The first installed-SDK GPU qualification of `077ad249b` failed its low-capacity
gate after the exact-source CUDA build succeeded (Slurm jobs 1759/1760).
Independent public-header probes of the preserved SDK, in scheduler-owned fresh
processes, separated status from ledger evidence: both RHF and UHF rejected a
one-byte capacity with a positive rejection count and no live charges. RHF
returned `OUT_OF_MEMORY`; UHF returned `NUMERICAL_FAILURE`, cold and after a
generous-capacity execution, for both energy-only and energy/force requests.
The failed test's combined assertion did not establish an unbudgeted retry.

The UHF single endpoint discarded the bucket's typed OOM in its generic failure
branch. Preserve that status through `std::bad_alloc`, as the RHF entry already
does, so the public exception mapper returns OOM without changing admission or
scientific execution. Regression coverage must check the status, rejected
allocation receipt, and final numeric/provider release separately, before and
after a successful request. Do not accept numerical failure as an equivalent
refusal or introduce an unbudgeted fallback to make this gate pass. Historical
failed qualification and frozen inputs stay unchanged; the fix needs a fresh
source-matched qualification.

## Consequences and revisit conditions

This removes a frontend lifetime authority without pretending to finish #2227
R2. Global selection/model validation, provider-internal physical coverage and
full allocation/RSS authority remain separate work. Extend the public contract
when those native authorities exist, preserving explicit old-library behavior
and avoiding a second planning engine. R3 response/HVP remains outside this slice.
