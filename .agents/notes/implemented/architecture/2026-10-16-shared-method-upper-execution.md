# Decision: Share correlated batch and physical-reference control across methods

Status: implemented
Date: 2026-10-16

## Problem

The shared prepared lifecycle did not remove four method-local conventional
batch schedulers. MP2, UMP2, RCCSD and RCCSD(T) independently implemented input
ordering, geometry replacement, warm-state restoration, endpoint publication and
error isolation. Restricted and unrestricted reference owners also repeated
physical-export acceptance. The correlated warm retry loops caught every
exception, unlike the safer KS policy: OOM or an asynchronous CUDA error could
therefore submit another solve rather than propagate its original failure.

## Decision

Replace the four batch owners with `methods/correlated_batch.cpp`. Keep each
registered method factory as the scientific/backend/domain admission authority.
Use a narrow `ReferenceSeededCalculation` port implemented by the existing
calculation owners; it accepts optional last-good HF state, returns an optional
candidate, and transfers only compatible reference execution storage on rebind.
The shared dispatcher does not own a second solver, source, context or cache.

Reuse `correlated_warm_reference.hpp` for one reference acceptance and bounded
warm/cold policy. MP2/UMP2, RCCSD and unrestricted CC call it with their existing
providers. RCCSD(T) and DF CC consume the common RCCSD native-state preparation
and inherit this same acceptance. Release a rejected proposal before allocating
one cold replacement. Reuse KS's typed retry distinction rather than correlated
catch-all behavior: resource, driver, unsupported-capability and unknown runtime
failures are terminal, and a cold failure is never retried.

Move geometry admission/packing and item exception classification to common
method helpers. KS retains its existing CPU/CUDA scheduling and only consumes
these controls. GFN2 retains its already-shared runtime request/bridge; it uses
the shared coordinate packing without rebuilding its CPU/CUDA SCC leaves.

## Rejected alternatives

- A new universal execution engine, generic callback graph or additional
  lifecycle/cache would duplicate existing owners and erase useful placement
  distinctions. Keep `PreparedCalculation`/`PreparedBatch` and native providers.
- Merging CPU loops and CUDA streams/graphs into a per-operator host schedule
  changes execution/data movement without scientific justification.
- Calling all errors seed failures risks retrying pending device work and hides
  resource or capability errors. Preserve their typed status and exception.
- Pretending unsupported UMP2/UCCSD CUDA or GFN batch combinations now exist
  confuses upper-control consolidation with new scientific capability.

## Invariants

- Model, approximation, spin, source identity and physical tolerances stay with
  the original scientific owners; no CUDA request executes a CPU oracle fallback.
- Candidate warm state commits only after the complete requested endpoint
  succeeds. Rebind, correlation and force failures preserve last-good state.
- Checkpoint restore validates every supplied state before changing any seed;
  unspecified slots keep their existing state. Frozen seeds can be consumed but
  are not replaced.
- Reusing a CUDA executable is independent of density warm-start admission and
  never authenticates a physical reference at a different geometry.
- Preserve method-specific bucket IDs, optional diagnostic facets, admission and
  resource observations. Unknown resources remain unknown rather than inferred.

## Evidence and qualification boundary

- `tests/native/test_method_execution.cpp` executes the actual shared reference
  helper and batch dispatcher with typed provider doubles, including terminal
  failures, unsuccessful-reference retirement, placement metadata, source order,
  rebind recovery, frozen seeds and transactional checkpoints. Its CUDA label is
  not real-device evidence.
- Existing native MP2/UMP2, batch, lifecycle, C++ API and density-fitting contracts
  exercise the retained production CPU paths. Public MP2/UMP2, RCCSD/RCCSD(T),
  UCCSD and GFN2 tests retain their independent pinned numerical references and
  finite-difference gates; no tolerance is relaxed.
- The preliminary/KS host-control harnesses consume the moved production failure
  policy and geometry binder rather than maintaining old local implementations.
- Eight changed upper-level C++ translation units also compile with
  `GENERATIVEQC_HAS_CUDA=1` against CUDA 13.0.88 headers, through the verified
  compiler cache. This checks host-side CUDA branches, not device compilation,
  linking, numerical parity or real-device execution. The default system CUDA
  12.4 headers lack the existing batched cuSOLVER API required by master.
- GFN CPU execution needs an LP64 OpenBLAS provider on the loader search path.
  The installed `scipy_openblas32` library supplies it for local tests; the
  scalar Gaussian build setting does not remove this GFN runtime prerequisite.
- Real-GPU qualification requires a permitted Slurm allocation. The local
  scheduler advertised only drained maintenance node `node3`; no real GPU work
  was run outside Slurm or on the prohibited node.

## Consequences and revisit conditions

The four scientific calculation classes remain distinct, but their batch policy
has one owner. Reference provider adapters retain backend-specific storage,
budgeting and completion behavior. New native batch methods can opt into this
port only if their admission, seed layout and complete-endpoint contract match;
otherwise retain an explicit method-specific schedule. Revisit the port when a
real new consumer needs another capability, not to anticipate unsupported paths.

Related tracking: #926, #933 and #934. The separate HF fleet work remains in
PR #2536; this change is based on master and does not depend on that branch.
