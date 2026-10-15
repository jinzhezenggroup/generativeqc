# Decision: share the HF fleet item lifecycle across CPU and CUDA

Status: implemented
Date: 2026-10-15

## Problem

`FleetPlan` already owned all HF batch schedules, but its independent CPU/CUDA,
fused direct CUDA and resident DF CUDA branches repeated coordinate validation,
warm-seed selection, cold-retry decisions and successful-state publication.
Sharing the SCF providers did not make these upper-layer policies single-source.

The independent branch enclosed both warm execution and its cold fallback in
one catch/retry block. Without a preliminary-guess policy, an exception from the
cold fallback could start a third solve. The same compatibility branch retried
allocation failures. Direct CUDA's cold retry could instead throw out of the
whole fleet, bypassing per-item isolation. These were lifecycle differences, not
scientific reasons to select different solvers.

## Decision

Keep the existing `FleetPlan` as the only owner. Its common preparation and
publication closures now serve every schedule. `fleet_item.hpp` holds the
existing result vocabulary and stateless attempt/cold-retry control used by
all branches. There is no additional executor, registry, cache or allocator.

Allow one cold attempt after a failed warm attempt, never after an explicit
CUDA-driver or out-of-memory status. The fallback flag makes a second invocation
of cold-retry control inert, even if the first cold callback failed. Catch cold
callback exceptions at the item boundary. Keep the existing status conversion
and backend-specific preliminary diagnostics and work-counter accounting.

This deliberately retires the disabled-preliminary-policy compatibility that
could retry OOM or execute three solves. It does not change numerical tolerances,
the convergence/finalization rules, accepted physical state or integral science.

## Invariants

- CPU bounded workers, independent CUDA execution, fused CUDA buckets, resident
  DF plans, streams, Graphs, overlap caches and resource observations remain with
  their original owners. No transfer, fence or scientific work is introduced by
  the common item control.
- Source-order indexing, bucket identity and selected backend are established
  before validation; invalid coordinates do not bind a warm density.
- Only successful items can update their retained warm states. Frozen snapshots,
  malformed DF subset invalidation and geometry-driven source rebuilding remain
  unchanged.
- Fock-only diagnostic results do not authorize an extra SCF attempt.
- This is a bounded HF/M3 slice of #926, related to #933/#934. It does not complete
  the all-method migration, DFT response/HVP or device-resident solver control.

## Rejected alternatives

- Routing every CUDA bucket through the CPU-style per-item solve would discard
  legitimate batching/residency and change resource and performance contracts.
- Preserving independent catch/retry blocks would retain the third-attempt and
  allocation-retry discrepancy.
- A new generic runtime would duplicate the already authoritative fleet owner.

## Evidence

- Baseline: `890c425b03b312b7b23cb3251b78665313fea2b7` on `origin/master`.
- `tests/native/test_fleet_item.cpp` executes the production shared control for
  CPU/CUDA target labels, warm/cold and diagnostic cases, success/nonconvergence,
  explicit CUDA/OOM errors, exception mapping, terminal cold failures and neighbor
  isolation. It does not execute CUDA scientific work.
- `tests/python/test_preliminary_warm_retry_control.py` compiles the actual
  independent HF control with the production helper and explicit provider
  doubles. It retains 64-bit discarded-work and policy/seed diagnostics checks;
  this is host-control evidence, not a numerical oracle.
- Existing native HF/UHF/DF and public batch tests remain the endpoint gates.
  Device qualification must use Slurm on an allowed node; the local scheduler
  currently advertises only drained `node3`, which must not be used.

### Local qualification

- CPU Release build: GCC 11.4.0, scalar linear algebra, CUDA and CLI disabled.
  Verified ccache 4.5.1 and the generated fleet compilation command's launcher;
  checkout-root `CCACHE_BASEDIR` and the existing shared cache are preserved.
  Before/after cache statistics and build/test logs are retained under the
  ignored `build/cpu-release/` directory. Shared-cache totals are not attributed
  exclusively to this build.
- Eleven selected CTests pass: shared fleet control, RHF, UHF, DF, preliminary
  guesses, initial densities, warm subspaces, proposal control, ragged batches,
  C++ batch API and prepared lifecycle. Existing reference/force and convergence
  tolerances are unchanged.
- Public batch/preliminary-guess/automatic-policy/host-retry Python gates:
  161 pass, nine CUDA-named cases explicitly deselected. The 71 host retry checks
  are included in that count, not added again.
- SCF/electronic-structure/provider-selection/shared-iteration/API/runtime
  ownership regression gates: 267 Python tests pass; the corresponding five
  standalone architecture checkers also pass.
- Ruff, new-file ClangFormat, changed-line ClangFormat and whitespace checks pass.
  No CUDA build/device qualification or endpoint speedup is claimed.

## Revisit when

Another existing method needs exactly this item lifecycle, or backend-specific
failure types can distinguish untyped driver exceptions from numerical failures.
Do not infer retry safety solely from an untyped exception's public numerical
status when extending the contract to additional owners.
