# Decision: execution-local CUDA KS retry work census

Status: proposed (native qualification pending)
Date: 2026-10-12

## Problem

A prepared KS solve may discard a nonconverged warm attempt and retry cold.
The returned KS history and Fock counter describe only the returned attempt.
Initial-guess diagnostics may also retain discarded work, but are optional;
`initial_guess=None` must not make complete endpoint work unknowable by design.

## Decision

Preserve returned-attempt histories, physical counters and precision audits.
Certify an ordinary device-fused CUDA attempt's physical Fock count separately
from convergence. Its terminal build ledger must match the actual counter,
including post-SCF warm-energy rebasing. A normally exhausted attempt may have
a complete physical ledger without a final-state audit; do not fabricate one.

The prepared owner begins execution-local accounting before each attempt.
Complete normally returned attempts contribute to the same total. An exception
leaves pending work; missing inventory, overlapping attempts, unmatched
completion or overflow permanently revokes that execution's total. A successful
retry cannot repair it. Each new execute owns a fresh accumulator.

An optional internal owner distinguishes incomplete instrumented work from
legacy uninstrumented work. Publication never falls through from an unavailable
owner total to the returned retry's partial count. Legacy CPU and other owners
keep their guarded behavior. No public ABI changes. Preliminary-method work
remains separately diagnosed; chunk/replay and host-unfused CUDA schedules remain
unavailable rather than receiving inferred totals.

## Rejected Alternatives

- Iterations omit physical rebase/finalization and speculative work.
- Summing only successful attempts erases discarded work.
- Reusing final precision flags for nonconvergence falsifies the final audit.
- Treating exceptions as zero work cannot prove that no kernels were submitted.
- Removing the retry publication guard without an owner census publishes partial work.

## Evidence

The host accumulator regression compiles the real header directly and exercises
zero, multiple attempts, incomplete/exceptional work, overlap, unmatched
completion, the exact uint64 limit, overflow and fresh-execution independence.
Existing publication/invalidation probes retain legacy fail-closed controls.
Three focused executable host tests pass. Opt-in scheduled-GPU tests exercise
curated RKS/UKS real nonconvergence/cold retry, exception/cold retry, subsequent
replay, failed-to-converge physical counts and unsupported schedule controls.
They have not yet been qualified against this exact phase source.

## Consequences

Ordinary prepared CUDA KS may expose complete target work even with discarded
normally terminated attempts. This does not certify mathematical correctness,
promote automatic functionals/forces, alter convergence, or imply complete
operator inventory for a nonconverged returned state. Existing historical force
receipts retain their returned-attempt caveat until independently rerun.

## Revisit When

Other owners acquire independent complete physical work inventories, or a public
counter can distinguish known zero from unavailable without ABI incompatibility.

## References

- #1118, #1121 and #1124.
- `docs/developer/ks_diagnostics.md`.
- `tests/python/test_target_fock_work.py`.
- `tests/python/test_cuda_ks_retry_work.py`.
