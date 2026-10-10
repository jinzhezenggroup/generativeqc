# Decision: VRAM-guarded FP64 Lambda defaults

Status: implemented; explicitly authorized conditional adoption
Date: 2026-10-09

## Problem

The [original opt-in batch/replay investigation](2026-10-09-df-lambda-batch-fresh-replay.md)
showed lower complete-endpoint and Lambda times for batch thirty-two plus
original-equation matrix fresh replay on the qualified 230-AO workload. It also
reserved more Lambda device capacity. A complete numerical budget alone cannot
establish that the currently available GPU memory admits that reservation.

The user authorized making the faster schedule the default when VRAM permits.
That authorization is not a five-pair shared performance qualification, and
does not change the historical cohort's numerical-only publication decision.

## Decision

Request Lambda auxiliary batch thirty-two and matrix fresh replay in both the
standalone options and complete force owner. Keep CCSD batch eight, strict FP64 W,
the standalone GMRES cadence, the complete owner's cadence thirty, all original
equations and every scientific gate unchanged.

Query free device memory before admitting owner-local storage. Bound admission
by the smaller of that snapshot and `df_max_device_bytes`, an optional independent
device ceiling. Charge optional cuts, ordinary matrix batches, shared replay/audit
storage, immutable core retention and the conservative matrix-provider allowance
against that limit, as well as the existing complete host/device numerical
budget. Clamp the batch to Q and halve it until ordinary matrix storage fits.

Admission remains granular: if enlarged shared replay storage is refused, retry
the original audit-only arena. Preserve scalar fresh replay, scalar expanded
audit and scalar operator execution as bounded fallbacks. A post-preflight CUDA
allocation shortage drops core retention, then the shared audit/replay arena,
then ordinary matrix storage, then optional cuts. Failure to fit mandatory scalar
storage aborts without publishing a response. Non-resource CUDA, binding,
nonfinite and numerical failures are not fallback signals.

Report the actual free-memory snapshot and admitted device ceiling separately
from reserved capacity. None of these is a measured peak. A snapshot cannot
eliminate races with another GPU consumer, so allocation-time fallback remains
necessary even after successful admission.

Preserve experiment identity by passing explicit replay zero/one in all ablation
commands. Add `--default-candidate` for a forward/reverse two-cell comparison of
the old batch-eight/scalar-replay policy and the new conditional request.

## Rejected alternatives

- Unconditionally reserving batch thirty-two ignores actual available VRAM and
  makes a complete numerical budget an unreliable device admission criterion.
- Changing the complete numerical budget into a device-only budget would drop
  simultaneous host storage from the existing scientific resource contract.
- Treating all CUDA errors as resource refusal could hide binding or numerical
  regressions behind a slower schedule.
- Calling the historical two-pair timings a formal performance promotion would
  skip the required five pairs, measured memory/compile cost and complete
  iteration-history evidence.
- Rewriting the historical opt-in note or pooling its old binary measurements
  with the current-master default qualification would destroy provenance.

## Invariants

- Fresh replay lowers the original virtual residual from immutable amplitudes
  and DF factors; it never borrows accepted residuals or staged solver cuts.
- The original retained-core replay and independent expanded Lambda audit remain
  mandatory; no gate is relaxed to admit a faster schedule.
- Shared replay/audit storage is valid only for disjoint owner-stream lifetimes.
- A scalar fallback is explicit, budgeted and reports actual schedule admission.
- Oracle/PySCF data is validation-only and never enters production execution.

## Evidence

Host/native fixtures cover both actual omitted defaults and explicit old
controls. Real-device tests cover full and partial Q batches, source/no-source,
a device-only ceiling selecting batch eight with scalar audit/replay, and
failure-without-publication when mandatory storage exceeds the device ceiling.
The ceiling tests compare the retained response bit-for-bit to the same selected
old schedule; ordinary default parity retains the existing independent tolerance.

Current-master qualification is retained separately from the historical opt-in
cohort. Both clean complete-endpoint observations and separate instrumented
observations are numerical acceptance/descriptive timing, not formal promotion.
GPU qualification passes 102 tests plus 17 memcheck cases with zero errors;
the separate host/default/lifetime/evidence regressions pass 210 tests. See the
[current-master evidence bundle](../../../../benchmarks/results/df-lambda-vram-defaults-20261009/README.md)
for frozen identities, source reconstruction and the complete-endpoint gates.
Raw logs, source archives and binaries remain ignored local artifacts; the
existing aggregate and per-file evidence caps remain unchanged.

## Revisit when

Reconsider the requested batch if qualified complete endpoints regress on other
Q/occupied/virtual shapes, or a full shared performance qualification supports a
different schedule. Revisit the device admission policy if provider workspace
requirements change, or concurrent-consumer workloads demonstrate that a fixed
additional headroom policy is needed. Independently owned replay would still
require a separate lifetime proof, budget and explicit resource fallback.
