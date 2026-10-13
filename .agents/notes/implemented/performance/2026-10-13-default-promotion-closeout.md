# Decision: finish default-policy decisions without claiming product qualification

Status: implemented
Date: 2026-10-13

## Problem

#1598 explicitly permits a negative default decision, tuning-only TensorIR
winners and retained explicit Graph replay. Keeping those choices classified as
undecided makes a finite policy review look like a requirement to promote every
experimental optimization. New experiments also accumulated with #1598 as their
only owner, although their scientific and endpoint gates belong to narrower work.

This review uses master `e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c`. It changes
policy records and research ownership, not numerical code or runtime defaults.

## Decision

The authoritative inventory adds `retained-opt-in`: a completed decision to
require explicit selection. This differs from a measured negative result and
from `needs-qualification`, which denotes continuing research.

- **Incremental Direct J/K:** preserve the negative default decision already
  delivered by #2277 and its original evidence limitations. #2053 still owns
  late-SCF PBE0 scientific and profitability qualification.
- **Public precision AUTO:** retain FP64 as the default. The existing DFT
  mixed-warmup campaign supplies scoped negative evidence; historical HF and
  these DFT trials do not establish a global default. Explicit AUTO remains
  available. #1189's positive, frozen per-method product requirement is unchanged.
- **TensorIR schedule/layout winners:** retain explicit bounded tuning. The
  ordinary `plan_cuda` call uses `BASELINE_SCHEDULE`; it does not automatically
  load prior tuning winners. `tune_cuda` returns a concrete plan/artifact with
  identity-bound evidence for its caller. Preserve qualified positive historical
  results, direct-GEMM defaults and existing compiler transformations. Do not
  enable every physical schedule flag or add an unqualified global winner loader.
- **TensorIR CUDA Graph:** retain explicit `execution_mode="cuda-graph"` and
  ordinary execution by default. Historical replay gains are not a universal
  capture/resource admission rule. A global resource owner currently lacks a
  qualified retained graph-storage candidate and uses ordinary fallback. Existing
  method-owned HF graphs, explicit replay, invalidation and failure behavior stay
  unchanged.

Remaining `needs-qualification` controls acquire existing domain owners:

| Control family | Continuing owner | Gate retained |
| --- | --- | --- |
| Partial phase-local Direct values | #597; PBE0 composition #1895 | Source/resource/reuse guard, full-cache comparisons, cold/warm/moved E+F |
| Reachable Coulomb states, Hermite convolution, shell-class overrides | #597 | Shared target/schedule admission, independent derivatives, work and full endpoints |
| Standalone Lambda serial scratch reuse | #933; complete CC endpoint #1401 | Lifetime/peak bounds, complete response parity and endpoint A/B |

These are ownership mappings, not completed scientific gates or new default
promotions. The independently qualified native DF molecular-force consumer of
Lambda scratch reuse remains distinguished from the standalone default. The
inventory checker rejects future unqualified entries owned only by #1598.

## Evidence

The existing [mixed-warmup bundle](../../../../benchmarks/results/dft-mixed-warmup-20261012/README.md)
retains five alternating FP64/AUTO pairs per method for small-water complete
cold/warm/changed E+F; even the earliest tested handoff loses. Its repaired
larger cold screens bind library
`46d599c1303d7d5b6c2a6f439a16193ad7dbe80fc2513bb48401ad120c067603`
and source snapshot
`c24aa94dcf161befaebc083dab77303fdeded2f657a66c46a9a340b8c63d776c`:

| Method / case | FP64 seconds | AUTO seconds |
| --- | ---: | ---: |
| PBE0 / caffeine | 50.455724806 | 79.021784600 |
| PBE0 / water16 | 54.486693144 | 73.335884832 |
| PBE / caffeine | 50.151576690 | 70.057480000 |
| r2SCAN / caffeine | 94.795799881 | 116.226582821 |

The four repaired comparisons are single pairs, not statistically qualified
promotion trials. Returned endpoints pass their recorded residual/energy/force
checks and execute mixed J. Distinct cold guesses mean their iteration changes
cannot all be attributed to arithmetic. Historical failures, unsupported UKS
early handoff and incomplete larger/moved populations remain in the original
bundle. None of these observations is relabeled as a new current-master run or
as a formal DFT-MP-v1 row.

This review verifies all ten current `storage.json` members and the 22 exact-byte
records in `fix-captures.json.xz`. The existing [scheduler evidence](../../../../benchmarks/results/tensor-resource-scheduler-508/README.md)
and [TensorIR execution contract](../../../../docs/developer/tensor_cuda.md)
retain their positive, identity-limited results. They need not be rerun to decide
against broadening automatic dispatch.

## Rejected alternatives

Calling every opt-in a losing optimization would contradict retained positive
TensorIR results. Calling absent automatic dispatch implemented would misstate
current code. Treating this policy decision as #1189/#1190 product completion
would erase their explicit positive speed and scientific gates. Creating a new
umbrella for the same finite review would only move its backlog.

## Invariants and revisit conditions

No tolerance, physical model, supported method, resource limit or default changes.
No missing measurement becomes zero or PASS. A future promotion must supply an
identified production domain, repeated complete endpoint benefit, all setup and
refinement costs, independent scientific acceptance and deterministic bounded
fallback. New experiments belong to their domain owner rather than reopening the
completed policy review by default.

## References

#1598, #2277, #1189, #2053, #597, #1895, #933, #1401;
`manifests/maintenance/default_promotion_inventory.json`;
`tools/check_default_promotion_inventory.py`.
