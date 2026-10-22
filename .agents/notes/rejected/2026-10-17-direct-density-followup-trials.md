# Decision: reject three small Direct-force recurrence follow-ups

Status: rejected; retain the qualified density-first baseline
Date: 2026-10-17

## Boundary

These are follow-ups to the [qualified opt-in force reuse](../implemented/performance/2026-10-17-direct-force-endpoint-qualification.md),
not part of PR #2587 or #2588's scientific implementation. All retain exact
Direct J/K, strict FP64, original screening/convergence, grids, basis and
1800-second cold/warm5/moved/moved-warm5 populations. Real-GPU execution uses
finite Slurm allocations and assigned visibility. Logical generated work is
not a hardware FLOP count or a substitute for complete E+F measurements.

Keep the experimental branches and ignored source/force/profile receipts, but
do not cherry-pick their scientific changes into the retained implementation.
Small positive measurements below are reported, not rounded away as no benefit.

## Dependency-degree pruning

Local scientific commit `b358a2e873d007a1730fd247616b900cbded34e8` prunes the
packed Coulomb recurrence dependency closure. Emitted DAG instructions fall
1991 -> 1193 and publication levels 24 -> 19. Secondary source-matched complete
96-atom warm/moved-warm change **+0.0586%/+0.1606%**, not a speedup. Cold
changes +3.5238%, with actual 24/24 -> 25/25 iterations/Focks. All eight
secondary populations complete with independent numerical gates.

Reject: smaller emitted graphs do not establish fewer bottlenecked runtime
operations or an endpoint win. Node5's earlier candidate-only combined density+
pruning qualification cannot isolate pruning. The planned further node4 pruning
populations were replaced before execution by the prefix trial; they were
**not run**, not silently successful pruning populations.

## Separate-cache XY-prefix CSE

Local scientific commit `16ad69901a70964fe9b51201eaa598dcf75e939e` shares
weighted XY products across z powers and four density-response channels, on
the degree-pruned branch. Shared workspace grows 31,720 -> 44,680 bytes.
Node1 RTX 5090, Slurm9737, completes eight primary populations and two reverse
96-atom populations, independently checked. Warm/moved-warm initially change
**-0.9926%/-0.9350%**, but only **-0.3588%/-0.0438%** in reverse order.
Initial moved -5.9493% is actually 14/14 -> 13/13; reverse moved +6.0158% is
13/13 -> 14/14. Do not attribute those changes to prefix arithmetic.

Clean warm order-six/seven device times improve
2.437961/0.478163 -> 2.315287/0.452110 seconds, while the complete diagnostic
endpoint changes 29.997074 -> 30.151435 seconds. Counted and clean cold/warm
diagnostics all pass independent geometry-specific oracles. The generated
fixture's raw-AD/host-orbit and three sanitizers pass, with no density-consumer
spills; correctness does not rescue weak and unreproduced complete benefit.

Reject the extra shared storage and projection publication for this return.
Do not report its isolated device reduction as an accepted production win.

## Borrow the not-yet-live Coulomb frame for XY prefixes

Local scientific commit `80d5fbecc85b9eafa96a89629984d20c206a476e` is directly
based on density-first contraction, **without degree pruning**. It borrows the
double-valued Coulomb DAG frame before Coulomb preparation, keeping 31,720
shared bytes. Bra/ket projection publications and the next recurrence's
overwrite/read invariant require extra lifetime reasoning; no type-punning or
unbounded storage is introduced. Capacity and nonfinite-prefix refusal are
explicit, with the original bounded fallback before force publication.

The logical multiplication count changes `12*T -> 6*F + 4*T`, where counters
`T` and `F` respectively count entered density terms and actually prepared XY
prefix triples. Addition counts and Hermite joins are unchanged. This is
scientific CSE, not a page-width/CTA-size or enumeration change.

Node4 RTX 5090, finite Slurm1109, first matched 96-atom pair:

| Phase | Density baseline s | Prefix s | Change |
| --- | ---: | ---: | ---: |
| Cold | 244.760757 | 244.796994 | +0.0148% |
| Warm | 30.368268 | 30.227925 | -0.4621% |
| Moved | 144.777826 | 144.610768 | -0.1154% |
| Moved-warm | 30.372361 | 30.194161 | -0.5867% |

Actual counts are 24/24 cold, 13/13 moved and 1/1 every warm call in both arms.
All 24 calls independently pass. The finite secondary node2/Slurm3363 campaign
completes all eight populations and the two reverse populations. Forward
warm/moved-warm change -0.3854%/-0.3794%; reverse -0.6286%/-0.5215%.
Forward cold +3.7237% and reverse +3.0314% both take 24/24 -> 25/25.
Reverse moved -5.7582% takes 14/14 -> 13/13; forward moved has matching 13/13
and changes +0.0619%. Preserve all trajectories; an extra iteration is not
normalized away and no causal SCF explanation is inferred.

Fresh native raw-AD/host-orbit, all sixteen linked Libcint cases and generated
memcheck/synccheck/racecheck pass on both nodes with zero errors/hazards.
Combined RKS/UKS fixture consumers use 124 registers/416-byte stacks;
separate RKS/UKS use 159/141 registers and 448-byte stacks. All use 31,720
shared bytes and zero spills. This is not a production occupancy measurement.

Reject retaining a third implementation for this marginal complete gain and
additional scratch-lifetime contract. It is mathematically valid and sometimes
slightly faster; it does not close the remaining 1.58/3.78-second warm/moved-
warm gap and retains real cold trajectory variation. Do not describe the
rejection as a numerical failure or hide its small positive endpoint changes.
The initial node1 prefix allocation was canceled while pending, with zero GPU
calls; actual primary qualification runs on node4, not that canceled job.

## Revisit when

A larger reusable contraction fusion removes repeated joins/preparations or
shared publication, with a reproducible complete-endpoint improvement large
enough to justify its storage/lifetime contract. Preserve the independent
operator gates, full four-phase populations, work counts and resource fallbacks.
The residual low<=5, one-electron and separately measured grid-response owners,
and cold SCF exact-J/XC, have more measured potential than these follow-ups.

Raw evidence remains ignored under `.artifacts/direct-parity-20261017/`:
`degree-collected/`, `cse-collected/`, `prefix-collected/`, the corresponding
source exports and `profiles/`. This rationale is not a new raw-evidence archive.
