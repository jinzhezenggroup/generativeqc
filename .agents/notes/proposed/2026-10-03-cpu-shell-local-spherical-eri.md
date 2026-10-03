# Proposal: consume Cartesian shell ERIs in the public spherical basis

Status: proposed; local qualification prototype, no general performance/default promotion
Date: 2026-10-03

## Problem

The existing all-s/p/d value path evaluates each canonical Cartesian component
once, scatters it into a full Cartesian tensor, then projects every ordered
public spherical output separately. HF96 source counts give 800,000,000 bytes
of Cartesian tensor plus a required 679,477,248-byte public tensor. Intrusive
preparation profiling identified allocation, full projection, and Cartesian
buffer destruction as material additional costs. This is evidence for a
schedule experiment, not a speed claim.

## Decision under qualification

Preserve generated scalar mathematics, primitive order, contracted component
normalization and component order. Share the native canonical shell traversal
between the original Cartesian consumer and a shell-local spherical consumer.
The latter reconstructs at most 6^4 Cartesian values, consumes existing
spherical expansion metadata, and scatters canonical public values eightfold.
Shell offsets come from the same metadata construction; coefficients and their
product association remain unchanged.

Only include_eri + value-only + spherical + all-s/p/d is eligible. Cartesian,
derivative, any f/g-containing system, range-separated ERIs and omitted-ERI
calls retain their explicit paths. The generic transform of externally supplied
Cartesian tensors remains an ordered linear transform, including nonsymmetric
inputs. No allocation-failure catch reruns a more expensive path.

## Invariants and resource boundaries

- The additional scratch is one invocation-local 6^4-double array: 10,368 bytes
- The existing bounded component record buffer remains; no molecular block cache
- Reinitialize exactly the current compact block extent before consuming it
- Fill only eightfold permutations whose AO indices belong to the four current
  shell slots; other shell blocks must never write into this scratch
- Same-shell AO triangles and equal ordered shell-pair triangles remain distinct
- Do not apply a global AO-pair filter to unequal shell pairs, even if i=k
- Retain conservative Cartesian-size overflow/admission checks; do not relax
  public resource policy until complete actual accounting is qualified
- Public O(N^4) output allocation remains required; its cost is not eliminated
- Existing Cartesian primitive work remains identical, with no screening,
  approximation, precision change, recurrence fork or reduced angular scope

## Floating-point behavior

Canonical public reduction preserves the prior i/j/k/l expansion-term order.
Other orbit members previously received separately rounded reductions. Scattering
one representative now enforces exact symmetry and can change their last bits.
This can affect SCF iteration counts. Scientific gates must include all samples;
iteration counts classify timing only. Universal bitwise equivalence is not an
acceptance requirement or a claim.

## Rejected alternatives

A second spherical recurrence or handwritten scientific algebra would duplicate
compiler ownership. A molecule-wide Cartesian tensor discards the immediate
consumer opportunity. A cache of transformed blocks would add capacity/lifetime
costs without reducing primitive work. Running the old dense path after allocation
failure could duplicate work or memory and is not a bounded fallback.

## Qualification evidence

The Python source/compiled-helper tests preserve explicit fallback ownership and
execute the actual local block assembly/projection functions against nonzero
synthetic integer orbit values. They exhaust every ordered 1–4-shell assignment
with Cartesian sizes 1,3,6, verifying every compact scratch slot, unchanged tail
canaries, and every public slot initialized from NaN. The independent schedule
audit also covers true public dimensions 1,3,5 and detects deliberate missing
pair-exchange and wrong AO-triangle negative controls.

The linked native test compares every spherical element with the retained
Cartesian-then-transform route, and checks independent RawSource mixed signed
contractions plus dddd, all-equal and equal-pair blocks. Both normal and reversed
shell storage are tested. Physically independent PySCF/libcint endpoints and
force consumers remain separate required gates. A larger system, clean complete
cold/warm/changed-geometry timings, semantic work and actual resource accounting
are required before any general promotion.

The initial bounded gate passes six Python controls and eight linked native
controls. Across every tested tensor, the maximum difference from the retained
Cartesian transform is 3.33e-16; independent RawSource differences are at most
8.33e-16. The generic transform requires physical ncoord metadata even with
optional derivative arrays absent; the test adapts this metadata explicitly,
without modifying the generic production function or numerical thresholds.

An interposed C++ allocation ledger on four d shells (24 Cartesian / 20 public
AOs) records the original spherical route's 3,988,832-byte tracked heap peak,
including both rank-four tensors, versus 1,334,664 bytes in the candidate.
The candidate has exactly one 1,280,000-byte public tensor allocation and no
2,654,208-byte Cartesian tensor allocation. This is requested C++ allocation
accounting, not RSS or complete process/stack memory; the new 10,368-byte stack
scratch is separate. Cartesian controls have a 2,701,440-byte tracked peak in
both binaries. Skip-ERI probes in both representations allocate neither rank-four
tensor, and native tests require identical one-electron outputs.

All local attempts, counter scripts, raw logs and exact source/binary identities
are retained in the experiment's sibling evidence directory. Initial standalone
coverage executed correctly; its first Python assertion had an incorrect expected
block count (5016 versus 5079). That test expectation was corrected without a
production change or tolerance adjustment.

## References

- `2026-10-01-cpu-eri-shell-geometry-reuse.md`: retained scalar producer and order
- `docs/maintainer/performance_engineering.md`: endpoint/work qualification gates


## Follow-up qualification and publication integration

The frozen implementation is 854a7b7 on experimental baseline bdcbc008. Its
projection-only source is integrated separately onto master 30dece25; the
unpublished HF orbital-policy prototype is not a dependency of this change.
Neither measured projection ancestor includes the separate trace-precision fix.
Fresh current-master builds/tests must retain their own identities and are not
relabeled as the measured binaries. The host source fixture now explicitly skips
with a missing-ccache prerequisite rather than silently compiling uncached.

Independent checks cover HF96 prepared/export physical states, smaller HF
forces/lifecycle/finite differences, and direct spherical PBE/PBE0 RKS/UKS
states/gradients/finite differences. Retained standard UHF24 torque failures,
PBE24 strict snapshot/work-budget rejections, and empty-beta oracle-contract
failures remain limitations; passing separate tighter-control cases do not waive
the standard gates. No large force/FD, physical OOM or vendor-workspace claim is
made. The additional 10,368-byte Cartesian block is incremental scratch; the
existing component array and generated primitive stack remain.

A process-first comparison initially had unequal source-tree Python bytecode
caches. Cross-binding libraries and Python trees established that the 16–17 ms
penalty followed lazy Python compilation. Original receipts remain retained and
cache-confounded for cold attribution. Corrected six-pair cohorts give both
variants an absent external PYTHONPYCACHEPREFIX with bytecode writes disabled;
every process records and checks this condition. The reference-policy HF96
warm medians are 7.915773 to 6.746130 s with 16 iterations in both arms; peak
process RSS is 1,498,532 to 718,854 KiB. Explicit OpenBLAS medians are 6.230007
to 5.614910 s with 16 to 17 iterations and RSS 1,502,958 to 722,168 KiB.
These are separate experiments, not compounded gains. Full HF operator/history
telemetry remains unavailable.

All 288 corrected endpoint energy gates pass. All small/control losses remain:
reference Cartesian cold/warm/moved +6.33/+4.10/+3.99 percent; OpenBLAS small
spherical cold +8.52 percent and Cartesian warm +0.68 percent. These controls
preclude a blanket speed claim. Raw receipts are local pending separately
approved compact evidence admission; a checksum alone cannot restore lost raw
bytes.

## Frozen main-only three-arm evidence

`benchmarks/results/cpu-shell-projection-20261003` retains the compact selected
scalar evidence and portable replay for baseline 30dece25, projection 2bda89f3 and
metadata 8ac08f0a. Their public exact-tree aliases are explicit; later integration
builds are separately identified and never relabeled as measured binaries.

The balanced six-permutation campaign records 54 fresh processes / 216 complete
energy endpoints. All recorded 16 iterations and pass unchanged independent energy,
convergence and density-RMS gates. For four waters/12 atoms/96 spherical AOs,
warm medians are 8.305469, 7.285550 and 6.914398 seconds; directly measured combined
reduction is 16.7489 percent. Median cumulative process peak RSS is 1,498,968,
718,936 and718,544 KiB. Full Fock/operator/history telemetry remains unavailable;
formal performance/default-promotion evidence is inconclusive.

Every small/control loss is retained: combined 24 AO cold is 6.6485 percent slower;
metadata versus projection 24 AO cold/warm/moved is 10.4325/3.9355/2.4377 percent
slower. Projection Cartesian cold/moved is 7.5075/2.4306 percent slower. Existing
historical failures, initial Python-cache confounding and later corrected cohorts
remain independently labeled. Full raw arrays are local; checksum anchors do not
provide recovery. Current-main independent prepared/export checks have their own
source/library identity and do not create new timing claims.
