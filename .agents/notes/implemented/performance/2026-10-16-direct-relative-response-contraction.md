# Decision: fuse Direct force relative-pair response contractions

Status: implemented on the unmerged #2558 branch; complete endpoint and device
qualification pass. The dependency stack remains unmerged.
Date: 2026-10-16

## Measured problem

The source-matched strict-FP64 r2SCAN-3c/96 diagnostic on Slurm node4, job 1023,
uses the #2556 image (source identity
`730c1f1c4f1d334af6749d5b8b966fb3c7b48aa689542b7e8e9449d89fc800c1`).
Its cold public call executes 24 SCF iterations and 24 physical Fock builds.
Device activity is 198.426592 s Direct J and 108.234467 s XC/AO/grid. These are
device times, not additional independent wall-time owners. The complete cold
diagnostic is 382.661019 s, including 8.839830 s preparation. The exclusive
public wall ledger reconciles exactly; final-state validation is 0.605311 s,
and final-state snapshot/decode are separate owners. No convergence trajectory
is normalized away.

The warm diagnostic has one iteration/build and completes in 62.264712 s.
Order-six, order-seven and order-eight force device times are respectively
13.785337, 7.308562 and 5.772689 s. Launch markers are asynchronous: the
`direct_force_classes` wall owner includes their shared synchronization, so
individual kernel device times must not be presented as exclusive host wall
times or added to that wall owner.

Order seven already convolves Hermite axes, but prepares pair derivatives and
contracts the same primal coefficients separately for each independent atom.
Order eight additionally retains the six pair-index contraction loops. Both
repeat scalar data and publish/retire ket responses between atom contractions.

## Decision

Store one scalar relative-center Dual per Hermite axis. Convolve each axis into
three channels: the primal coefficient, its bra relative response, and its ket
relative response. Contract nine response sums (bra, ket, P-Q, each xyz) once,
then apply the constant nuclear Jacobians to all independent atoms. Prepare
those Jacobians once per primitive product, not once per AO component/lane.

The compiler owns this shared schedule in
`direct_relative_response_cuda.py`. The existing cooperative Coulomb DAG and
materialized scalar Coulomb recurrence remain the radial owners. Gaussian
decay stays in the incumbent pair algebra, without division by Gaussian
factors. Both consumers retain screening, density/source weights, ordered
primitive traversal, independent-center ordering, negative force publication
and the last-center translational scatter. Resource/precondition rejection
continues to use the existing fallback owners.

The materialized response workspace decreases from 26888 to 11080 bytes;
the previous 32/44-KiB materialized/cooperative static bounds remain unchanged.
This is a contraction/recurrence reuse change, not an order-seven page-width,
CTA-size or order-eight enumeration experiment.

## Independent checks

- A compiled host probe checks 29160 normalized primitive atom/axis responses
  across order-six/seven/eight partitions, axis orientations, diffuse/tight
  exponents, common/repeated atom centers and underflowed Gaussian decay.
  Retained AD differs by at most 1.11022e-15; an independent displaced-value
  recurrence differs by at most 1.014e-12 after normalization. Translating the
  displaced oracle to its first center avoids roundoff in common-center odd
  integrals; a finite-difference cancellation bound is reported separately.
- The existing real-device fixture checks complete signed forces against raw
  AD and an independent host symmetry orbit for RKS/UKS, combined/separate J/K,
  inactive and screened work, repeated atoms and triangular shell pairs.
  Slurm job 1024 on the assigned RTX 5090 passes derivatives, memcheck and
  synccheck, with zero sanitizer errors. Actual preparation/contraction
  counters now describe one response-basis contraction rather than one per
  independent atom.
- The focused final host suite passes 47 tests; compiler ownership checks
  report 542 modules and zero dependency errors. Overlapping suite counts are
  not added as disjoint tests.

The complete endpoint qualification below establishes the retained advantage.
The candidate native identity is
`30657c71068e5eb14ce73b33256571ea9442512e4ee2ace2e8d2ac2d40beb201`.
Its four changed native inputs are the shared response module, the two
consumer generators, and their explicit CMake dependencies (1754 total).

## Rejected trials and limits

The intermediate order-six axis-polynomial prototype reduced pair-4/2 probe
time on the Slurm pro6000 resource by about 13%, but regressed pair-3/3 by about
1%. It is not retained or claimed as a complete endpoint optimization. A
28-box scalar-DAG trial was slower and duplicates an already rejected approach
documented in the earlier recurrence-reuse note; it is also removed. Their raw
patch, compile/resource logs and device observations remain in
`.artifacts/direct-axis-next/`. NVML metadata failed on that secondary node;
those observations are not primary RTX-5090 qualification.

Master advanced from `33d73af64` to `f0fe6b06f` during this work. The new changes
concern Fast nonlocal refinement, selected-native registry binding and XC
documentation, not this strict force contraction. Qualification remains
explicitly bound to the #2556 dependency source and its candidate; it is not
relabeled as a current-master measurement. #2553 remains a separate unmerged
stationary-convergence repair and is not silently mixed into this comparison.

An attempted supplementary #2556 packaged-AOT test reaches its independent
energy/force gates, then fails because its Python force-work spy remains empty
on the native force route (`force_work[-1]`, IndexError). The failing baseline
test is retained, not suppressed or described as a candidate correctness pass.

## Complete source-matched qualification

Slurm node5 job 1645 executes eight complete baseline/candidate populations
on one scheduler-assigned RTX 5090, within a finite 90-minute allocation.
All retain the 1800-second population timeout and cold + five warm + moved +
five moved-warm calls, original strict FP64 screening/convergence/reference
protocol and unchanged canonical basis/moving grid. There is no trimming,
prewarming, device-visibility override, compiler-cache reset or iteration
normalization. This compares the #2556 source to this force change, not to
current master; #2553 is not included.

| Method/atoms | Endpoint | Baseline s | Candidate s | Change |
| --- | --- | ---: | ---: | ---: |
| r2SCAN-3c/96 | cold | 383.888035 | 373.634581 | -2.671% |
| r2SCAN-3c/96 | warm | 62.350698 | 51.800627 | -16.921% |
| r2SCAN-3c/96 | moved | 232.498299 | 221.635959 | -4.672% |
| r2SCAN-3c/96 | moved-warm | 62.499865 | 51.678825 | -17.314% |
| r2SCAN-3c/48 | cold | 68.904710 | 66.426167 | -3.597% |
| r2SCAN-3c/48 | warm | 14.402611 | 11.680185 | -18.902% |
| r2SCAN-3c/48 | moved | 50.309942 | 47.564831 | -5.456% |
| r2SCAN-3c/48 | moved-warm | 14.428178 | 11.685657 | -19.008% |
| PBE0/48 | cold | 22.781601 | 22.521541 | -1.142% |
| PBE0/48 | warm | 4.274582 | 4.054773 | -5.142% |
| PBE0/48 | moved | 22.443889 | 22.335013 | -0.485% |
| PBE0/48 | moved-warm | 4.272208 | 4.070798 | -4.714% |
| HF/48 | cold | 11.183760 | 11.179372 | -0.039% |
| HF/48 | warm | 0.749126 | 0.749382 | +0.034% |
| HF/48 | moved | 7.265011 | 7.271700 | +0.092% |
| HF/48 | moved-warm | 0.749186 | 0.749792 | +0.081% |

Warm entries are medians of all five retained calls. Cold/moved include
preparation. All convergence counts match: r2SCAN96 cold/moved 24/24 and
13/13 iterations/Fock builds; r2SCAN48 19/19 and 13/13; PBE0 17/18 and 12/12;
HF 21/22 and 16/17. Every warm/moved-warm call remains 1/1. Extra physical
Fock builds are retained, not normalized away. PBE0 cold is a single matched
observation, not a claim of statistically reproducible cold improvement.

All 96 primary and four diagnostic calls independently pass the original
1e-8 Eh / 1e-7 Eh/Bohr gates. Maximum errors are 3.865352482534945e-11 Eh
and 6.014522213604323e-11 Eh/Bohr. No population is truncated. Six final-linked
Libcint checks pass. Native preparation-count assertions are refined to permit
further reuse within retained work bounds, not mandate exact internal calls;
all numerical/screening assertions remain. The refined fixture compiled from
the final generated headers additionally passes on Slurm job 1646, including
memcheck and synccheck with zero errors. It follows, rather than overlaps,
the timed endpoint job.

### Matched device work and resources

| Warm device kernel | Baseline s | Candidate s | Registers | Dynamic shared bytes |
| --- | ---: | ---: | --- | --- |
| order6 spatial | 13.804019 | 13.865095 | unchanged | unchanged |
| order7 cooperative | 7.363464 | 2.018463 | 214 -> 173 | 42984 -> 27176 |
| order8 materialized | 5.810217 | 0.455951 | 224 -> 150 | 26888 -> 11080 |

Order6 arithmetic is unchanged; its observed device variation is +0.442%.
Order7/8 device reductions are 72.588%/92.153%. Nsight reports zero local
bytes/thread for the changed kernels. Existing static resource limits remain.

| Order | Bra preparations | Ket preparations | Component contractions |
| --- | --- | --- | --- |
| 7 | 2606140 -> 1027752 | 3147016 -> 1250544 | 1937135584 -> 754811008 |
| 8 | 231424 -> 95976 | 231424 -> 95976 | 270222624 -> 107642016 |

Primitive products remain 2400504912/1250544/95976 for orders6/7/8. Coulomb
preparations remain 1250544/95976 for orders7/8. Enumerations, screening and
published component counts are identical. The contraction counter now counts
one shared nine-response basis rather than one atom's response: these are
actual preparation/contraction invocations, not hardware FLOPs. Hardware FLOP
counters remain unavailable (`ERR_NVGPUCTRPERM`), so no invented FLOP reduction
is claimed. Convolution also reduces arithmetic inside each contraction;
retiring the per-center preparation removes its repeated CTA barriers.

The matched diagnostic warm public wall owner for stationary force/publication
falls 43.782045 -> 33.158927 s, with SCF/publication 12.946022 -> 12.917465 s,
snapshot 4.233840 -> 4.229824 s and validation 0.588381 -> 0.589387 s. Full
exclusive wall ledgers reconcile separately from device activity. Cold J and
XC/AO/grid device totals remain 199.918086/109.108998 ->
199.662481/109.110204 s at 24 builds; preparation 8.823115 -> 8.797689 s.
This is a force contraction improvement, not an SCF-J/XC optimization or a
sum of asynchronous kernel timers presented as wall time.

### Delivery and remaining boundaries

The default generated consumers select this path wherever their existing
admission conditions hold; no feature flag is needed. Master is unchanged.
PR #2558 is stacked on draft #2556, itself on #2549/#2541, and dependency
reconciliation/review is still required before any merge. No merge is authorized.
Historical GPU4PySCF warm 26.59 s is not rerun: 51.800627 s is still 1.95x
that time, about 25.21 s remaining.

`benchmarks/results/direct-force-reuse-20261016/` retains all 100 lossless
energy/force arrays, original independent references, timings/gates/counts,
source/protocol hashes, numerical convergence trajectories, matched wall/device
profiles, work/resource census, immutable drivers and sanitizer/cache receipts.
Per-iteration component-energy decomposition and unprofiled per-call force-wall
detail remain in hash-bound raw histories, not the compact packet. The packet
is 104878 bytes, superseding only this task's own earlier scoped packet without
increasing the aggregate cap or deleting foreign evidence. Previous publication
remains at commit `11ea887b1606c67062b0542a2ac84434abf07c37`, with its older
#2541/#2549 packet links. No Release, external backup, release tag or merge.

Raw current artifacts live under `.artifacts/direct-cooperative-next/`;
the clean final source export is
`/data/jzzeng/gqc-direct-fused-20261016/candidate/source`.

## Next major cold owner

Repeated SCF J/XC, not inventory planning, dominate cold latency. A host
inventory estimate for the 96-atom canonical basis gives approximately 188 MiB
of MD-J metadata/transforms/scratch, beyond the existing 128-MiB cap; public
active-ledger plans additionally reserve no MD-J optional allowance. This is
not evidence that changing a cap alone is safe or faster. A future exact
density-first Direct J path needs an explicitly charged bounded reservation,
unchanged per-output screening, failure-injection/fallback tests and full
SCF endpoint qualification. Do not bypass the ledger or change screening to
obtain a cold improvement.

## References

- PRs #2541, #2549, #2556, #2558 and issue #1117.
- `2026-10-16-direct-force-recurrence-contraction-reuse.md` and
  `2026-10-16-direct-order-six-response-factoring.md` under implemented/performance.
- `docs/maintainer/performance_engineering.md`.
