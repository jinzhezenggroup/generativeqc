# Proposal: fuse Direct force relative-pair response contractions

Status: proposed; implementation and device correctness checks exist, endpoint
performance qualification is in progress.
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

## Candidate

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

## Independent checks so far

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

These checks do not yet establish a complete endpoint performance advantage.
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

## Remaining qualification

Complete matched r2SCAN-3c/48/96, PBE0/48 and HF/48 populations must retain
cold, five warm, moved and five moved-warm calls within the original
1800-second population limit. Every completed call must independently pass
1e-8 Eh energy and 1e-7 Eh/Bohr force gates with actual SCF/Fock counts.
Keep clean endpoint timings separate from the intrusive matched cold/warm
profiles and executed-work census. A default performance claim depends on
these results, not the isolated resource/counter improvements.

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

- PRs #2541, #2549, #2556 and issue #1117.
- `2026-10-16-direct-force-recurrence-contraction-reuse.md` and
  `2026-10-16-direct-order-six-response-factoring.md` under implemented/performance.
- `docs/maintainer/performance_engineering.md`.
