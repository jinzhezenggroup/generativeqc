# Decision: retain immutable FP32 scale proofs within nonlocal packets

Status: implemented
Date: 2026-10-16

## Problem

The shared nonlocal packet schedule repeatedly checks each point's omega/kappa
domain for every physical pair. These inputs do not change while partners rotate
through a packet. FP32 arithmetic can therefore spend its savings on redundant
FP64 comparisons, rather than delivering a faster complete endpoint.

This change follows the independently qualified
[density-screened mixed Fock work](2026-10-16-density-screened-mixed-fock.md).
That work is the incumbent, not a new speedup attributable to this change.

## Decision

Split the existing sufficient domain into point-scale and pair-radius predicates.
The mixed packet specialization computes each point's scale proof when loading
it, then carries the partner proof in bit one of the existing live-state shuffle.
Bit zero continues to mean live. Both row and partner proofs must hold before
the pair-specific radius predicate permits the unchanged FP32 bilateral closure.
Pairs outside the sufficient domain retain the unchanged FP64 closure.

The checked bilateral entry point and the retained-proof consumer share one
generated mathematical body. The unchecked entry point documents its caller's
proof obligation. This is an operator/precision specialization, not a functional
name condition or a second implementation of the nonlocal science.

Strict packets retain their original specialization. There are no extra global
allocations, packet rotations, scientific screening predicates or pair sweeps.
The four-row-per-lane, 128-point tile, FP64 coordinates, weighted sums, scatter
and final force arithmetic remain unchanged.

## Rejected alternatives

- Giving strict packets the retained-proof state slowed the diagnostic strict
  kernel. Keep the independent strict specialization instead of paying admission
  overhead in a path that cannot use FP32.
- Two rows per lane gave only marginal feature-kernel gains in exploration and
  changes packet geometry. Retain the existing four-row schedule rather than
  adding another default-selection policy without endpoint evidence.
- A relative tolerance on a nearly cancelled signed force sum is not implied by
  a relative bound on each positive-denominator pair. The added mixed independent
  long-double oracle propagates the existing `3e-6` bilateral pair error gate through
  absolute contribution sums, plus `1e-11` reduction allowance. The strict oracle
  keeps `rtol=2e-13, atol=1e-11`; complete molecular gates are not relaxed.

## Invariants

- The FP32 domain remains omega/kappa in `[2^-8, 2^8]`, with `r2 == 0` or
  `r2 >= 2^-16`, and `r2 <= 2^12`. Admission is decided before narrowing.
- A proof follows its own point through all rotations, including inactive
  partners, tail lanes and explicitly masked negative-zero rows.
- Complete endpoints retain genuine FP32 J/K work, strict refinement and the
  existing independent energy/force gates. Native FP64 is not the accuracy oracle.
- Timing must compare the frozen incumbent and candidate on the same physical
  GPU, with fresh calculators and identical scientific settings. Diagnostic
  kernel gains must not be reported as complete endpoint speedups.

## Artifact and environment qualification

The stationary AOT contract includes the broad compiler closure. Modifying the
nonlocal compiler therefore invalidated retained force manifests even though
their generated force mathematics did not change. All six existing RKS s/p/d
packages were genuinely rebuilt with the verified ccache, and their official
CMake manifest targets rerun. Their binary hashes and emitted source identities
remain unchanged; the new contract identities are recorded rather than patched
by hand. The incumbent's packages and receipts remain separate.

Initial remote checks exposed an incomplete GCC 12 installation. Selecting the
complete GCC 11 toolchain through PATH preserves strict compilation's rejection
of `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`; the strict guard is not weakened.
Those failed environment runs remain in the evidence archive and are not counted
as passing qualification. All real-device checks use finite Slurm allocations
on nodes 2 or 4 and preserve the scheduler's device visibility.

## Evidence identities

Candidate source identity:
`9e709e551c8261fad08c1c0011fbb3cec4e1ae259c9ca466c78c07d5a76b3b28`.
Core SHA256:
`b6a06237afe3f03b0eb416b0dc9f64c13bece29ad86297b4a2730610601975f7`.
The independent native probe remains
`122906ff32bc22786ec7229221798bc55c16aa23ca7d330de47745712352faee`.

Raw build logs, independent oracle checks, failed and successful environment
receipts, sanitizer reports and same-card comparisons are retained under
`../evidence-fast-packet-20261016/`. The qualified package freeze is
`candidate-hoist-qualified-freeze.json`; earlier exploratory freezes are not
the qualification authority. The timing comparator uses the unchanged driver
from `../evidence-fast-followup-20261016/native96_compare.py`.

## Independent numerical and safety checks

- The host nonlocal/precision admission selection passes 304 tests, with 1086
  explicit device/optional skips. Thirty added bilateral boundary cases exercise
  the exact bounds and their adjacent FP64 values before narrowing to float.
- Slurm 3212 on node2 passes all 1046 selected real-device tests: five compiled
  schedules, partial/mixed-domain packets, independently differentiated signed
  sums, eight ingredient-family complete endpoints, four range-source selections
  and four nonlocal strict-anchor cases. No retries or relabeled strict solves
  satisfy the mixed-work endpoint gates.
- In the same finite allocation, packet memcheck and synccheck each pass twelve
  mixed/strict-domain cases with zero sanitizer errors. The complete three-atom
  nonlocal energy/force endpoint also passes memcheck with zero errors. Their
  instrumented timings are not performance measurements.
- Slurm 3217 retains a fresh independent GPU4PySCF moving-grid twelve-atom
  reference, with basis/geometry/grid equality checked against the measured
  workers. Every small repeat, including both artifact warmups, passes the
  unchanged `1e-5 Eh / 1e-4 max force / 3e-5 RMS force` gates.
- The compiler ownership check reports 541 modules and zero dependency errors;
  Ruff, CUDA formatting, whitespace checks and a ccache-enabled no-op core
  rebuild pass. These selected checks do not claim the entire repository suite
  is green; the incumbent's stale Fast-label tests remain documented separately
  in the preceding decision.

## Complete endpoint results

The frozen density-v2 incumbent and packet-v4 candidate run on the same node4
RTX 5090, UUID `5430ba2e-194a-b6d1-f12c-6b99f6d34efd`. Both sizes use direct
WB97M-V RKS, spherical def2-SVP and `GridSpec(48,16,32)`. Each measurement uses a
fresh process, calculator and hcore density, complete moving-grid forces and
teardown. Disk/compiler artifacts are warm, not the density or CUDA owner. The
small three-repeat comparison swaps arm order in the middle repeat; the larger
comparison is incumbent/candidate/candidate/incumbent across Slurm 1032/1034.

| Complete E/F endpoint | Incumbent median | Candidate median | Elapsed reduction |
| --- | ---: | ---: | ---: |
| 12 atoms / 96 AO, three samples per arm | 15.042745 s | 14.828744 s | 1.42% |
| 96 atoms / 768 AO, two samples per arm | 767.212377 s | 749.552174 s | 2.30% |

The 96-atom samples are 767.172680/767.252074 seconds for the incumbent and
749.625012/749.479335 seconds for the candidate. These are limited repeated
measurements, not a statistical confidence interval or a universal speedup claim.
The earlier density-v2 improvement is not included again in these percentages.

All small samples retain 18 Focks (16 mixed, two strict). All four large samples
retain 21 Focks (17 mixed, four strict), 34 mixed K applications, eight strict K
applications and exactly 20,390,627,797 primary shared J/full-range-K mixed AO-ERI
evaluations. There are no retries. The clean timing driver does not serialize
the nonlocal pair census, so these receipts do not establish a measured equality
of complete 96-atom pair counts. The separate 65,536-point diagnostic executes
the same 2,151,677,952 physical pairs for both packet specializations; the
3/12-atom strict-anchor tests independently retain the expected sweep/cache work.

The Release `-O3 --fmad=false`, `sm_120` diagnostic reduces mixed feature-kernel
time from 104.464638 to 84.434494 ms and mixed geometry-kernel time from
147.716034 to 127.379395 ms. Those 19.2%/13.8% kernel elapsed reductions are not
the 1.4%/2.3% complete-endpoint improvements. No local-memory spills are reported.
Strict production packets do not select the slower retained-proof specialization.

Both large candidate repeats pass the authenticated retained independent
96-atom accuracy oracle bound to the exact basis, geometry, quadrature and
original reference source/driver. No reference timing is reused. The maximum
candidate errors are `6.69e-10 Eh`, `3.86e-6 Eh/bohr` maximum force and
`1.17e-6 Eh/bohr` RMS force, below the unchanged `1e-5 / 1e-4 / 3e-5` gates.
All incumbent repeats are independently gated too. The fresh small reference
also gates every small repeat and both artifact warmups.

The first recorded small artifact warmups are 15.030665 seconds for the incumbent
and 53.871529 seconds for the candidate. Their caches are not equally cold;
retain these values rather than presenting the warm medians as a cold-start
comparison or silently discarding the candidate's first-use preparation cost.

Final machine-readable evidence includes `packet-v4-comparison-summary.json`,
`packet-v4-96-all-arms-accuracy.json`,
`packet-v4-small-all-arms-accuracy.json` and
`packet-v4-qualification-summary.json` in the evidence directory above.

## Revisit when

Further optimization should target measured complete-endpoint costs, including
J/K recurrence/contraction and strict nonlocal sweeps. Removing strict closure,
loosening molecular gates or using kernel timing as the endpoint result is not
an acceptable way to enlarge the reported speedup.
