# Decision: fold same-shell primitive exchange in the shared MD owner

Status: implemented; source-bound qualification below
Date: 2026-10-20

## Problem and diagnosis

This follows [low-order Boys reuse](2026-10-20-md-j-low-order-series.md), on
master `eac2de2b33544b104bb318e53aa02aecbea5009f` (#2643). Fresh n4:1262
complete fixed-density J medians reproduce 0.766843 s for 96-atom r2SCAN-3c
and 0.243473 s for PBE0. Paired pinned GPU4PySCF MD takes 0.626535/0.347346 s.
The native r2 radial census is 1,645,838,484 versus upstream 986,115,008.

A count-only diagnostic copies the actual target/source norms and thresholds.
An offline two-dimensional dominance census exactly reproduces both native
GPU work counts. The census itself agrees with independent brute-force counts
on 99 small random fixtures. Its timings are never performance evidence.

The following separate hypotheses were evaluated before selecting a change:

| Offline r2 screening/inventory estimate | Radial sources |
| --- | ---: |
| Current four bins per exponent | 1,645,838,484 |
| Sixteen bins per exponent | 1,637,024,650 |
| Sixty-four bins per exponent | 1,634,749,896 |
| Common marginal cutoff within each contracted parent | 1,625,018,076 |
| Same-shell primitive exchange folded, original certificate | 1,584,854,836 |

Finer bins save at most 0.67% in the explored range; parent budget redistribution
saves 1.27% before accounting for its extra setup. Neither was promoted. A
counterfactual local cutoff with the *same native bounds* predicts 1,059,278,696
sources, but it drops the cumulative guarantee and is not a production option.
It does not reproduce upstream's complete screening and representation policy.
These experiments do not partition the full upstream/native difference into
additive causes or establish that all remaining extra work is necessary.

A separate actual GPU arm groups comparable q bounds by their physical atom
pair. It preserves every radial source, direction and summand but weakens coarse
tile rejection. Complete J becomes 0.810669/0.259601 s, slower by 5.72%/6.62%.
The patch and measurements are retained; original descending-q ordering remains.

## Decision and mathematical invariant

For two primitives of the same shell, the polynomial product, exponent sum,
center and contraction coefficient product are invariant under exchanging the
primitive indices. Store the triangular primitive inventory. A diagonal pair
has unit multiplicity; an off-diagonal representative's cached E transform has
multiplicity two. Use the exact shell center for a same-shell product, avoiding
irrelevant exponent-order rounding in its weighted-coordinate formula.

Both density precontraction and public-AO projection consume the weighted E.
The existing exact J/K source consumer uses those same transforms. When both
sides represent exchanged pairs, their two weights correctly recover all four
original products; reciprocal radial self-pair handling gets no extra factor.
Distinct shells retain every ordered product, even for coincident atoms,
identical exponents or separate columns of a general contraction.

Resource admission charges the reduced descriptor, transform and scratch
inventory. Primitive target norms include the folded E weight, source norms
bound the represented precontracted density, and target multiplicity counts
retained representatives. The existing complete-tail proof therefore still
bounds omitted contributions per public J entry by min(screening, 1e-12).
No screening threshold, radial recurrence, precision, public AO mask or
non-MD fallback changes. All three retained MD recipes share the inventory.

## Complete J and work evidence

Final n4:1266 uses five clean warm samples, after first-call and warm-up samples.
Preparation is separate; complete J includes upload, evaluation, AO mapping
and host publication. Tracing and work counters run outside clean samples.
The independent comparator uses pinned native MD source
`9af748a71ca8f4881c604332c2f0cc1632e033ac` with released 1.8.1 frontend/helpers.
This is a matched-density implementation diagnostic, not a grid-converged
production ranking.

| Complete J median | Parent | Final | Paired GPU4 MD | Gain |
| --- | ---: | ---: | ---: | ---: |
| r2SCAN-3c | 0.766843 s | 0.744910 s | 0.627211 s | 2.86% |
| PBE0 | 0.243473 s | 0.234705 s | 0.348028 s | 3.60% |

r2 remains **18.77% slower** than the paired reference. The initial folded arm
independently measured 0.745109/0.234894 s; final formatting and owner comments
do not change its work inventory or numerical results.

| Work | r2 parent | r2 final | PBE0 parent | PBE0 final |
| --- | ---: | ---: | ---: | ---: |
| Radial evaluations | 1,645,838,484 | 1,584,854,836 | 656,368,438 | 628,238,726 |
| Directed contractions | 3,058,022,000 | 2,943,216,144 | 1,217,100,552 | 1,165,303,624 |
| Contraction summands | 89,200,193,168 | 87,370,807,472 | 18,077,298,744 | 17,583,281,528 |
| Owner bytes | 693,472,836 | 693,235,652 | 347,775,940 | 347,628,708 |

The 896/608 eliminated primitive descriptors reduce actual radial work by
3.71%/4.29%. Measured work matches the offline folded prediction exactly.
Residual work is zero. All fifteen classes retain zero local-memory bytes.
Matrices differ from the parent by at most 4.62e-14 and from upstream by
1.614e-10, passing the unchanged atol=1e-9/rtol=1e-10 gate.

## Scientific and endpoint qualification

Final source identity:
`9166b1f802a7b7f61e6d1e191bc39bde0be43ac7856a29d8d970c4c179bd8bac`.
Library SHA-256:
`86fefb98639063107eb45dfae3c8358e23b8446b308550b296e6dc35e73ceb65`.
Snapshots verify library/source identity before execution. Builds use verified
ccache 4.5.1; compiler commands and before/after cache statistics are retained.

The local host suite has 1,504 passes; 43 independent libcint cases require
PySCF, which is unavailable in that local Python environment. A separate
reference-enabled CPU Slurm run passes all 54 reference tests, including those
43 previously skipped cases (1,547 distinct host tests overall).
Its first run exposed the test bridge's old hand-built rectangular inventory:
feeding both exchanged descriptors into already-folded production transforms
double-counted them. The bridge now supplies the documented triangular
inventory; the independent libcint expectations and acceptance gates are
unchanged. Both the initial failure and passing rerun are retained.
Final n5:1807 has 132 real-GPU passes; the unavailable fully pinned-package test
is deselected, while the separate pinned-native-MD comparison is executed.
All four CUDA sanitizers report zero errors. New independent libcint cases
exercise three signed primitives, translated centers, p/d components, separate
contraction columns, both representations, nonsymmetric RKS/UKS densities,
reversed primitive order, J and K, and all three MD recipes. Existing cumulative
omission, graph replay, capacity/fallback and owner-lifetime gates remain.

Final n4:1267 completes all 18 energy/force records, including all five warm
repeats and moved geometry. Every record passes the unchanged E=1e-8 Eh and
F=1e-7 Eh/bohr gates; maxima are 9.095e-12 Eh and 4.211e-11 Eh/bohr.

| Complete E/F endpoint | Seconds | Fock builds |
| --- | ---: | ---: |
| r2 cold | 119.249033 | 24 |
| r2 warm median | 24.356931 | 1 |
| r2 moved | 76.615556 | 13 |
| r2 moved-warm median | 24.391905 | 1 |
| PBE0 cold | 59.521928 | 17 |
| PBE0 warm median | 10.212688 | 1 |

The exact parent binary's retained n4:1261 endpoint has r2
119.883867/24.391457/80.235406/24.386074 s with 24/1/14/1 Focks, and PBE0
59.843555/10.209123 s with 17/1 Focks. These are explicitly historical samples,
not newly rerun parent endpoints. In particular, moved SCF changes from 14 to
13 Focks; do not attribute its full time reduction to J throughput. Warm
endpoint gains are not uniform.

Actual cold-SCF profiling on n4:1268 retains 24 Fock builds and passes the same
E/F gates. Median J kernel sum falls from the parent's retained 0.775632 s to
0.753894 s (2.80%); total J kernel time falls from 18.716536 to 18.195625 s.
The retained n4:1258 GPU4PySCF actual-SCF sample has 60 cycles and 61 J calls,
with median synchronized complete device-returning J call 0.719065 s. That
scope is wider than our kernel sum and uses its recorded 1e-14 direct cutoff;
it is a historical comparison, not a newly rerun matched-work experiment.
Even this narrower native measurement is still about 4.84% slower, so fewer
native SCF iterations must not be presented as per-iteration J parity.

An all-target build encountered the unchanged native test
`test_ks_resident_weights.cpp` calling absent
`PreparedBatch::ks_transport_diagnostic`. The same compiler failure is reproduced
from a clean archive of the parent. Required production, stationary profile and
manifest targets build successfully; no all-target build success is claimed.

## Evidence retention and revisit

The [qualification receipt](../../../../benchmarks/results/md-j-same-shell-fold-20261020/qualification.json.xz)
retains source/binary bindings, all complete J samples, endpoint rows, semantic
work, trace/resource summaries, test outcomes, diagnostic and rejected patches,
reproduction scripts, and the baseline-build failure. Raw diagnostic arrays,
matrices and full traces remain ignored local artifacts with retained hashes.

One historical MINAO sample is compacted by removing only JSON whitespace
outside strings; exact tokens and parsed values are preserved. The old hash
has no tracked references, including decompressed xz records. The receipt keeps
both hashes and sizes. This frees space for the new receipt without increasing
the existing evidence cap or discarding measurements.

Remaining work should target further exact reuse or tighter certified screening.
Do not repeat spatial ordering or more complex budget allocation solely on an
estimated source count, and do not replace the cumulative certificate with a
local cutoff to claim parity.
