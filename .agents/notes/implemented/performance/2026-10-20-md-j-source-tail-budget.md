# Decision: certify the complete omitted J source tail

Status: implemented; qualification below
Date: 2026-10-20

## Problem

After #2635, 96-atom r2SCAN-3c still spends 0.977 s per complete fixed-density
J evaluation versus 0.627 s for the pinned GPU4PySCF MD consumer. The
[source audit](2026-10-19-gpu4pyscf-cold-source-audit.md) identifies a deliberate
remaining difference: native certifies a cumulative absolute omission cap,
whereas upstream applies its own pair cutoff. Counting every omitted primitive
at the threshold wastes the native allowance when most sources are much smaller.
Copying upstream's cutoff would change the scientific contract.

The parent is `f0ee227aba55c106ae1fdbe9f236e907654d03da`. This decision supersedes
the two-pass budget refinement in the
[prefactor/active-budget note](2026-10-20-md-j-prefactor-active-budget.md), while
retaining its compiler-owned scaled Boys seeds and unchanged radial arithmetic.

## Decision and proof

The existing source-norm kernel also counts positive finite bounds in 8,188
integer bins. A bin is selected by the FP64 exponent and leading two mantissa
bits. Its exact lower boundary is the nonnegative bit pattern `bin << 50`;
its upper boundary is the next pattern. This includes subnormals without
logarithms, estimates, or floating-point atomic sums. Zero and unknown/infinite
norms are not counted. Unknown/infinite sources cannot be removed by a finite
histogram threshold.

A short device scan constructs an outward upper bound on the sum of all source
norms strictly below each boundary. Integer count times the exact upper endpoint
and every addition round upward. The owner already restricts the primitive
census to `INT32_MAX`, so count-to-double conversion is exact. Overflow leaves
an infinite conservative tail. The last bin is never summed into a finite
threshold. The arrays add 131,008 bytes to the already-budgeted replay state;
no private cap, uncharged allocation, CPU integral work, or host poll is added.

For a target primitive with outward norm q and parent multiplicity m, set
b = downward(min(screening, 1e-12)/m). Binary search selects the largest finite
boundary t with tail(t) <= downward(b/q). Retain the existing uniform threshold
as another complete certificate and choose the larger. This selects a certified
omitted set; it does not sum two independently spent allowances.

The source-prefix threshold uses the global maximum norm and multiplicity, so
it is no larger than any target threshold. A target leaves its prefix only if
all sources are below its threshold. Therefore prefix and inner-loop omissions
are subsets of the same certified tail. Each target loses at most b, and the m
targets contributing to a public AO pair lose at most epsilon in total. There
is no need to split the budget or rebuild the hierarchy a second time.

Density changes refresh the count arrays, tail, hierarchy and prefixes in stream
order, including graph replay. Complete transforms and public AO reconstruction
are retained. Zero cutoff, zero density, empty/subset windows, infinite bounds,
signed/general contractions, RKS/UKS and bounded owner fallback keep their
existing contracts. Histogram capacity is included through the existing state
size admission; refusing optional storage retains the incumbent provider.
The exact-mask `reference` recipe does not use the new screening policy.

The Coulomb prefactor expression is also scoped to the single radial-producing
component lane, matching upstream `gout_id == 0`. Its arithmetic is unchanged.
No separate speedup is attributed to that placement.

## Measured alternatives

Same 96-atom fixed-D protocol, five clean warm samples per arm:

| Arm | r2SCAN-3c J, s | PBE0 J, s | Disposition |
| --- | ---: | ---: | --- |
| Fresh parent | 0.976853 | 0.310970 | baseline |
| Histogram in old second pass | 0.874233 | 0.274919 | superseded |
| Full-tail certificate, one bin per exponent | 0.851091 | 0.266706 | useful predecessor |
| Shared positive-series Boys plus full tail | 0.903530 | 0.302075 | rejected: slower |
| Weighted parent allowance plus full tail | 0.856768 | 0.269275 | rejected: more work |
| Weighted allowance plus exact global target minimum | 0.856955 | 0.269329 | rejected: no work reduction |
| Four bins per exponent, final repeat | 0.835771 | 0.261872 | retained |
| Paired pinned upstream MD | 0.627763 | 0.347132 | independent comparator |

The final improvement is 14.44% / 15.79% versus the fresh parent. Native r2 J
remains 33.13% slower than upstream; PBE0 is 24.56% faster. This is progress,
not r2 per-Fock parity. Preparation is retained separately in fixed-D receipts;
complete J includes density upload, evaluation, AO mapping and host publication.

The positive-series Boys candidate improved the independent radial error but
lost complete J time. A separate reciprocal-Boys experiment saved about 4%
before tail screening. It passed the unchanged 80-digit bound-scaled oracle,
although cancellation-sensitive comparisons against rounded auxiliary values
differed. It is not promoted or endpoint-qualified here. Neither experiment
changes the final common Boys owner or compiler emission.

An isolated prefactor-placement run failed during preparation: restoring files
with old timestamps left a stale host object with a different view layout.
That run has no valid timing. Rebuilding every header consumer corrected the
artifact; the final source and binary identities match. Future experiment
restores must update modification times or explicitly rebuild affected objects.
Source identity alone cannot certify an incorrectly reused object file.

## Work and numerical evidence

| Work | r2 parent | r2 final | PBE0 parent | PBE0 final |
| --- | ---: | ---: | ---: | ---: |
| Radial roots | 1,940,227,124 | 1,645,838,484 | 787,382,540 | 656,368,438 |
| Directed contractions | 3,644,181,776 | 3,058,022,000 | 1,475,383,320 | 1,217,100,552 |
| Summands | 104,946,669,664 | 89,200,193,168 | 21,980,413,304 | 18,077,298,744 |
| Source tiles | 7,244,034 | 5,528,314 | 2,304,239 | 1,798,645 |

All residual work is zero. Final matrices differ from parent by at most
6.53e-14 and from pinned upstream by at most 1.62e-10, passing the declared
atol=1e-9 / rtol=1e-10 gates. No scientific recurrence or precision changed.

Host qualification has 1,161 passes, including independent radial arithmetic,
compiled CUDA-body host emulation with ASan/UBSan, replay/window/prefix omission
sums, subnormal precontraction, full exponent/mantissa boundary tails, overflow,
and an executable useful-screening case. The final GPU suite has 120 passes and one unavailable fully pinned package
case deselected; the separate pinned MD comparator ran and passed. The real
CUDA tail probe passes 150 independent 80-digit checks. Memcheck, initcheck,
racecheck and synccheck each report zero errors, with zero race hazards or
warnings. Sanitizers cover the two retained small scientific cases, not the
full 1088-AO endpoint.

| Complete E/F endpoint, s | Parent | Final | Fock builds (both) |
| --- | ---: | ---: | ---: |
| r2 cold | 124.926143 | 121.546332 | 24 |
| r2 warm median (5) | 24.602244 | 24.490373 | 1 |
| r2 changed geometry | 79.791699 | 77.839278 | 13 |
| r2 changed-warm median (5) | 24.600042 | 24.506714 | 1 |
| PBE0 cold | 63.353657 | 62.384679 | 18 |
| PBE0 warm median (5) | 10.263626 | 10.216764 | 1 |

All 18 clean final endpoint samples pass. Maximum errors are 1.01e-11 Eh and
4.28e-11 Eh/bohr. The complete endpoints include unchanged solver/force work,
so their improvement is smaller than the J improvement. These observations
are not normalized by iteration counts; the separate actual-Fock profile
measures J within the real SCF trajectory. With 24 Focks in both profiles,
the median J kernel time falls 0.988019 -> 0.847151 s (14.26%), summed J
23.834203 -> 20.441372 s, and median Fock interval 3.206185 -> 3.065230 s.
XC stays 2.191571 -> 2.191559 s. All 15 native angular classes retain zero
local bytes per thread; profile traces report no dropped records/callback
errors.

A fresh released-GPU4PySCF 1.8.1 actual-SCF profile uses its endpoint's
`direct_scf_tol=1e-14`, unlike the matched fixed-D comparator's `1e-12`.
It performs 54 counted cycles and 55 J calls, including the initial potential,
versus native's 24 Focks. Median complete device-returning `get_j` wall time is
0.719819 s, with 39.157563 s summed across calls; the E/F gate passes.
Those calls are individually synchronized diagnostics and include Python/host
orchestration. Native's 0.847151 s is a narrower J-kernel sum, already 17.69%
longer, so the remaining per-Fock gap is real. Do not turn these different
scopes/trajectories into an exact same-work speed ratio or mix this profile's
55 calls with a different clean run's convergence count. The 33.13% comparison
above remains the distinct, same-D, nominal-1e-12 consumer comparison.

The receipt is
[`qualification.json.xz`](../../../../benchmarks/results/md-j-source-tail-budget-20261020/qualification.json.xz).
To retain this receipt under the existing evidence cap, only whitespace outside
JSON strings was removed from `minao-default-20261007/samples/core1-96.json`
and `core2-96.json`. Numeric tokens, key order and string bytes are unchanged;
parsed equality, old/new hashes and a search for old hash references are
recorded. No measurements were removed or cap increased.

Raw source snapshots and scripts remain under `.artifacts/j-boys-20261020/`;
final Slurm snapshots are `/data/jzzeng/qc-j-boys-final-20261020` on n4 and n5.
The final scientific source identity is
`108e88a5a6afff1fbed0074c4343b303a92064df8ff7176b73b98229d42c07eb`.

The protocol uses 32 waters/96 atoms, r2SCAN-3c/def2-mTZVPP (1088 spherical AO)
and PBE0/def2-SVP (768 AO), direct FP64, an explicit moving unpruned version-1
48x16x32 grid (2,359,296 points), E/D convergence 1e-12/1e-10 and screening
1e-12. Energy/force gates remain 1e-8 Eh / 1e-7 Eh/bohr. GPU4PySCF is MD pin
`9af748a71ca8f4881c604332c2f0cc1632e033ac` with released 1.8.1 helpers, not a
fully pinned package. This matched-grid diagnostic is not a grid-converged
production ranking. Profiles/counters are separate from clean measurements.

## Consequences and revisit conditions

A small density-dependent histogram replaces conservative worst-case counting,
reducing work without weakening the cumulative cap. Its launch/scan overhead is
included in the measured endpoint. The bins and extra resident state can be
revisited if larger-size or architecture measurements change their economics.

The remaining r2 gap is explicit. Future work should distinguish extra work
required by the cumulative certificate from arithmetic/schedule inefficiency,
and qualify complete endpoints before claiming parity. Do not revive the
weighted shares, exact-prefix reduction, or positive Boys series merely because
they appear tighter or resemble upstream more closely.

The subsequent [Boys arithmetic qualification](2026-10-20-md-j-boys-reuse.md)
promotes reciprocal reuse and a narrower positive-series region after separate
ablations and complete endpoint validation. It preserves this tail certificate;
the rejected whole-owner positive-series experiment above remains negative evidence.
