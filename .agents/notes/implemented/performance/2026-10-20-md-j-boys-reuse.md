# Decision: reuse FP64 Boys arithmetic in compact MD J

Status: implemented
Date: 2026-10-20

## Problem

After #2639, a fresh 96-atom r2SCAN-3c/def2-mTZVPP complete fixed-density J
still took 0.836018 s versus 0.626016 s for the pinned GPU4PySCF MD target.
The actual native cold SCF also spent longer per J application; dividing total
SCF time by iteration counts would hide the different solver work and timing
boundaries. This campaign continues from immutable parent
`61033aaca9d6c14de6cab68ffbeb2a6185eb69f2`.

The upstream audit followed `gvhf-md/md_contract_j.cu` and `unrolled_md_j.cu`
through `gvhf-md/boys.cu` into `gvhf-rys/gamma_inc.cu` at
`9af748a71ca8f4881c604332c2f0cc1632e033ac`. Copying a contraction schedule did
not copy every operation in its scientific dependencies.

## Decision

The shared Boys owner exposes an explicit FP64 reciprocal-reuse specialization,
with the original arithmetic as its default. Compact MD emission opts in:

- `sqrt(T)` supplies both the erf argument and F0 denominator, using the nearest
  FP64 `sqrt(pi)/2` constant.
- One `0.5/T` supplies every upward recurrence step, replacing repeated division.
- For orders 5–8 and T < 6, the emitter uses the existing positive-series
  `range_detail::long_moments` owner. It does not duplicate the series algebra.
  A bounded failure falls back to the shared Boys owner.

The last condition is deliberately narrower than invoking the positive owner
for every argument. The earlier whole-owner experiment extended the series
region to order+1.5 and lost time. Here the positive series replaces only the
already-series high-order region. Low orders and the upward-domain boundary
retain their existing policy. Native storage, screening, tiling and resource
admission are unchanged. The selected behavior is the default compact J path.

## Audited boundaries relative to upstream

| Boundary | Native behavior and disposition |
| --- | --- |
| Upward Boys arithmetic | Now shares the one-root/reciprocal strategy; FP64 rounding is independently qualified. |
| Small-T Boys | Positive shared series for high orders below six; lower-order alternating series retained. Upstream uses positive series below m/2+0.5 and its tiny-T limit, so thresholds/termination are still different. |
| Coulomb exponent/prefactor | Original formula retained. Tile-local exponent reciprocal reuse is an additional optimization beyond upstream, not an upstream omission; rejected below. |
| Radial recurrence | Compiler-owned descending scalar in-place lowering, versus upstream cooperative/unrolled storage. No new contraction or radial algebra is introduced here. |
| Density and symmetry | Native folded Hermite density and caller-owned directional/self-source weights remain; upstream decontracted-shell half factors cannot simply be transplanted. |
| Screening | Native cumulative error cap, primitive norms and complete source-tail certificate remain. Upstream float-log thresholds are a different omission contract. |
| Scheduling | Native persistent 128-thread source/component tiles, exact tail handling and optional-owner budget remain. Upstream block geometry, larger rectangular tiles and padding work differ; earlier experiments are preserved in the linked notes. |
| Frontend/SCF | Same-D comparison isolates J; actual SCF retains each solver and counts actual Focks. GPU4PySCF includes an initial potential outside its counted cycles. |

This audit does not claim identical instruction streams or prove the remaining
performance gap unavoidable. The arithmetic gains leave a measurable gap.

## Numerical invariants and test interpretation

All storage and accumulation remain FP64. The independent radial oracle uses
80-digit Boys integrals and closed Cartesian derivatives with factorial
multiplicities, sharing neither generated state maps nor native recurrences.
Its original acceptance gate, `abs(error) <= 3e-12 * sum(abs(exact terms)) +
1e-290`, is unchanged. Both compact and retained full auxiliary now have to
pass it. Direct seed tests cover both sides of every existing branch boundary
and require relative error <= 3e-12.

The former additional 2e-14 relative comparison against another rounded full
auxiliary failed at near-zero Cartesian derivatives after reciprocal
reassociation. Those failures were not failures against the independent oracle.
Keeping that comparison as the accuracy authority would freeze an evaluation
order. The replacement qualifies both implementations independently and retains
contraction, Cartesian parity, buffer sentinel and runtime synchronization tests.
The old Boys comment promising <4e-15 seed error was inaccurate near the retained
alternating-series boundary; both baseline and reciprocal CUDA seeds reach
1.568161e-12 on the new boundary-neighbor grid. No threshold was relaxed to admit
this behavior.

## Rejected alternative

Staging exponent reciprocals in the unused ninth shared descriptor word kept
storage/work counts unchanged and passed 1,195 host checks, including an 80-digit
parameter oracle and extreme-exponent fallback cases. Complete J improved only
0.73% (r2SCAN-3c) / 0.88% (PBE0). That gain did not justify the extra admitted
exponent range, fallback branch and reassociated prefactor; the patch and raw
results are retained as negative evidence. Source restoration used fresh mtimes.

## Evidence

Final source/binary identities, raw clean samples, separate CUPTI/work records,
endpoint/Fock data, independent CUDA probe, host/GPU tests, sanitizers, compiler
cache records, exact patches and reproduction scripts are in
`benchmarks/results/md-j-radial-arithmetic-20261020/qualification.json.xz`.

Final scientific source identity:
`d8ec594f77f26b8733592e323a290fde85c876aafac490982fbde134dc5c7e58`.
Native library SHA-256:
`bf8ff8979bea0f50e2075a1a15b0ef04ea0966aa62b409f7ecee7e89771942b6`.

All measurements used finite Slurm allocations on RTX 5090: n4 jobs 1239–1246
for performance/endpoints, n5 jobs 1802–1803 for independent CUDA qualification.
Source identity assertions passed on both nodes. Five clean warm fixed-D
samples, with preparation separate, give:

| Complete J | Parent | Final | Paired pinned GPU4 MD | Native gain |
| --- | ---: | ---: | ---: | ---: |
| r2SCAN-3c | 0.836018 s | 0.792027 s | 0.627390 s | 5.26% |
| PBE0 | 0.261907 s | 0.250584 s | 0.347185 s | 4.32% |

The r2SCAN-3c fixed-D gap is still 26.24%; PBE0 is 27.82% faster than this
comparator. Reciprocal-only r2SCAN-3c took 0.806395 s; adding the restricted
positive series took 0.794320 s before final formatting/rebuild qualification.
Separate traces show 3–4% gains in most low-order classes and 8–15% in the
high-order classes using positive series, with zero local bytes in all 15 classes.

Work and planned owner bytes are exactly unchanged in every arm:

| Case | Radial sources | Directions | Summands | Tiles | Screened tiles | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r2SCAN-3c | 1,645,838,484 | 3,058,022,000 | 89,200,193,168 | 5,528,314 | 2,329,750 | 693,472,836 |
| PBE0 | 656,368,438 | 1,217,100,552 | 18,077,298,744 | 1,798,645 | 818,053 | 347,775,940 |

Residual work is zero. Final matrix errors versus the parent are <=5.69e-14,
and versus pinned MD <=1.62e-10 (gates atol1e-9/rtol1e-10).
Complete energy-and-analytic-force endpoints give:

| Endpoint | Parent | Final | Focks, both |
| --- | ---: | ---: | ---: |
| r2 cold | 121.688224 s | 120.359919 s | 24 |
| r2 warm, median5 | 24.580237 s | 24.407304 s | 1 |
| r2 moved | 80.229248 s | 77.233266 s | 13 |
| r2 moved-warm, median5 | 24.454673 s | 24.408625 s | 1 |
| PBE0 cold | 60.070693 s | 59.635608 s | 17 |
| PBE0 warm, median5 | 10.232403 s | 10.239918 s | 1 |

All 18 final endpoint records pass E<=1e-8 Eh and force<=1e-7 Eh/bohr;
observed maxima are 1.01e-11 Eh / 4.15e-11 Eh/bohr. Cold and moved rows are
single complete samples; warm rows retain five repeats and their variability.
PBE0 warm is essentially flat, with a slightly higher final median; no complete
warm-endpoint speedup is claimed there. This fresh parent PBE0 run uses 17 Focks,
whereas the preceding campaign's same-source run used 18. Actual counts remain
visible rather than being normalized away.

Actual r2 cold SCF has 24 Focks in both native profiles. Median J kernel sum
falls **0.847152 -> 0.802065 s (5.32%)**, with total J
**20.441561 -> 19.353563 s**. Median XC remains
2.191501 -> 2.191520 s; the median Fock interval falls
3.065176 -> 3.020205 s. Fresh released GPU4PySCF 1.8.1 uses 54 counted cycles
and 55 complete synchronized J calls, median **0.718421 s**, sum 39.062243 s.
Its E/F gate passes. The narrower native kernel sum still exceeds the reference's
complete call: actual SCF has not reached J parity either.

Final qualification: 1,170 host math/runtime/contract checks; 28 additional
mixed-default host checks; 120 GPU tests (one unavailable fully pinned-package
case deselected, with the separate pinned-MD comparator retained); memcheck,
initcheck, racecheck and synccheck with zero errors/hazards/warnings. The
independent CUDA probe checks 33,660 Cartesian radial values plus 6,120 Boys
values. Final maximum radial absolute-term-scaled error is **1.88e-15**, versus
3.74e-13 in the reciprocal-only probe. The declared 3e-12 gate is unchanged.
Compiler structure, CUDA ownership and repository checks pass. Cache version,
actual compiler launcher command and before/after shared-cache statistics are
retained; they are not represented as exclusive campaign cache counters.


The fixed-D and actual-SCF comparisons have distinct scopes. Fixed-D native and
pinned MD include upload, evaluation, AO mapping and host publication, with
owner preparation retained separately. Actual GPU4PySCF uses released 1.8.1 MD,
full-density rebuilds and 1e-14 tolerance; its synchronized complete get_j calls
include Python/orchestration. Native per-Fock J numbers are narrower kernel sums.
Instrumented profiles are not mixed into clean endpoint times. No result is an
independently grid-converged production ranking.

To retain evidence under the existing 64-MiB cap, only JSON whitespace outside
strings was compacted in two previously unbound MINAO samples. Numeric tokens,
string bytes, key order and parsed values are unchanged. Exact old/new hashes
and a search of all tracked bytes plus decompressed xz are in the receipt. No
prior measurement was deleted or retimed.

The [low-order follow-up](2026-10-20-md-j-low-order-series.md) extends the
positive series to the existing order-1–4 domains after separate qualification.
It also records actual upstream/native source censuses and negative wider-tile
and upstream-threshold experiments; the historical measurements here remain
unchanged.

## Revisit when

Profile the remaining semantic work, Boys domain distribution and lane
utilization before adding a broader series policy or more reciprocal caches.
Do not remove cumulative omission bounds to imitate upstream timings. Recheck
higher angular domains, mixed precision and derivative consumers independently
before expanding this opt-in shared arithmetic beyond compact FP64 J.

## References

- [Source-tail budget](2026-10-20-md-j-source-tail-budget.md)
- [Prefactor and active budget](2026-10-20-md-j-prefactor-active-budget.md)
- [Threshold/residency experiments](2026-10-20-md-j-threshold-residency.md)
- [Quartet scheduling](2026-10-19-md-j-quartet-lane-scheduling.md)
- [Current contract](../../../../docs/developer/md_j_engine.md)
