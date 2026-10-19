# Decision: apportion primitive J omission by its actual target

Status: implemented; final endpoint qualification recorded below
Date: 2026-10-20

## Problem and source correspondence

The source-matched #2609 baseline at `0721ef3f6d2720a7ddd7a5b93a98be49fd959758`
retains a slower J round despite fewer SCF builds. Fresh RTX 5090 job n4:1193
reproduces complete fixed-density r2SCAN-3c J at 1.435229 s and PBE0 J at
0.449158 s. The pinned GPU4PySCF MD arm takes 0.627324/0.348555 s. Both use
96 atoms, identical public-AO MINAO density and independent AO normalization;
these medians include upload and host matrix publication, with five warm samples.
The upstream arm replaces only MD at pin
`9af748a71ca8f4881c604332c2f0cc1632e033ac`, retaining released 1.8.1 helpers.
Its heuristic omission contract differs from our cumulative absolute allowance.

Continued reading of `md_contract_j.cu`, its driver and native `md_j_engine.cuh`
identified repeated per-source screening arithmetic, tile metadata loads,
conservative shared-workspace sizing, low-order barriers and density-panel
layout differences. None was assumed causal merely because upstream differed.
This extends the [scoped source audit](2026-10-19-gpu4pyscf-cold-source-audit.md),
not a whole-repository equivalence claim.

## Decision and proof

Let epsilon be `min(screening_tolerance,1e-12)`, N the complete primitive source
count and m the actual number of primitive products in one output shell pair.
Each of its m targets receives a per-source allowance rounded downward from
epsilon/(m*N). All N sources are still counted, including zero sources.
Cauchy-Schwarz bounds a directed contribution by q_target * source_norm, using
the existing outward Hermite certificates. The target stages a downward-rounded
allowance/q_target. A rejected nonzero source lies strictly below this ratio;
therefore its contribution is below the allowance. Summing at most m*N omitted
products bounds the absolute screening error for every public J entry by epsilon.
This bound is separate from FP64 accumulation error.

The old global maximum m unnecessarily constrained singly and lightly contracted
outputs. It remains valid for coarse nodes and prefix rejection. Coarse rejection
and leaf rejection concern disjoint products, each with a valid per-target share;
the division's conservative boundary slack need not reproduce the old leaf mask.
Zero tolerance retains all nonzero sources, an unknown output norm fails open,
and zero output norm retains unknown/infinite source norms. Existing independent
forward/reverse multiplicities, signed density folding and public projection stay
intact. `reference`/`incumbent`, K, derivatives and capacity fallbacks retain their
contracts. No production oracle, allocation or host synchronization is introduced.

Tile metadata caches only the inner consumer's fields. Nine-double record stride
avoids the shared-bank aliasing of an eight-double record; it fits the existing
96-byte descriptor accounting. The original tile schedule and barriers remain.
The generated recurrence and arithmetic are unchanged.

## Controlled experiments

These ablations precede the final #2609 represented-source and XC finite-range
repairs; their base is `0721ef3f6`, not the repaired head qualified below.
Jobs use finite Slurm allocations on n4, with assigned GPU visibility preserved.
Clean samples exclude counters and profiling; separate calls collect semantic
work. Local data and immutable snapshots are under
`.artifacts/j-optimization-20261020/` and `/data/jzzeng/qc-j-opt-*-20261020`.

| Experiment | Job | r2 J (s) | PBE0 J (s) | Decision |
| --- | ---: | ---: | ---: | --- |
| Baseline | 1193 | 1.435229 | 0.449158 | control |
| Stage metadata only | 1195 | 1.433376 | 0.450127 | no demonstrated gain |
| Stage metadata + exact workspace tiles | 1196 | 1.434689 | 0.450993 | retain old schedule |
| Actual target allowance, staged per tile | 1198 | 1.186422 | 0.370058 | retain target budget |
| Remove independent-source barriers only | 1199 | 1.435544 | 0.450739 | retain barriers |
| Target budget + transposed panels + barrier removal | 1200 | 1.189942 | 0.371712 | discard additional changes |
| Target budget + divided source threshold | 1201 | 1.143926 | 0.356595 | retain threshold |
| Additionally count only nonzero sources | 1202 | 1.135125 | 0.353959 | below 1%; retain complete source census |

The wider-tile target-budget arm reduces r2 roots 2,176,376,338 → 2,003,901,808,
directions 4,108,320,392 → 3,766,588,912 and Hermite summands
119,130,181,760 → 108,322,370,192. PBE0 roots fall 877,443,546 → 811,922,712.
Residual work remains zero. Subsequent qualification uses the final original
schedule, not these intermediate tile counts or a surrogate kernel timer.

Job 1194 is invalid: a versioned library sibling let the test bridge resolve the
baseline SONAME despite checking the explicitly selected file's identity. Corrected
job 1195 checks the actual new kernel symbols. The bridge now explicitly loads the
selected backend first and verifies its dependency symbol resolves to that same
loaded object. An injected wrong-backend test protects this failure mode.
Job 1197 could not collect Nsight hardware counters (`ERR_NVGPUCTRPERM`); its
instrumented times are excluded. CUDA event/CUPTI timing and work counts remain
separate evidence, not replacements for unavailable hardware counters.

## Qualification

The final comparison includes #2609's subsequent represented-source underflow
repair and XC finite-range repair on **both** arms. The baseline is
`9ceac708d60dccf55c2275d977f5767ce75ba1e0`, source identity
`c4c842c16cc24a5a4d81b462fbb6cec4858b2191644f12e7cdff52962345328c`.
The measured candidate is `e10337b1161f735fd061c458d5a37f6865d3d6a1`, source
identity `e2c40e30e095bb81f8c4dae4deb08a9d5176ffd5646f77a55da4cfeaec4d568d`.
Neither arm restores the invalid old source-norm cap. Earlier cold results
125.756/118.957 s exclude these repairs and are not final-head measurements.

After #2609 merged, the two commits were rebased onto master `14928e2c3` as
`cc2245307` and `19e4cee0f`. The measured and rebased SCF, DFT and integral
source trees are byte-identical; unrelated merged MP2/array API changes are
present in the new tree. The rebuilt integration binary authenticates source
`502d9ec71531e58317116c9918ab72ef1ae7cb9fe455618864b3af6a2b25f1fb` and has its
own n2:3496 numerical gates. Its PRO 6000 times are not pooled with RTX 5090
timings. Full library/force-library hashes, raw-output hashes, all clean endpoint
rows, fixed-J samples/work, profile summaries and validation receipts are in
the compact [qualification record](../../../../benchmarks/results/md-j-target-budget-20261020/qualification.json.xz).

The r2SCAN-3c endpoint uses 96 atoms, 1088 spherical AO, direct/noDF FP64,
48×16×32 unpruned moving grid (2,359,296 points), E/D convergence 1e-12/1e-10,
screening 1e-12 and eight CPU threads. It includes analytic forces and native
D4/gCP corrections. The independent E/F gates are 1e-8 Eh and 1e-7 Eh/bohr.
This is a matched-grid implementation diagnostic, not a grid-converged
production ranking. Fixed-density PBE0 J uses 768 AO/def2-SVP.

| r2SCAN-3c complete E/F, n4:1207 | Baseline (s) | Candidate (s) | Fock builds, each |
| --- | ---: | ---: | ---: |
| Cold, including preparation | 135.933645 | 128.912433 | 24 |
| Warm median, five samples | 25.038682 | 24.763745 | 1 |
| Moved geometry | 89.566342 | 88.121349 | 14 |
| Moved-warm median, five samples | 25.044566 | 24.768912 | 1 |

Cold is 5.17% shorter; moved is 1.61% shorter. These are single cold/moved
observations with unchanged Fock-build counts, not confidence intervals. Complete
E/F time is never divided by iterations to estimate J or Fock cost. Warm
complete endpoints still include force work and improve only about 1.1%.

Separate cold CUPTI runs each capture 24 actual Fock intervals, from density
bounds to folded XC totals. Median J kernel time falls 1.456442 → 1.170433 s
(19.64%), while median XC is 2.191487 → 2.191691 s. Median elapsed J+XC interval
falls 3.674611 → 3.388732 s (7.78%). Summed J kernels fall 35.118139 →
28.230150 s, accounting for about 6.89 s of the 7.02 s clean cold difference.
Both preparation/cold traces have zero dropped events and callback errors.
These intervals are direct profiles, not endpoint/iteration quotients, and are
kept separate from clean endpoint samples.

Fresh fixed-density J medians use five clean complete samples per arm, with
source-matched native baseline n4:1206 and candidate/upstream n4:1208:

| 96-atom J, complete publication | Native baseline (s) | Native candidate (s) | Pinned GPU4 MD (s) |
| --- | ---: | ---: | ---: |
| r2SCAN-3c basis | 1.438274 | 1.159571 | 0.629193 |
| PBE0 basis | 0.450472 | 0.358460 | 0.347533 |

Native J time falls 19.38% and 20.43%, respectively. Candidate ranges are
1.159328–1.161031 s and 0.358381–0.358478 s; upstream ranges are
0.628234–0.631823 s and 0.347382–0.349490 s. Thus r2 J is still 1.843× upstream
(84.3% slower), and PBE0 J is 3.14% slower. The different omission contracts
remain; these results do not establish J parity or prove the remaining gap is
unavoidable. Ancillary upstream controls from n4:1206 remain in the local raw
archive; the table and retained receipt use the freshly paired n4:1208 arm.

Final r2 roots fall 2,176,376,338 → 2,003,901,808, directed products
4,108,320,392 → 3,766,588,912 and Hermite summands
119,130,181,760 → 108,322,370,192. PBE0 roots fall
877,443,546 → 811,922,712, directions 1,648,377,816 → 1,522,688,792 and
summands 24,808,667,616 → 22,695,676,456. Tile inventories/coarse screening are
unchanged; all residual work is zero. Owner device bytes stay unchanged.
Matrices differ from baseline by at most 3.20e-14/4.27e-14 (r2/PBE0), and from
the independently mapped upstream matrices by at most 1.614e-10/1.346e-10;
all pass atol=1e-9, rtol=1e-10.

All 30 clean E/F records pass: the 24 r2 baseline/candidate records above and
candidate PBE0 cold plus five warm replays. Maximum r2 candidate errors are
1.00045e-11 Eh and 4.36343e-11 Eh/bohr; PBE0 errors are at most 4.54748e-12 Eh
and 2.44187e-11 Eh/bohr. PBE0 cold is 61.408158 s/17 Focks; its five warm
samples have median 10.333104 s/1 Fock. There is no paired PBE0 complete-endpoint
baseline in this repaired-head campaign, so no PBE0 E/F speedup is claimed.

Validation includes 999 host MD/math/contract/runtime/resource/harness checks,
two host XC finite-range checks, 120 MD GPU tests (one unavailable fully pinned
GPU4PySCF package case deselected), 131 XC GPU tests, and four native XC suites.
MD and XC each pass memcheck, initcheck, racecheck and synccheck with zero
errors/hazards/warnings. The pinned-MD/released-helper comparator is separately
matrix-gated; it is not relabeled as the missing fully pinned package test.
The rebased integration build passes another 120 MD GPU tests and four complete
r2/PBE0 cold/warm E/F gates; no GPU test ran outside Slurm.

New tests place unequal signed contractions and nonsymmetric/UKS densities near
the absolute omission cap, replay dense/sparse/zero/dense sources through one
captured owner, and compare against unscreened libcint. Executed host boundary
checks cover subnormal division, threshold neighbors, zero/infinite/invalid bounds
and independent wider-precision omitted-product inequalities.

Compilation reused verified ccache 4.5.1 through explicit CXX/CUDA launchers.
Across the repaired baseline/candidate build sequence the shared cache counters
increased by 30 hits/15 misses; the master integration build added 13 hits/8
misses. These are shared-server counter deltas, not attribution of every cache
operation to this task. Cache statistics and executed launcher provenance are
retained locally and checksum-bound by the qualification record.

Reproduction uses the repository's `benchmarks.readme_dft` native command with
`--method r2scan-3c --atoms 96 --density-fitting none --grid 48 16 32
--grid-pruning none --repeats 5`, the def2-mTZVPP basis JSON, and the same
independent reference identified in the receipt. Exact command ordering,
environment, fixed-D AO mapping and profile parser are retained in
`.artifacts/j-optimization-20261020/` and the immutable snapshot roots named
above, with script hashes in the receipt. Every GPU job uses finite Slurm
allocations and preserves assigned device visibility. Clean fixed-J samples
include D upload and host J publication; counter/profiler calls are separate.

## Revisit

The remaining gap needs further work-domain or scheduling evidence. In particular,
upstream heuristic cutoffs are not equivalent to our absolute cumulative cap;
this difference neither justifies declaring performance parity nor proves that
all remaining cost is necessary. Revisit discarded schedules only with a changed
workload or demonstrated complete endpoint gains.

The [threshold hierarchy and residency follow-up](2026-10-20-md-j-threshold-residency.md)
moves target thresholds into the hierarchy and replaces the coarse/global target
allowance with conservative minimum-threshold admission, retaining the source
prefix's global allowance and the same scientific cap.
