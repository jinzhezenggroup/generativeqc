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

Final source/binary identities, complete endpoints, actual per-Fock timing,
independent numerical gates and test results are recorded in the follow-up
qualification entry. Intermediate matrix comparisons pass atol=1e-9, rtol=1e-10
against the independently mapped upstream result. The target-budget candidate
changes J by at most 1.92e-13 versus baseline; upstream disagreement remains at
most 1.62e-10. No complete endpoint speedup is inferred from these raw-J samples.

New tests place unequal signed contractions and nonsymmetric/UKS densities near
the absolute omission cap, replay dense/sparse/zero/dense sources through one
captured owner, and compare against unscreened libcint. Executed host boundary
checks cover subnormal division, threshold neighbors, zero/infinite/invalid bounds
and independent wider-precision omitted-product inequalities.

## Revisit

The remaining gap needs further work-domain or scheduling evidence. In particular,
upstream heuristic cutoffs are not equivalent to our absolute cumulative cap;
this difference neither justifies declaring performance parity nor proves that
all remaining cost is necessary. Revisit discarded schedules only with a changed
workload or demonstrated complete endpoint gains.
