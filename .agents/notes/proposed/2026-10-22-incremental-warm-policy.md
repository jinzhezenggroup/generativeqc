# Proposal: incremental cold J/K with ordinary resident warm proof

Status: proposed; isolated research prototype, no default promotion
Date: 2026-10-22
Base: `afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e` (fetched origin/master)

Superseded for the implemented local candidate by
[bounded incremental proposals](../implemented/performance/2026-10-23-bounded-incremental-cold-proposals.md).
This note retains the V3 experiment; the successor adds bounded proposal epochs
with capability-based admission and no incremental functional whitelist.

## Problem

The goal is faster complete cold E/F without slowing warm execution. Generic
strict-FP64 incremental Direct J/K changes more than its density input. CUDA KS
resets incremental energy history on each solve and requires two full-density
builds after entering refinement. A converged warm seed consequently performs
three full Focks and zero delta applications.

Cold incremental mode also excludes the qualified Standard stopping policy.
At default E/D `1e-9`/`1e-5`, ordinary admitted KS uses a `1e-5` physical ceiling
with a complete canonical proof; incremental used `min(1e-9, density_tolerance)`
and its full-density history gate. On PBE0/96 this raises actual cold Focks
from 9 to 17, overwhelming screening savings. A warm-only fix is insufficient.

HF independently disables converged-Fock reuse with incremental mode. Its warm
endpoint reports one iteration but performs three Focks, including two post-SCF
builds. HF is diagnosed, not modified: captured graph contents must agree with
the effective per-solve policy before extending this design.

## Prototype and invariants

Prepared capability/storage remains distinct from the effective per-solve mode.
Each `CudaKsPlan::begin` recomputes that mode. A retained same-owner warm density
uses ordinary full F[D] and complete physical/canonical validation with fresh
analytic forces. Cold, explicit/external seeds, geometry changes and cleared
warm state restore incremental capability. Warm diagnostics report
requested=true, active=false, without inventing delta work.

The warm guard is limited to all-electron RKS without range exchange or nonlocal
correlation. UKS/ECP/range/nonlocal execution retains its policy.

The V3 cold prototype additionally:

1. Admits incremental solves to existing capability-based Standard convergence.
   Explicit tighter E/D controls retain strict stationarity and closure.
2. Starts the safer full-density tail when density and physical RMS pass, without
   waiting for AO maximum. This selects arithmetic, not convergence. AO maximum
   and complete canonical products remain acceptance gates.
3. Counts an already-executed full F[D] as the first closing energy. A delta
   energy cannot count; two consecutive full energies remain mandatory.
4. Uses ordinary resident final-validation products only after the Standard
   canonical proof and consecutive full-density closure succeed.

The screened lower still allows at most one delta before a full refresh. No
screening tolerance, nonlinear XC input, force operator or publication gate is
weakened. Density-RMS thresholds `0`, `1e-3` and `1e-4` were measured; no default
threshold change or size whitelist was introduced.

## Alternatives and evidence

V1 only guarded warm. At 12/48/96 atoms, PBE0 warm medians returned from
0.665/4.461/12.960 seconds (unguarded master on) to 0.394/2.692/8.619 seconds,
matching ordinary master 0.394/2.684/8.629. Actual warm Focks changed from 3 to 1.
But cold remained 4.263/18.384/42.690 seconds against master
3.235/13.103/30.758. This did not satisfy the complete goal.

V2 admitted Standard proof and counted an executed full build. PBE0/96 reduced
to 10 Focks but its pilot still took 32.10 seconds. V3 starts the full tail
earlier and restores resident final validation. The three-round rotated-order
comparison, fresh processes and two frozen warm calls per cold, reports:

| PBE0 atoms | Master cold | V3 cadence cold | Master warm | V3 cadence warm |
| --- | ---: | ---: | ---: | ---: |
| 12 | 3.0947 | 3.1377 | 0.3884 | 0.3894 |
| 48 | 12.9396 | 12.6430 | 2.6560 | 2.6451 |
| 96 | 30.1698 | 29.5942 | 8.4032 | 8.5284 |

All values are medians in seconds. Cold Focks are 10/9/9 in both arms; V3
replaces three with delta builds. Warm always performs one full Fock. Cold
improvements at 48/96 are 2.29%/1.91%, with every paired round faster. Twelve
atoms show no reliable benefit. Ninety-six-atom warm has visible timing variance
(V3 8.159–9.183 seconds versus master 8.182–8.501); do not claim universal zero
latency impact from the equal work count. Adjacent warm and intrusive component
controls are recorded separately in the evidence report.

The follow-up same-binary adjacent warm control reverses owner preparation
order in two processes, with six frozen calls per arm. Off/on medians are
8.43761/8.42605 seconds; adjacent changes range from -0.81% to +1.08%. This
supports no fixed warm policy overhead, without erasing the fresh-process
variance above. Independent gates pass for all 22 added diagnostic/adjacent
E/F endpoints. Intrusive late-delta diagnostics reduce cold J/K event time
10.0528 to 9.4021 seconds (K included: 7.8538 to 7.3338), while XC stays
4.289 seconds. MD-J unscreened source tiles fall 8,837,481 to 8,285,045;
warm J work is identical. These counters are tiles, not ERI quartets. The
incremental owner retains approximately 18 MiB additional state at 96 atoms.

Delaying delta until density RMS `1e-3` did not consistently beat cadence-only:
its cold medians are 3.0878/12.8647/29.8485 seconds. The initially encouraging
single 96-atom tuning sample is not promoted over the repeated evidence.

The new executable controller test distinguishes full/delta energy history and
preserves AO-maximum acceptance; 80 targeted host tests pass. V3 qualification
includes default and strict PBE0, PBE, moved geometry, frozen moved warm and
clear-warm. Independent tight full-density GPU4PySCF references use identical
models/default grids and grid-response forces. Gates remain `2.5e-6 Eh/atom`,
maximum force `2.5e-5 Eh/Bohr`, RMS force `7.5e-6 Eh/Bohr`. This is same-grid
solver qualification, not quadrature convergence. The cold research summary
qualifies 133 complete endpoints across V2/V3/tuning/repeats and lifecycle controls.

## Boundaries and revisit conditions

No generic default promotion is made. Gains are small and size dependent;
production qualification needs broader independent molecules, functionals,
basis sizes, devices and resource-lifetime coverage, with reproducible cold
wins and warm nonregression. The prototype remains reviewable locally.

Public KS resource-budget/plan admission remains unsupported for incremental
mode. Extra anchor storage remains allocated for the owner's lifetime; there is
no claim of capacity neutrality under memory pressure. DF and mixed arithmetic
remain excluded. Fast WB97M-V has a separate qualified fixed-anchor cold policy.

Previous unguarded negative decisions remain applicable:
[PBE0](../rejected/2026-10-09-pbe0-incremental-endpoint-promotion.md) and
[HF](../implemented/performance/2026-10-11-incremental-direct-jk-negative-closeout.md).
Revisit promotion with complete-endpoint evidence, actual work reduction,
unchanged observable budgets and explicit unsupported-owner fallbacks.

## Reproducibility

Local evidence:
`/home/jzzeng/codes/qc-branch-audit-20260922/evidence-delta-d-warm-20261022`.
Remote root on n1: `/data/jzzeng/delta-d-warm-20261022`.

All 5,920 relevant baseline build/runtime/test files were byte-verified against
the fetched master before edits. Measured V3 source and host tests match local
copies. Retained patches, snapshots, library hashes, drivers, arrays and logs
distinguish every revision. The evidence README and `summary-cold.json` contain
all raw samples, paired comparisons and independent reference identities.
GPU work used finite Slurm allocations on n1 RTX 5090, preserving scheduler
visibility. Compilation used verified ccache 4.5.1 and existing cache storage.

## 2026-10-23: GPU4PySCF delta control

A subsequent same-wheel PBE0/96 comparison explains why upstream can benefit
more. At its default screening 1e-13, three fresh-process pairs give complete
cold medians 21.24781 seconds full versus 20.17548 delta (-5.05%). Setup+SCF
improves 8.28%; both arms use nine iterations. Actual operator calls include
initialization: ten full versus one full plus nine deltas. The native prototype
instead uses six full plus three deltas. Upstream keeps the previous potential
through consecutive delta calls and has no corresponding per-delta refresh or
mandatory two-full-energy tail in the inspected RKS solver.

At screening 1e-12, upstream delta needs ten iterations versus nine full;
complete medians 20.69758/20.67049 seconds show no win. All twelve clean and
four diagnostic endpoints pass existing independent E/F gates. Keep the slow
22.58452-second clean delta sample and the separate 4.76-second K diagnostic
outlier. Native/GPU4 grid array sizes differ by 192 zero-weight padding entries,
as the earlier default-grid audit established; this is not a quadrature change.

The evidence report `gpu4-delta-comparison.md` retains source hashes, actual
full/delta/J/K counts, raw forces and separated component timings. It explicitly
does not assign a measured fraction to different screening error contracts.
The one-delta refresh rule is a conservative implemented policy, not an algebraic
necessity. Consecutive deltas with a bounded cumulative error ledger, selective
refresh and a final full proof remain a separate future candidate. The current
~2% native result is not an algorithmic upper bound or grounds to dismiss that
candidate; no such extension has been implemented or qualified in this task.
