# Decision: do not promote MINAO or orthonormal CDIIS for WB97M-V cold SCF

Status: rejected
Date: 2026-11-06

## Problem

The retained 96-atom Direct WB97M-V comparison reports 18 native molecular
steps versus 10 GPU4PySCF cycles. PR #3018 diagnosed differences in operator
staging, initial guess and CDIIS metric. The user asked whether those solver
differences could materially accelerate A100 without regressing RTX5090.
Plausible differences are not evidence of a performance benefit.

## Decision

Reject both measured controls for default promotion, and remove their
experimental production code, selector and temporary tests. Keep the native
Huckel/raw-AO defaults. Preserve one fixed A100 2x2 cohort and one 5090
raw-versus-orthonormal guard; do not search parameters until timing turns green.

The first control changes only the typed native initial guess to MINAO. The
second changes only the CDIIS history metric to
`X.T @ (FDS-SDF) @ X`, using resident `X` and existing scratch. Physical
AO residuals and every final publication gate remain unchanged. The prototype
supports restricted single-step execution only; it is not a production
contract for UKS, chunking or all DFT models. Timed native processes reject
imports of both PySCF and GPU4PySCF.

## Evidence

Actual A100 LSF225852, 18-minute `gpu_chem/gpu22` allocation:

| Initial guess / metric | Cold E+F (s) | Warm E+F (s), one sample |
| --- | ---: | ---: |
| Huckel / raw AO | 111.897483 | 15.095748 |
| MINAO / raw AO | 116.359662 | 15.108940 |
| Huckel / orthonormal | 113.102631 | 15.067734 |
| MINAO / orthonormal | 116.629770 | 15.120110 |

Actual RTX5090 Slurm16344, 18-minute `main/node1` allocation:
Huckel/raw cold **194.035070 s**, orthonormal **194.495628 s**; warm
three-call medians **14.353750 / 14.349936 s**. One cold sample per arm
does not establish a statistically significant small regression, but neither
device provides positive evidence for promotion.

More decisively, **all six cold arms retain identical semantic work**:
18 molecular iterations/Focks, nine semilocal/RSH seed steps, one full target,
four frozen-potential inner steps and four full-target tail steps. Thus every
arm still performs 13 provisional + 5 target Focks, 4 VV10 pair sweeps and
1 pair-cache hit. Warm is one physical Fock with no pair sweep. MINAO really
executes; its native preparation is about 2.55 s on A100 versus about 0.15 s
for Huckel. Its zero preliminary Focks do not imply cheaper preparation.

All 16 endpoint samples pass 32 comparisons against the pinned independent
static reference with unchanged intensive scientific gates. The largest
energy/atom, force-max and force-RMS errors are respectively `7.37e-12`,
`4.81e-7`, `2.01e-7` in Hartree/Bohr units as appropriate. No new reference
timing or GPU4PySCF ranking was measured.

The first water parity test did fail: two cross-trajectory force differences
around `2e-7` exceed its new `1e-7` arithmetic-parity bound. The default MO
stationarity gate does not promise that inter-trajectory force precision.
The same parity bound passes under tighter explicit SCF controls; independent
default-profile GPU4PySCF original/moved-geometry checks also pass unchanged
scientific gates. Preserve the original failures and exact test snapshots;
do not confuse this diagnosis with permission to loosen production accuracy.

Base master is `318ae3b99f7af1e7fdc83d33dc7acd9da2917efb`; both cards run
source identity `e66dfb990044fd595d0969df728c0ecbfb0f8a0114570aa48796d8c041aefc95`
and ELF SHA256 `8cd1bf438532d52c763cc95e817845e7566d6f911fea565f5189e1dc719ea1dd`.
Verified sccache 0.16.0 serves compilation; device visibility remains exactly
scheduler-assigned. Transport failures do not trigger duplicate GPU work.
The n1 borrowed image and owned A100 transfer process have explicit restoration
and cleanup receipts.

## What explains the observed 17/18 steps, and what remains unmeasured

The user correctly challenged treating a stage count as a causal explanation.
The measured native algorithm solves three successive fixed-point problems:
`F0[D]` without VV10; `F0[D] + Vnl[D_anchor]` with a frozen VV10 potential;
then the self-consistent `F0[D] + Vnl[D]`. The retained GPU4PySCF implementation
instead updates the full target through one continuous CDIIS solve. These are
different solver policies, not simply different guesses for the same policy.

The initializer and frozen-inner refresh predicate compares the **AO-element
maximum** commutator with its provisional tolerance, not the printed spin RMS.
For this Standard outer route the seed tolerance is `1e-5`; the frozen-inner
selector also resolves to `1e-5`. Thus a printed step-seven RMS `4.95e-6`
does not establish that the initializer's refresh predicate already passed.
Actual AO maxima were not retained: the branch outcomes can be inferred from
the finite-budget controller, but their numerical values must not be invented.

At step nine the incomplete-operator spin RMS reaches `1.46e-7`; introducing
VV10 at ten raises it to `5.06e-5`, about **346x**. Therefore the initializer's
late residual reduction is not an equivalent reduction of the full-target
error. Handoffs after nine and fourteen also clear CDIIS history and restart
the target energy epoch. Step fifteen is consequently ineligible for final
convergence even if its residual were good: the target-ready contract needs
two fresh target observations. The extra iterations are consequently produced
by a deliberately staged solver, repeated operator correction, and restarted
history, rather than demonstrated inferiority of its initial guess/metric.

The existing diagnostic then identifies the tail predicates precisely: step
sixteen passes the energy gate but fails the physical MO-gradient gate;
seventeen passes stationarity but has `|delta E| ~= 1.215e-9 > 1e-9`.
Eighteen checks unchanged density, passes energy, and accepts its first
canonical probe. The seventeenth state and eighteenth confirmation must not
be conflated; density-change stopping or repeated canonical rejections are
not the demonstrated reason for that tail.

This explains the controller decisions producing the observed trajectory,
**not a quantified counterfactual attribution of seven avoidable steps**.
No otherwise identical continuous-full-target run was measured here, and no
experiment isolates the cost of each history reset. The smaller number of
expensive VV10 evaluations may pay for additional cheap provisional iterations.
Calling all provisional work waste, claiming these resets alone cause exactly
seven lost steps, or claiming a ten-cycle alternative is necessarily faster
would overstate the evidence. Initial-guess/metric rejection is complete;
that broader causal/performance comparison is not.

## Consequences and revisit conditions

The guess/metric differences are real, but **they do not explain a removable
iteration gap in this cohort**. GPU4PySCF's 10 cycles also exclude its initial
Veff build, whereas native steps include deliberately provisional operators.
Matching a displayed cycle count is not the objective.

The remaining candidate is algorithmic work at the semilocal seed / frozen
VV10 / fresh-target handoffs and the expensive exact VV10 pair sweeps. This
cohort does not demonstrate a safe way to eliminate any sweep. A future
proposal needs a concrete operator/work argument before another bounded
experiment; repeating these two controls, switching packet widths, or simply
discarding a tail check is not justified. Reusing CDIIS history across
different operators needs an explicit correctness/ownership design; a good
semilocal residual does not certify the full target. Do not weaken grids,
screening, precision or energy/MO gates to force fewer printed steps.

Revisit these controls only with a distinct failed-system diagnosis, a changed
solver/DIIS implementation, or new complete-endpoint evidence that actually
reduces target/provisional work while preserving independent E+force gates
and both-device qualification. No claim is made that either choice is worse
for every molecule or method.

## References

- Issue #2976: external evidence/decision memory.
- PR #3018, diagnosis commit `b45934475d6b6d44ec84fac2a4c8fa5c7c9b9032`.
- [Retained cohort and stdlib replay](../../../benchmarks/results/wb97mv-scf-controls-20261106/README.md).
- [Performance engineering](../../../docs/maintainer/performance_engineering.md).
