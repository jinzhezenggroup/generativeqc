# Decision: retain defaults after shorter-SCF proposal experiments

Status: rejected as production defaults
Date: 2026-10-24

## Problem

The strict-convergence advantage largely disappears with relaxed controls.
The historical [GPU4PySCF DIIS diagnosis](../proposed/2026-10-09-pbe0-gpu4pyscf-diis-tail.md)
isolates one cause: an absolute augmented-system eigenvalue cutoff discards
useful small-residual directions. Common Gram normalization reduces that
reference's median from 42 to 19 cycles without changing its strict controls.
Native already normalizes its KS metric. A relaxed solve usually stops before
this asymptotic failure dominates; this mechanism cannot promise an equally
large advantage in the first nine iterations. Those historical measurements
retain their original sources and are not relabeled as current-master timings.

The [Standard convergence policy](../implemented/numerics/2026-10-22-standard-cold-scf-convergence.md)
also removed a separate conservative native stopping tail. On frozen master
`48918670f59a8e10c254c822565a17796aa42f54`, PBE0/96 now needs nine relaxed target
Focks versus fourteen with explicit strict controls. Both counts include actual
target work; the seven native atomic-HF guess Focks are reported separately.

## Decision and evidence

Retain the current solver and all accuracy gates. The
[source-bound experiment and replay](../../../benchmarks/results/scf-convergence-probes-20261024/README.md)
cover 76 fresh-process populations and 152 complete returned E/F endpoints on
Slurm n4 RTX 5090. This is a PBE0 water/def2-SVP stock-grid diagnostic, not broad
production qualification or a refreshed cross-engine comparison.

Changing DIIS history length alone does not improve relaxed 96-atom convergence;
four entries make it worse. Orthogonalizing only the proposal residual also
leaves that count unchanged while adding two matrix products per step. Earlier
Fast precision handoff can reduce target iterations, but clearing history
increases the strict recovery work.

The best retained-history candidate switches at maximum AO residual `1e-2`.
Three independent alternating confirmation rounds reduce 12/48/96-atom target
Focks from 10/9/9 to 8/8/8. It replaces 8/7/7 mixed plus two strict applications
with five mixed plus three strict applications. Complete cold medians become
1.30%/1.51% slower at 12/48 atoms and only 1.42% faster at 96 atoms, below the
prespecified 2% improvement floor. Warm calls remain one strict Fock with no
demonstrated gain. Thus the iteration reduction is real but does not justify
changing the default for the performance objective.

The `1e-3` retained-history 96-atom counterexample is stronger: its nine-step
candidate fails the force consumer's final physical/canonical proof, leading
to seventeen additional strict Focks. The subsequent warm call also falls back
(four plus seventeen). Both returned endpoints pass independent force gates,
but their actual target counts are 26 and 21. Their rejected attempts' atomic
work is unavailable, not inferred from the successful fallback's guess record.

## Invariants and revisit conditions

- Mixed histories may only propose a density. They cannot authenticate a strict
  energy predecessor, physical stationarity, final canonical frame or force.
- Preserve the new strict energy epoch, full-density target observations,
  final-state validation, bounded history retirement and explicit fallbacks.
- Count complete endpoint work across failed proposals and retries. A shorter
  returned history or fewer inexpensive mixed Focks is not sufficient evidence.
- Qualify a stronger early-stage solver or initial guess against unchanged
  independent observables, complete cold/warm endpoints and displaced/holdout
  cases. Revisit these policies only when such evidence also demonstrates a
  robust endpoint improvement; do not weaken final force validation to suppress
  the retained counterexample.

Only diagnostic artifacts and this rationale are retained. All experimental
production edits were removed; the frozen patches remain reconstructible.
