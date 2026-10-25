# Decision: evaluate complete auxiliary metric values once per unordered pair

Status: implemented
Date: 2026-10-25

## Problem and decision

The source preparation kernel evaluated both orientations of every public AO
pair in the symmetric two-center Coulomb metric. Complete value panels now
evaluate the lower triangle and write both orientations. Diagonal entries are
evaluated and written once. The primitive-cooperative mapping assigns a whole
warp to one pair, so the early exit remains uniform across its reduction.

Partial/offset panels and coordinate derivatives retain their existing ownership
and cannot write outside their caller's interval. Public Cartesian/spherical
expansions, normalization, metric rank threshold, generated recurrence, allocated
capacity, and subsequent symmetrization are unchanged. The two old orientations
could differ by roundoff; choosing one value therefore requires independent
numerical qualification, not a bitwise equivalence claim.

## Evidence and limits

The [retained capsule](../../../../benchmarks/results/df-metric-values-20261025/README.md)
contains 54 complete energy/force calls in three alternating rounds, 12 independent
original/moved/rebuilt-owner checks, 96 native source/oracle cases, 50 GPU tests,
and a zero-error compute-sanitizer run. The full-panel source domain changes from
`naux*naux` to `naux*(naux+1)/2` evaluations; dense publication/storage is unchanged.

PBE96 cold median changes from 15.749 to 15.545 seconds; r2SCAN-3c96 from 33.271
to 32.506 seconds. PBE48 and warm medians show no material improvement. This is
a preparation optimization, not complete GPU4PySCF parity or a new README cohort.

The larger screening, force tile/capacity, initial-guess and staged HF experiments
are independent research and are not enabled by this change. Do not infer that
metric symmetry authorizes folding arbitrary nonsymmetric derivative weights.
