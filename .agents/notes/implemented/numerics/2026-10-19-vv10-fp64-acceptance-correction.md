# Decision: withdraw unqualified VV10 FP64 grid recommendations

Status: implemented
Date: 2026-10-19

## Problem

The [quadrature population note](../performance/2026-10-18-readme-quadrature-populations.md)
recommended NLC32×12×24 / XC48×16×32 after an exploratory reference-only
comparison. That comparison used `1e-5 Eh` / `1e-4 Eh/bohr`, not the retained
README FP64 thresholds of `1e-8 Eh` / `1e-7 Eh/bohr`. Its maximum differences,
`6.38e-7 Eh` / `3.23e-6 Eh/bohr`, fail the FP64 thresholds. Same-grid native
parity and finite-difference agreement cannot repair quadrature error.

## Decision

Withdraw the grid recommendation for FP64 and label all existing timings as
exploratory fixed-grid measurements. Preserve the measurements, historical
exploratory gates, all other README methods and all earlier publications.
Do not change production defaults or relabel reference speedups as native wins.
Comparators report the requested gate and the FP64 threshold gate separately;
the split comparator defaults to the FP64 thresholds. A same-grid rerun report
explicitly does not evaluate grid convergence or overall precision admission.

NLC32 is at most an unqualified Fast candidate. Fast must be accepted against
its configured total budget, including RMS forces when configured, on the actual
native path and full cohort. The reference probe does not establish that budget.
The `PrecisionPolicy` profile named `strict` has its own tighter energy target;
these README publication thresholds do not redefine that profile.

## Evidence and conditions for revisiting

At fixed XC48×16×32, even NLC48×16×32 versus NLC64×16×32 differs by up to
`7.63e-8 Eh` / `3.86e-7 Eh/bohr` on 12 atoms. The original shared-grid study
also fails FP64 grid thresholds, despite same-grid engine parity. Neither the
old default nor the selected dense reference is newly certified converged.

Converge radial and angular resolution for XC and NLC independently, establish
a stable denser reference, and account for the combined error before admitting
a full endpoint. Native/reference parity remains an independent check. Rerun
all original eight HF/DFT methods and 3/12/48/96-atom cases for Direct and DF,
including GPU4PySCF; do not replace the historical overview with an incomplete
or unqualified cohort. No further coarse-grid FP64 promotion is authorized by
the earlier exploratory pass.
