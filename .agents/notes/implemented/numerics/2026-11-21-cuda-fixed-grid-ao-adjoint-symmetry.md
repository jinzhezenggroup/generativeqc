# Decision: Publish a canonical AO pair for the fixed-grid adjoint

Status: implemented
Date: 2026-11-21

## Problem

The PBE0 UKS Hessian of doublet H₂⁺ with spherical s/f orbitals and the
`8/4/8` product grid failed intermittently, with and without a `ResourceBudget`.
Changing the CPKS residual tolerances did not reliably help. On Slurm 2269,
the density direction and XC point-response gates had passed. The assembled
empty-beta induced XC potential contained the transposed pair
`(-3.24519012645e-10, -3.23382144319e-10)`, outside the existing symmetry
tolerance. The raw nuclear Hessian had not been reached. An empty beta
density has zero variation, but alpha motion can still change its XC
potential through cross-spin correlation.

## Decision

Each fixed-grid AO feature is a symmetric bilinear in AO labels. Its density
adjoint is therefore symmetric for any admitted physical coefficients. The
two independently rounded evaluations of that adjoint can disagree near
cancellation, since weighted AO products and their gradient partners are
associated in different orders. Publish the canonical upper triangle to
both entries of each unordered AO pair. The compiler opts in for this
specific fixed-grid output; general affine maps are unchanged.

Do not relax the symmetry threshold, drop the empty-spin potential, change
the functional, or symmetrize a general input to hide a failed audit. Both
triangles' intermediates are still checked for finiteness before publication.
This uses the existing output launch and workspace with unchanged
contraction counts. The density, assembled induced Fock and raw nuclear
Hessian retain their existing audits.

## Evidence

- The original `direct-f-pbe0-uks` budget regression (same molecule, basis,
  grid, and default CPKS options) passes on Slurm 2270 for the unbudgeted
  matrix and budgeted matrix/action. The public matrix and HVP also pass a
  `3e-8 Ha/Bohr²` absolute gate against an independent Libcint/Libxc oracle.
- The oracle in `tests/data/xc/pbe0-uks-direct-f-hessian.json` uses PySCF
  UKS(PBE0) with the same finite molecular partition rebuilt at each of 145
  geometries. It reconverges each calculation (`conv_tol=1e-14`,
  `conv_tol_grad=1e-11`, no density screening), then Richardson-extrapolates
  central diagonal and mixed energy differences at `0.003` and `0.0015` Bohr.
- Slurm 2271 passed 25 tests, including the four Direct f-shell
  PBE0/B3LYP × RKS/UKS budget cases, independent fixed-grid references, and
  failure/publication gates. The original PBE0 UKS case also passed two more
  fresh-process replays in the same allocation.
- Slurm 2272 passed all 11 remaining global resource-budget cases, including
  fitted hybrid and shared/independent rVV10 grid contexts.
- The fixed-grid probe checks exact transposed equality separately for the
  semilocal, nonlocal and combined adjoints, in both spin layouts. An
  independent Libxc/VV10 finite-difference gate checks the adjoint values.

The local replay script, scheduler logs and compiler-cache statistics are
under `build/hessian-gpu-sm120/r2scan-orbital-f/` (`hessian-direct-f-*`).

## Revisit when

Revisit accumulation only if an actively occupied channel needs tighter
accuracy near a cancellation. Canonical publication alone does not improve
the accuracy of a single FP64 sum; it prevents equivalent AO labels from
receiving different rounded representations of that sum.
