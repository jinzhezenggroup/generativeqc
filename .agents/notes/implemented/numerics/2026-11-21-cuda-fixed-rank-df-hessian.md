# Decision: Differentiate the accepted CUDA DF spectral branch

Status: implemented
Date: 2026-11-21

## Problem

The native CUDA Hessian previously required each fitted radial metric to be
full rank. Once SCF discards auxiliary directions, the inverse derivative
`-V δM V` misses the motion of the retained subspace and gives a different
energy Hessian even when no eigenvalue crosses the cutoff.

## Decision

Lend the accepted single-system SCF eigensystem and branch flag under the same
identity as the raw integral source and whitening. The full-range and erf
providers classify independently. Induced CPKS acts at fixed geometry and uses
the existing `V = WᵀW` regardless of rank. A truncated provider differentiates
`V = f(M)`, where `f(λ)` is `1/λ` on retained roots and zero on discarded roots.
The compiler-generated fitted-energy maps end at `G = ∂E/∂V` and `δG`. The
component computes `δV = Df(M)[δM]`, the raw weight `Df(M)[G]`, and its motion
`Df(M)[δG] + D²f(M)[δM,G]`. It reuses the original spectrum, first Fréchet
action and spectral rotations, without rediagonalizing or storing a dense
second-derivative tensor. Full-rank providers keep the inverse rule.

## Invariants

- A root near the relative cutoff invalidates the derivative, not the SCF
  energy. Neither freezing the selected eigenspace nor changing the fitted
  radial operator is an admissible fallback.
- Same-branch divided differences have analytic coincident-root limits. Only
  cross-branch terms depend on spectral gaps; scaled products avoid premature
  FP64 overflow from separately forming coefficients or seed products.
- Borrowed spectra, whitening and raw sources are read-only. Component and
  molecular publication must not overwrite that storage. Extra spectral scratch
  is quadratic and the cubic work is included in explicit admission bounds.
- CPU fitted derivatives do not inherit this CUDA support.

## Evidence and reproduction

Under Slurm on an RTX 5090, `test_cuda_pseudoinverse_second.py` (job 2078)
compared the second Fréchet action to re-diagonalized mixed differences and
checked degenerate roots, trace pairing and extreme FP64 scales.
`test_native_wb97mv_df_response.py` (job 2079) qualified separated RKS/UKS J,
J/K and K-only components against independent energies, with cutoff, alias
and workspace rejection.

`test_native_wb97mv_stationary.py -k 'fixed_rank_fitted_stationary or
metric_cutoff_preserves_device_output'` exercises the accepted molecular owners
in RKS H₂ and UKS H₃. Each displaced energy independently reconstructs the
Libcint metric, re-diagonalizes it, checks the branch rank and reconverges KS.
The probe records the raw Hessian, HVP, Fock and density. The paired CLI tests
exercise the linked library and Python frontend without a numerical fallback.

Molecular evidence on sm_120 (RTX 5090), using Richardson-extrapolated
energy stencils at 0.004 and 0.002 bohr:

| Model | Primary / erf rank | Max matrix error (Ha/bohr²) |
| --- | --- | ---: |
| WB97M-V RKS H₂, only erf truncated (job 2080) | 2 / 1 | 9.11e-10 |
| WB97M-V UKS H₃, only erf truncated (job 2080) | 3 / 2 | 5.16e-9 |
| WB97M-V RKS H₂, both truncated (job 1675) | 1 / 1 | 9.22e-10 |
| WB97M-V UKS H₃, both truncated (job 1675) | 2 / 2 | 3.85e-9 |
| PBE RKS H₂ (job 1674) | 1 / — | 9.87e-10 |
| PBE0 UKS H₃ (job 1674) | 2 / — | 3.86e-9 |

Jobs 1674–1675 also exercised Python/CLI Hessian and HVP agreement on the
full-rank, erf-only truncated and doubly truncated RKS models, and rejected
cutoff-adjacent derivatives at the Python, CLI and native device boundaries.
The double-truncation probe receipts were checked offline against independent
energies, as was the UKS erf-only receipt after job 2080 reached its time limit
during the host oracle. The numerical oracle did not run inside the native
executable.

## Revisit when

If large auxiliary sets hit the cubic-work limit, consider an alternative
validated schedule that preserves both subspace motion and the immutable
owner boundary. Changing numerical representation does not authorize a
rank-crossing response or a Hamiltonian substitution.
