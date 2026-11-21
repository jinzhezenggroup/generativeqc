# Decision: Compare the d-orbital WB97M-V Hessian on the deployed VV10 domain

Status: implemented
Date: 2026-11-21

## Problem and diagnosis

The sphere-based d-orbital fixture in the
[fitted LR qualification](2026-11-20-fitted-long-range-df-stationary-hessian.md)
showed an independent Fock mismatch for both LR exchange providers. At the
same accepted density, subtract independently rebuilt libcint DF J/K and the
one-electron matrix from the native Fock. The remaining difference to the
independent grid potential was `1.3478e-6 Eh` for both UKS spin channels.
Disabling Libxc's optional FHC projection did not affect the mismatch.

The native grid probe on the same packed AOs, weights and density reproduced
the isolated SCF potential to `6.3e-14 Eh` (RTX 5090, Slurm job 1672). The
native molecular VV10 policy selected 576 rows; the oracle selected 550. The
old oracle had kept the `rho >= 1e-8` criterion while both native SCF and the
native response had moved to `rho >= 1e-10` on both pair legs.

## Decision

Read the declared molecular domain in the test selector and hold the
reference active set fixed during displacement. Retain external libcint and
Libxc plus the independent real-space VV10 values and gradients; do not use
the generated response or loosen tolerances. A separate density-domain test
pins the actual threshold boundary and signed-weight behavior. This is an
oracle fix, not a change to the Hamiltonian or production derivatives.

## Evidence

Previously retained d-orbital receipts for UKS exact and fitted LR and RKS
fitted LR pass the original reference Fock/density gates with the corrected
domain. At the published densities the worst Fock error is `6.5e-14 Eh`,
and the independently reconverged energies disagree by at most `4.5e-14 Eh`.
For the retained fitted-LR directions, two displaced-energy stencils give
`|vᵀHv − d²E/dt²|` of `1.94e-10` (RKS) and `8.45e-9` (UKS) Ha/Bohr².
See `build/hessian-gpu-sm120/wb97m-orbd-oracle-triage.log`,
`wb97m-orbd-grid-gpu.log`, `wb97m-orbd-oracle-corrected.log`,
`wb97m-orbd-fock-corrected-errors.log` and `wb97m-orbd-direction-oracle.log`.

The complete spherical d-orbital Direct RKS matrix on RTX 5090 (job 1673)
has a worst independent reconverged stencil error of `8.72e-10 Ha/Bohr²`.
It also passes raw symmetry, translation nulls and HVP/matrix parity. The
per-provider/spin gate uses the original matrix tolerance and verifies the
complete fitted-primary/exact-LR and fitted-primary/fitted-LR surfaces
separately; an HVP projection alone does not qualify either full matrix.

For long cases, capture each native matrix and its central state under Slurm,
then run the same `_qualify_wb97mv_d_orbital_receipt` gate on CPU. Job 1673
reached its time limit during Direct UKS energy reconvergence, after the
native Direct UKS matrix had been saved. Job 2068 supplied the four remaining
fitted matrices without holding a GPU during displaced-energy reconvergence.
Native receipts alone do not pass the independent full-matrix gate.

The independently reconverged fitted-primary RKS matrices from job 2068
pass: `6.15e-10 Ha/Bohr²` maximum error with exact LR and
`8.81e-10 Ha/Bohr²` with fitted LR.
All three UKS matrices also pass, with maximum errors of `1.30e-9`,
`1.62e-9`, and `3.63e-9 Ha/Bohr²` for Direct, DF/exact LR, and DF/fitted
LR respectively. The independent matrices and both stencil estimates are
retained in `build/hessian-gpu-sm120/wb97m-orbd-receipts/`.
