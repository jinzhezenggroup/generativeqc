# Decision: Share the scalar ECP owner with the native r²SCAN+rVV10 Hessian

Status: implemented
Date: 2026-11-21

## Problem

The combined r²SCAN+rVV10 endpoint already differentiated a self-consistent
nonlocal quadrature, and bare r²SCAN already had a scalar ECP core and
response. An all-electron guard in native KS preparation, public capability
selection and resource planning prevented combining them. Lifting only the
public guard would admit a request the native owner still rejected.

## Decision

Admit scalar semilocal ECPs for the canonical r²SCAN+rVV10 composition only
when SCF selects the linked variational r²SCAN point domain: strict FP64,
device-fused XC, no application precision budget, order-3 primary and
nonlocal grids, and orbital shells through f. Authenticate the exact rVV10
parameters and forbid range and full-range exchange. Continue to require the
matching integral inventory and the existing ECP quadrature and DF metric
gates. The guard runs at preparation, independent of which observable the
caller will request.

The effective-charge core and residual ECP second derivatives do not modify
the rVV10 molecular active set. They enter the same stationary scalar action
as for bare r²SCAN; a fitted primary still supplies only the two-electron
response. The resource planner uses the existing concurrent ECP setup and
nonlocal reservations. No standalone force support or CPU fallback is added.

## Evidence

- Slurm 2273 (RTX 5090) passed the public Python/native CLI dense Hessian and
  HVP comparisons for all-electron and ECP RKS/UKS. The ECP cases exercise
  the Direct shared grid and explicit-DF independent grid, including budget
  admission, peak recorded bytes and response release.
- Slurm 2274 captured eight native ECP matrix/HVP pairs (RKS/UKS × Direct/DF
  × shared/independent grids). The independent oracle rebuilds the displaced
  geometry, Libcint ECP and Coulomb data, Libxc r²SCAN, and the direct-space
  rVV10 energy, then reconverges the determinant at each geometry. This is a
  check of the total energy surface, not a stencil of the native Hessian.
  All eight passed, with maximum Hessian and action errors of
  `3.18e-7 Ha/Bohr²` and `5.12e-8`, respectively.
- Slurm 2276 passed public/native agreement, translation, symmetry, HVP
  replay, and budget/release checks for UKS DF with an f orbital and a g
  auxiliary shell. The independent energy reconvergence ran outside the GPU
  allocation and passed, with maximum Hessian and action errors of
  `9.91e-9 Ha/Bohr²` and `1.75e-9`, respectively.
- The host frontend, budget and KS resource suites passed 169 tests. Build,
  scheduler and oracle logs are under `build/hessian-gpu-sm120/r2scan-orbital-f/`
  (`hessian-rvv10-ecp-*`, `rvv10-ecp-evidence/`).
