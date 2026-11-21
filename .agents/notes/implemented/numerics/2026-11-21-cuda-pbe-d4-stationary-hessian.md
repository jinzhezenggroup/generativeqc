# Decision: Compose the PBE-D4 Hessian with the primal correction owner

Status: implemented
Date: 2026-11-21

## Why

D4 is geometry-only and does not enter the stationary electronic model token.
This is not evidence that a PBE-D4 result has the curvature of bare PBE. Both
the intrinsic PBE-D4 method enum and a catalog PBE carrier plus dispersion
descriptor can create the primal correction; combining the two routes would
double-own it. The CLI uses the named descriptor with its bare carrier. Python
rejects a compatibility correction evaluated outside the native KS batch rather
than labeling the bare electronic Hessian with the corrected method identity.

## Decision

Borrow the authenticated standard-profile plan and add its D4(BJ)/ATM/EEQ
directional action to each private electronic candidate. Unlike r²SCAN-3c,
the standard-profile owner has no gCP scratch allocation, launch or status.
The same aggregate arena accounting and device numerical gate precede
publication. The correction neither changes the DF provider nor requests a
second electronic response solve. The Python receipt ABI is unchanged.

## Qualification

On RTX 5090 (Slurm job 1658), an independent libcint/Libxc reconverged PBE
surface plus dftd4 3.7.0 supplied two energy stencils on bent STO-3G water.
The full raw Hessian max absolute errors were `5.39e-8` (Direct) and `3.25e-8`
(full-rank DF) Ha/Bohr². Both exercised nonzero ATM and checked raw symmetry,
translation nulls, and matrix/HVP parity without post-hoc repairs.
See `test_pbe_d4_cuda_complete_hessian` in
`tests/python/test_native_wb97mv_stationary.py` and the local
`build/hessian-gpu-sm120/pbe-d4-water-qualified-gpu.log`. Public CLI and Python
matrix/HVP parity passed both integral providers on RTX PRO 6000 (job 5481).
The intrinsic PBE-D4 DF constructor now skips the electronic-only split-hybrid
force predicate, without enabling DF forces for that corrected method.

The coarse 8×4×8 water grid failed in the bare-PBE electronic endpoint as
well as in the corrected endpoint. Qualify PBE-D4 on the 16×8×16 fixture
instead of relaxing a numerical gate or interpreting the coarse-grid failure
as a dispersion error. Do not infer wider D4, UKS, ECP or first-gradient
qualification from this test.
