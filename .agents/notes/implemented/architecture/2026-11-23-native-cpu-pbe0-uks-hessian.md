# Decision: Bind canonical CPU PBE0 UKS to the native stationary Hessian

Status: implemented
Date: 2026-11-23

## Decision

The default host Hessian inventory binds the exact polarized PBE0 electronic
MethodIR to the existing Direct FP64 CPU stationary owner. Named uncut,
two-body D3(BJ) uses the same electronic response and the existing additive
geometry-only column. SDK, CLI and Python use the same prepared batch and token.
The admission is all-electron, with canonical exchange scaling and scalar
response columns. A package without the exact polarized binding cannot route
to the historical Python electronic fallback, even for bare Cartesian UKS.

Fixed-grid induced response, moving-grid RHS, and frozen curvature use the
physical PBE point partials with 3/4 semilocal exchange and full correlation.
The inputs are separate alpha and beta density/gradient blocks; the restricted
adapter continues to use total fields. A zero-density spin admits only a zero
physical tangent, not particle creation. Both integral contraction and XC
registries derive their spin sets from the same inventory. The B3LYP scalar
leaves and LDA/PBE RKS domains are unchanged.

## Why not additional scalar leaves?

The [earlier scaled-PBE0 attempt](../../rejected/2026-11-22-pbe0-host-hessian-maple-leaves.md)
failed at positive dilute-tail points in an ordinary molecular grid. Reuse the
SCF-domain evaluator and physical JVP rather than generating an uncontracted
rho/sigma Hessian or suppressing tail work. RKS-compatible field factors or a
fabricated beta tangent would also violate UKS response semantics.

## Validation

At baseline `cae2959e1` plus the worktree changes, independently reconverged
same-grid PySCF UKS on distorted water cation checks the accepted energy,
spin-resolved D/W, J and exchange/XC energies, analytic forces, and every raw
Hessian column. Cartesian STO-3G and spherical def2-SVP gave maximum column
errors of `1.42e-7` and `1.68e-7` Hartree/Bohr² at step `3e-4` Bohr. The raw
matrix also agrees with the native HVP, obeys translation and symmetry gates,
and never symmetrizes after assembly. D3(BJ) has a separate force-difference
test of the small additive contribution without extra electronic solves.

The installed SDK/CLI/Python parity checks exercise both bare and corrected
open-shell HeH. The SDK replays exact derivative capacity, refuses one byte
below it without output, rejects stale geometry after moving a warm batch,
and succeeds at the moved geometry. SDK and CLI subprocesses have no Python
or toolchain on their runtime path. No real CUDA device was available via
Slurm, and this change makes no CUDA claim.

Complete installed-library Python endpoints on the STO-3G water controls were
timed in five pairs after warmup. PBE0 RKS median HVP/dense times changed by
`+1.38% / +2.37%` and B3LYP UKS by `-0.14% / +0.08%`, with matching outputs
and identical SCF/response, integral, XC and reservation counters. The new
PBE0 UKS HVP/dense endpoints took `0.475 / 2.642 s`; D3(BJ) took
`0.482 / 2.645 s`, with no before timing because the endpoints were unlinked.
For the new bare slice, 19 SCF Fock builds precede 28/242 induced Fock actions
and 60,480/522,720 response XC point visits (HVP/dense respectively). D3(BJ)
leaves those electronic counts unchanged.

Raw results, compiler-cache statistics, and complete endpoint timing/work
receipts: `/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-pbe0-uks`.
