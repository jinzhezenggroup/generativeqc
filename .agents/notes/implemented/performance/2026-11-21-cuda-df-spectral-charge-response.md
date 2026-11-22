# Decision: Reuse the rotated metric direction for the rank-one Coulomb seed

Status: implemented
Date: 2026-11-21

## Problem

[Full-rank charge response](2026-11-21-cuda-df-coulomb-charge-response.md)
admits the 48 AO, 226 auxiliary PBE water dimer, but its separated truncated
metric still used the generic dense inverse tangent and mixed pullback. The
generic spectral action's bound alone was 369,381,632 term visits at this
shape, above the 100-million cap. Slurm `1766` reproduced that rejection with
an independently separated cutoff.

## Decision

For `b = A:D` and the accepted eigenvectors `Q`, retain `Q^T b` alongside
the fitted charge `W^T W b`. Rotate `dM` once to the eigenbasis. A first divided
difference followed by matrix-vector multiplication gives `(dV)b`, which
enters the frozen-density Fock tangent. The source graph emits `db = dA:D + A:dD`.
Apply the moving inverse seed `d(bb^T/2)` in the same eigenbasis. For the second
Fréchet action on `bb^T/2`, contract the first ordered leg against the retained
charge and add its transpose (the second ordered leg). Combine the two metric
terms before rotating back. Use the existing scaled first and second divided
differences; retain the general dense rule for exchange. The accepted `W`
determines the primal charge, and cutoff stability and authentication are
unchanged.

This path does not form an explicit inverse, dense inverse tangent, fitted
AO-pair panel, or cubic scratch. The conservative bound at the dimer shape is
29,910,914 visits during preparation and 89,704,952 per nuclear action. The
upper bound counts matrix rotations, the ordered contraction including both
rationalized contributions, and quadratic work. The host-only probe admits
both spin conventions at this shape and rejects a larger spectral shape. No
limit increase or fallback is needed.

## Evidence

RTX 5090 Slurm `1767` used spherical cc-pVDZ/full def2-universal-JKFIT with
the existing PBE water dimer geometry and quadrature. A relative cutoff of
`3.636967353327937e-6` discards 12 of 226 roots. The independent reference
rebuilds the raw integrals and truncated inverse and reconverges PySCF/Libxc
at every displaced geometry. Three energy-difference steps (`0.016`, `0.008`,
`0.004` Bohr) give two projected HVPs; consecutive Richardson estimates differ
by at most `3.54e-8`. Native projections agree to `3.33e-8` and `1.12e-9`
Ha/Bohr². The native HVP differs from the full-rank oracle by up to `7.96e-5`
Ha/Bohr², and passes the translation check. This is projected endpoint
validation; the small independent spectral regression separately checks all
individual Fock and source-weight components, including degenerate clusters.

The endpoint took 50.21 s and charged 26 induced Fock actions, one DF value
pass, and one DF contraction pass, without host numerical fallback, displaced
SCF, or an intermediate panel download. This validates admission and accuracy,
not an endpoint speedup. Reproduction inputs and source-matched checksums are
under `build/hessian-gpu-sm120/r2scan-orbital-f/` (`probe-pbe-dimer-spectral.py`,
`pbe-water-dimer-oracle/spectral-reference.npz`, `spectral-charge-binaries.sha256`,
`pbe-dimer-spectral-endpoint-n4.log`).

Slurm `1768` passed 48 native fitted-algebra tests and 17 selected public
frontend regressions (16 angular cases were not enabled for this replay).
The native suite includes independent finite differences through repeated
retained/discarded eigenvalue clusters for pure Coulomb, hybrid and
exchange-only sources, as well as cutoff rejection on both sides.

## Revisit when

Larger hybrid or exchange-only systems still hit separate preparation and
action bounds. Keep the dense general rule until an independently qualified
schedule handles its non-rank-one inverse seeds. A second GPU architecture is
not covered by the SM120 Slurm inventory used here.
