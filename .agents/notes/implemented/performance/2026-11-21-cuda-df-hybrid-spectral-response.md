# Decision: Reuse the spectral frame for dense fitted Hessian response

Status: implemented
Date: 2026-11-21

## Problem

The 48 AO, 226 auxiliary PBE0 RKS water dimer fits the full-rank schedule,
but the same system with 12 roots discarded exceeded the generic spectral
term-visit bound (RTX 5090 Slurm `1774`). The pure Coulomb rank-one rule
cannot represent the exchange cotangent. Increasing the cap would admit
repeated frame rotations and two equivalent ordered cubic contractions.

## Decision

Use the accepted column-major eigenvectors `Q` as a row-major `Qᵀ` view
in the compiler-generated FP64 rotation maps. Retain `Qᵀ G Q` for the dense
inverse cotangent `G` and rotate `δM` once per direction. After evaluating
`δV`, reuse the same rotated metric motion in the moving raw-metric pullback.
The second divided difference is symmetric in its three arguments, and its
matrix seeds are symmetric, so evaluate one ordered leg and reflect it. Add
the first divided difference of `δG` before the return rotation. Keep the
stable branchwise difference formulas, finite checks and fixed-rank cutoff
separation. No CPU numerical path, new eigensystem or rank-one assumption
is introduced for exchange.

Just changing the rotations still exceeded the fitted owner's device cap:
the default depth-ordered graph built the raw Gram and fitted tangent legs
alongside one another. The emitted schedule now visits `δG` dependencies
first, then the remaining outputs and retained definitions. This retires
large scratch earlier without adding contractions or publishing partial
results. Final copies still follow the finite checks on the whole stream.

## Evidence and scope

At 48 AO and 226 auxiliary functions, the RKS hybrid plan (including its
retained raw source) admits 63,841,280 bytes against 64 MiB. The reference
plan charges 11,516,861 scalar visits and 434,515,968 padded matrix-product
slots. The directional plan charges 46,101,633 scalar visits and 733,667,328
padded slots. These are fitted-algebra admission counts, not FLOP estimates
or complete endpoint work counts. The same RKS shape also admits the
exchange-only schedule. The 48 AO, 226 auxiliary UKS hybrid plan still
exceeds both the 64 MiB budget and the one-billion tiled-product bound.

RTX 5090 Slurm `1775` completes the public PBE0 RKS HVP in 50.40 s
(one direction, 28 induced Fock actions, 13 response iterations, one DF value
pass and one DF contraction pass). Independent PySCF/Libxc energies rebuild
the truncated fitted ERIs and reconverge at each displacement; two projected
Richardson estimates have absolute errors `4.76e-8` and `1.89e-8 Ha/Bohr²`
against the `2e-7` gate. The oracle's successive Richardson estimates differ
by at most `5.31e-8`. Translation passes, and the native endpoint reports
zero numerical host fallbacks, intermediate panel downloads and displaced
SCF evaluations. The HVP differs from the full-rank oracle by `6.87e-5
Ha/Bohr²` at its largest component, confirming the truncated branch was
exercised.

Slurm `1778` passed 74 native and independent-oracle tests plus 17 public
endpoint regressions; 16 angular cases were not enabled in this replay.
The added 19-root fixtures exercise tail tiles across degenerate retained
and discarded clusters in both spin conventions. Widening the inner
energy-difference stencil reduced cancellation in the nested metric-weight
oracle; the directional acceptance tolerance did not change.

Source-matched logs, checksums, basis inputs and the independent probe are
retained in `build/hessian-gpu-sm120/r2scan-orbital-f/`, including
`hybrid-spectral-binaries.sha256`, `hybrid-spectral-work-estimate.log`,
`probe-pbe-dimer-spectral.py`, `hybrid-spectral-regression-final-n4.log`,
and `pbe0-dimer-spectral-{baseline,endpoint}-n4.log`.
These measurements qualify SM120 only.

The later [packed spectral tangent](2026-11-21-cuda-df-uks-packed-spectral-motion.md)
addresses the UKS limit documented in this measurement.
