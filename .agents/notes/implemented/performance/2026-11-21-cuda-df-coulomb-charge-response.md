# Decision: Apply the pure Coulomb metric in auxiliary charge space

Status: implemented
Date: 2026-11-21

## Problem

PBE RKS water dimer (spherical cc-pVDZ, 48 AO, full def2-universal-JKFIT,
226 auxiliary functions) exceeded the 100-million scalar-term limit before
its first Hessian action. The fixed-geometry response owner charged 118,250,884
terms for whitening the entire AO-pair panel alone. Retaining the fitted AO-pair
legs from [the hybrid response](2026-11-21-cuda-df-retained-fit-hessian.md)
also performs unnecessary work for pure Coulomb.

## Decision

For fixed geometry, contract the spin-summed density into `b = A:D`, solve
`x = W^T (W b)`, then form `J = A x`. Project the input and output onto symmetric
AO coordinates; accept signed directions without an occupied-density factor.
Retain the raw panel, not an extra whitened copy. This uses the same accepted
whitening, even when SCF discarded separated metric eigenvalues.

For full-rank nuclear response, retain `x` across directions. Differentiate
`M x = A:D`: at fixed density, `dx = W^T W (dA:D - dM x)`, and
`dJ = dA x + A dx`. For source weights, include `A:dD` in the right-hand side
and differentiate the external density in `(A*, M*) = (D x, -x x^T/2)`.
Use the same auxiliary-vector solve primitive in both compiler graphs. Do not
use the full-rank tangent for a truncated metric: it misses motion between the
retained and discarded eigenspaces. Keep the existing spectral pullback and
the separate exchange paths.

## Admission and verification

Use the existing authenticated SCF stream and source, device allocation limits,
finite/symmetry checks, and work accounting. No change to cutoffs, term caps,
or failure behavior, and no CPU numerical fallback. For the dimer shape, the
generated fixed-geometry J and assembly graphs charge 1,749,900 visits per
evaluation, including validation and broadcasting. The full-rank nuclear
graphs charge 6,467,030 visits for charge preparation plus the reference map,
and 8,544,736 per direction for the response map. These are scheduled affine
term counts, not wall times or the work of the raw integral derivative kernels.

Compiler tests use expanded NumPy ERIs, a nonsymmetric whitening gauge, and
complex-step metric/source motion for both spins. The host-only native probe
admits both spins at the dimer shape and retains oversized-hybrid rejection.
The larger hybrid and spectral nuclear paths still have their own work limits.

On RTX 5090 (Slurm `1759`), the public Python HVP completed in 50.19 s with
26 induced Fock actions, one DF value pass and one DF contraction pass. The
CPKS residual was `9.25e-11` at a `1.95e-10` tolerance. The native receipt
reports 304,115,280 device bytes (20,075,520 for the nuclear DF owner), no
numerical host fallback, no intermediate panel download, and no displaced SCF.
This is one pass at one shape, not an endpoint speedup claim.

The fixture's smallest metric root relative to its largest was `2.20e-8`.
The oracle used independent PySCF integrals and Libxc, sharing only the basis,
geometry and molecular quadrature definition (18 radial, 8 polar, 16 azimuthal).
Central differences of independently reconverged gradients used `8e-4` and
`4e-4` Bohr, with a maximum step-to-step change of `6.36e-9 Ha/Bohr²`. The
largest native error against the Richardson-extrapolated HVP was
`1.02e-9 Ha/Bohr²`; the output also passed the translation check. The input and
replay artifacts are retained locally under
`build/hessian-gpu-sm120/r2scan-orbital-f/` (`probe-pbe-water-dimer.py`,
`pbe-water-dimer-oracle/`, `coulomb-charge-binaries.sha256`,
`coulomb-charge-water-dimer-n4.log`).

RTX 5090 job `1763` passed the selected fitted algebra and public frontend
regressions, including PBE, r²SCAN+rVV10, r²SCAN-3c and fitted-range WB97M-V.
Job `1765` supplied PySCF for all eight live Fock/CPKS and DF ownership
oracles. The exact-range Fock fixture pins that provider to match its oracle;
its work receipt accepts resident preparation counts in place of repeated
integral evaluation.

See `docs/developer/dft_hessian.md` for the current contract.
The separate [spectral charge response](2026-11-21-cuda-df-spectral-charge-response.md)
now specializes the truncated pure Coulomb branch without using the full-rank tangent.
