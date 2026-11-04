# Decision: reuse coupled spin response for full-range qualification candidates

Status: implemented, low-level qualification only
Date: 2026-11-04

## Problem

Issue #2844 needs physical polarized global-hybrid response, not a restricted
surrogate or a second named-functional solver. The existing composite owner
already binds independent alpha/beta occupied spaces, same-spin primitive
exchange, coupled XC response, and spin-resolved overlap/Pulay relaxation for
WB97M-V. Its additional full-range candidate admission unnecessarily required
one density block.

## Decision and invariants

Remove only the spin-count restriction from the CPU, direct, all-electron
full-range candidate admission. Reuse `CompositeUKSResponse`, its native state
authentication, physical MethodIR, generic source planner, and existing
integral/grid/response owners without changing their mathematics or policies.

The public capability registry is unchanged. Candidate admission is not an
assertion that every graph, representation, grid, spin state, or resource budget
has passed molecular qualification. Missing primitives, mismatched Fock
provenance, invalid native states, and numerical/response failures still reject.
No reference engine or displaced SCF enters production derivative evaluation.

## Independent evidence

`tests/python/test_global_hybrid_uks_hessian.py` uses a non-collinear neutral H3
doublet with both occupied spin spaces active, physical Gaussian/VWN-RPA B3LYP
under an unrelated identifier, STO-3G, and an explicit 8x4x8 grid per atom.
Both Cartesian and spherical s-shell representations are checked; this is
deliberately not evidence for a genuine spherical d/f space.

Independent PySCF SCFs receive the identical basis and moving-grid quadrature,
not native initial densities. Tests match energy, overlap, both spin densities,
electron counts, gradient/force sign, and both same/cross-spin Fock actions.
Every raw matrix column is compared to reconverged independent analytic
moving-grid gradients at 1e-3, 5e-4, and 3e-4 Bohr, with an unchanged maximum
error gate of 1e-5 Eh/Bohr². Observed errors are 1.99315e-6, 4.75365e-7, and
1.78694e-7 in each representation. Raw symmetry, translation, H@v, and bilinear
gates remain 2e-8, before any presentation cleanup.

The eight source identities and their sum are checked. One HVP uses one
response solve (four iterations; residual 1.04894e-17) and zero production
displaced SCFs; the raw nine-column matrix uses nine response solves.
Wrong charge/multiplicity, changed-geometry generation, closed owners, grid,
output and response-workspace budgets, and exhausted response iteration limits
reject without preventing a subsequent valid action.

The six-case warm-cache source-checkout run passed in 68.18 seconds; its entire
process, including independent references, took 69.45 seconds and peaked at
609808 KiB. These are test-process measurements, not an installed complete
SCF+response+assembly benchmark or a speedup claim. Receipts are retained in
`/tmp/gqc-2843-2849/uks-low-level-v2.{log,xml}` and matching cache statistics.

The existing full-range RKS molecular and WB97M-V UKS suites additionally
passed all 17 cases in 171.26 seconds (`uks-molecular-regression.{log,xml}`).
Compiler structure validation checked 586 modules with zero dependency errors.

## Retained failures and rejected alternatives

- The initial 3e-3-Bohr unit-column independent difference had maximum error
  1.80523e-5, above the same 1e-5 gate. Its receipt remains
  `uks-low-level.{log,xml}`. The three smaller-step differences exhibit the
  expected convergence; no numerical tolerance was increased.
- An exploratory PBE0 doublet response failed closed at two explicit-grid
  points whose beta density rounded to zero while its gradient tangent
  retained subnormal residues. The corrected diagnostic is
  `uks-pbe0-response-diagnosis-v2.log`; the initial failed run remains
  `uks-spin-blocks.{log,xml}`. Native host/CUDA tests explicitly require
  nonzero empty-spin tangents, including subnormal ones, to reject. Do not
  clip these tangents, replace the point provider, or weaken that policy just
  to admit this fixture. This change does not qualify PBE0 UKS molecular work.

## Remaining gates and revisit conditions

Issue #2844 remains open: independently matched real-spherical def2-SVP Ni
doublets, nearly degenerate/unstable response cases, finite rotation covariance,
default-grid/resource admission, and installed public full-Hessian/HVP endpoint
receipts are not supplied here. The #2224 and #2843 prerequisite qualification
is not inferred from this light-element slice. D3, ECP, TS and frequencies keep
their own source and endpoint gates. Public promotion requires that missing
evidence rather than a broadened method selector.
