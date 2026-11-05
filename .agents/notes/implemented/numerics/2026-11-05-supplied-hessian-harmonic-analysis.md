# Decision: separate supplied-Hessian analysis from TS workflow qualification

Status: implemented
Date: 2026-11-05

## Problem and decision

Issue #2849 explicitly permits fixed-Hessian normal-mode analysis before its
electronic derivative and TS-task dependencies are complete. Implement that
independently useful consumer, but do not turn an array receipt into a native
electronic or optimization success receipt. The public host analysis performs
no SCF, external electronic-engine work, finite differences or optimization.
Its diagnostics always state that scientific state and stationarity were not
verified; no existing calculator second-order capability is promoted.

Require explicit canonical units and complete atom-major shapes. Caller-supplied
positive physical atomic/isotopic masses are independent of ECP core counts.
Report raw matrix asymmetry and rigid residuals before any mass weighting or
projection. Reject asymmetry above the declared gate; even admitted nonzero
asymmetry requires explicit averaging permission. Never silently rescue a
truncated or bad matrix through projection or symmetrization.

Use mass-weighted translations and center-of-mass rotations, SVD rotation rank
and an orthogonal complement for the vibrational eigenproblem. Rotation rank
has a declared `1e-8` relative threshold with singular values/ratio retained.
Keep distinct linear, nonlinear and monatomic counts; fail coincident-only
multiatom geometry. Use total Cartesian input identity, but document that it
cannot establish matrix/geometry/method consistency without an upstream owner.

Preserve signed unscaled frequencies and all near-zero modes. The `1 cm^-1`
soft band changes only the separately reported significant-imaginary count,
not eigenvalues, negative-mode count or output length. Cartesian modes have
unit physical mass norm; mass-weighted modes have Euclidean unit norm. Keep
degenerate-mode comparisons subspace-based and visualization scaling separate
from physical normalization. Bound dense input dimension before converting H.

## Units and independent gates

SI conversion is `sqrt(Eh/(u*a0^2))/(2*pi*c*100)` times the signed square root
of an eigenvalue in `Eh/(Bohr^2 dalton)`. Pinned 2022 CODATA values are
`Eh=4.3597447222060e-18 J`, `a0=5.29177210544e-11 m`,
`u=1.66053906892e-27 kg`, and exact `c=299792458 m/s`.
The primary NIST complete table was retrieved from
`https://physics.nist.gov/cuu/Constants/Table/allascii.txt`, retained locally as
`codata-2022-constants.txt`, SHA-256
`77fb90e66c40db3e6eb16630bc9c88e4c7c8beddbe5e71be406f2f26e3f67e67`.

Before molecular endpoint qualification, fixed-Hessian tests freeze ordinary
mode comparisons at `1e-7 cm^-1`, stricter than the proposed molecular gates.
Independent references are analytic reduced-mass/bending/stretching models and
SciPy generalized eigenproblems/physical constants, not the production normal
mode function. Cover physical Cs/isotope masses, linear/nonlinear shapes,
one-negative-curvature saddle and reaction-mode overlap, soft signed modes,
rotation/translation covariance, normalization, raw integrity, unit errors,
immutability and XYZ atom ordering. These are mathematical input-analysis
tests, not B3LYP/Ni/ECP Hessian or TS evidence.

## Accepted source-tree evidence

- The 18 new fixed-Hessian cases pass. With existing accuracy, force-aware
  numerics and public-API contract regressions, the final run reports
  **112 passed plus 17 subtests**, 13.29 s; complete process 14.32 s and
  peak 424,280 KiB, including native HF/reference regression work.
- Recorded independent frequency errors are at most `1.365e-12 cm^-1`;
  independent mass-normalized reaction-mode overlap is
  `1.0000000000000004` (floating-point roundoff about one). SciPy 1.18.0 is
  test-only; production analysis uses NumPy, here 2.5.1.
- The warnings-as-errors Sphinx build succeeds; public export/contract checks
  include the result record, host analysis and XYZ exporter. No real GPU work.
- Retain `vibrations-complete-v4.{log,xml}`, source digests and Sphinx logs
  under `/tmp/gqc-2843-2849/`. An earlier optional native regression command
  omitted `GENERATIVEQC_LIBRARY` and failed six tests before native loading;
  it is retained as `vibrations-regression-v1.log`, not a scientific failure
  or passing receipt. The final run explicitly uses library SHA-256
  `e3a29ef62d72b70a2251721321a7dbf7484e2b3649f636d0cdc10aac016b2dc6`.

These are source-tree analysis/regression receipts, not installed-artifact
qualification or a complete molecular endpoint timing.

## Open integration boundary

No TSopt+freq API/CLI task, final-geometry/state/MethodIR identity handoff,
installed electronic receipt, D3 exactly-once composition, Ni/ECP stationary
fixture or complete molecular endpoint is delivered by this analysis consumer.
Stale electronic Hessian detection belongs to that future upstream handoff;
this standalone function cannot infer provenance from numerical arrays.
Do not close #2849 or infer a validated first-order saddle from this output.

## References

- Issue #2849 and its derivative/task dependencies #2843 through #2848.
- `python/generativeqc/vibrations.py` and `tests/python/test_vibrations.py`.
- `docs/user/vibrations.md` for current public behavior.
