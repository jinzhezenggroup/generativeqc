# Decision: bounded refinement validates the fixed native ECP Hessian

Status: implemented
Date: 2026-11-04 UTC

## Retained failure and diagnosis

PR #2993's first unscaled CsH unit Cartesian column genuinely failed the
original 160-radial/32-polar versus 224-radial/44-polar ECP Hessian check. The
unchanged 2e-8 source tolerance rejected a nonlocal difference of 3.5455244e-7;
the local difference was only 1.25e-14. The earlier failed experiment and
reference-only matrix are retained, not relabeled as accepted native evidence.

Fixed native density/response inputs isolate angular under-resolution of the
coarse check. At polar=44, radial 160/224/320 agree within 4.6e-14. The native
224x44 result differs from a separate denser 320x60 contraction by
8.77817e-10 for that original unit column. These comparisons alone are not an
independent molecular oracle.

## Decision and invariants

Keep the native 224x44 ECP matrix, frozen-Fock derivative, and returned HVP
unchanged. Keep the original two-grid fast admission. Only a finite mismatch
permits one 320x60 validation contraction at the same fixed 2e-8 tolerance.
Accept only if it agrees with the native fine result; return the original
224x44 values, never the validation values. This does not mix a new quadrature
Hamiltonian into a response solved for the original native operator.

Reject nonfinite values, overflowed differences, unresolved refinement, and
resource exhaustion. Local/nonlocal source errors cannot cancel. The existing
layer budget remains enforced for the larger validation grid; no unvalidated
value is returned after failure. Record the original comparison error, any
validation error, and attempted refinement count separately.

This is bounded staged validation of a fixed model, not an adaptive production
quadrature prescription or a rigorous global error estimator. Global native
matrix/force and CUDA grid constants remain unchanged. No displaced SCFs,
reference fallback, unit-vector rescaling, or posthoc symmetrization enters
production. Already accepted actions use the original integration work.

## Independent CPU qualification

`benchmarks/results/scalar-ecp-hessian-2993/qualification.json.xz` preserves the
original failing contractions, diagnostic grids, all six native unit columns,
three raw independent matrices, source identities, work observations, and
scope limits. Inputs remain the canonical neutral singlet CsH physical
Gaussian/VWN-RPA B3LYP/real-spherical def2-SVP/ECP46 fixture and explicit
16x6x12-per-atom moving grid.

All six actual unscaled native columns pass independently reconverged PySCF
analytic-gradient differences at steps 1e-3, 5e-4, and 3e-4. Maximum matrix
errors are 8.79755e-7, 2.20320e-7, and 7.95507e-8, below the retained 1e-5 gate.
The unsymmetrized native matrix has asymmetry 5.97890e-14 and translation
residual 1.66533e-15. Two separately evaluated native actions agree with that
matrix; their bilinear mismatch is 3.82374e-15. Unit-column response residuals are below
1.33e-12. There is no inferred, rescaled, or symmetrized native column.

The serial native qualification takes 2235.88 seconds with peak RSS 290828 KiB.
Its native library SHA256 is
`2541fc274add7a35c8da058f7c87e08584a6b0f580efc4887b1c1491a49a7a88`:
75d68784 plus the reviewed CPU UKS closure fix. CPU ECP sources match the
inherited parent1192; the remaining native diff contains CUDA files and a
helper not called by CPU execution. This is source-tree CPU evidence with an
explicitly reused library, not an exact integrated-head build or installed
artifact claim. The exact published head still requires normal repository CI.

A concurrent qualification attempt was killed with exit137. A separate
parallel regression attempt had nine compiler-killed errors before the ECP
HVP and 19 passes. These resource interruptions are retained. The successful
serial run uses matching reference matrices computed in a separate process.
The final regression replay reuses a hash-verified existing FactorizedERI
compiler artifact without changing compiler options, equations, or tolerances.

The final serial regression replay passes all 63 tests in 1483.42 seconds,
including the original direction and permanent unscaled Cs unit-x regression.
Their complete native endpoint observations are 451.53 and 661.68 seconds;
these include separate cold runtime-artifact preparation and are not directly
comparable to warm-action timings above. The original direction retains zero
refinement attempts and 2,590,784 observed AO points. Ruff lint and formatting
checks pass. The new controls cover finite/nonfinite and overflow rejection,
separate-channel errors, immutable native output, tolerance equality, budget
failure, and successful reuse of the same owner after failure.

## Work and remaining scope

For a fully populated ECP center, the original HVP integration uses 1,195,008
points. A failed finite coarse check adds 2,304,000 validation points, totaling
3,499,008 (2.928 times that formerly rejected integration attempt). Maximum
angular-layer allocation estimate grows by 1.8595 times, from 57,491,456 to
106,905,600 bytes for the 29-AO fixture. These are integration-work estimates,
not complete-endpoint speed claims. The existing-success path adds no grid
work, and its output values are unchanged.

The public B3LYP second-order registry remains closed. Rotation identity,
I/mixed Ni-ECP Hessians, UKS candidate admission, default grids, GPU execution,
installed public endpoints, total D3 curvature, stationary modes, and
25-40-atom workflows remain unqualified. Retained mixed-domain force failures
are not waived by this CsH Hessian result.

Integration retention (2026-11-06): the qualification record is losslessly XZ-compressed; decompression restores all 18,558 original bytes, SHA-256 `6997aa2353de6e0315f9bbb38ab2d3abe72caa1993568f9071f72beecae7cc7a`. No measurement, failed gate or provenance field changed. This preserves the unchanged aggregate evidence budget after master integration.
