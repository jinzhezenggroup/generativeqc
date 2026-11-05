# Decision: condition restricted stationary response with physical orbital gaps

Status: implemented
Date: 2026-11-04

## Problem

The explicit-grid spherical NiH4(2-) def2-SVP response probe reaches the shared
CPKS operator, but unpreconditioned default GMRES exhausts 200 iterations at a
true residual of 1.261e-7. Increasing restart to 100 and maximum iterations to
400 converges after 257 iterations / 514 actions at 1.051386150187735e-11,
but does not resolve the subsequent second-integral compilation bottleneck.
Neither run produces a complete Ni HVP. Increasing the default iteration and
workspace caps would conceal poor conditioning rather than address it.

## Decision

Reuse `DiagonalPreconditioner` in the existing bounded solver. A method-neutral
factory computes signed virtual-minus-occupied gaps in the actual restricted
`RotationLayout`, with occupied-major/virtual-minor packing. The CPU restricted
composite stationary owner requests this scaling explicitly. It changes neither
the response operator nor RHS, tolerance, true-residual gate, physical AO
representation, nor source composition. UKS and resident CUDA policies are
unchanged. Operator and preconditioner storage retain their existing independent
budget ownership; the GMRES workspace is not a complete endpoint memory bound.

Nonfinite/nonreal/overflowed inputs fail before publication. Zero/near-zero gaps
select the ordinary bounded unpreconditioned solve; no gap is clamped or made
positive. Generic `solve` still has no automatic preconditioner. HVP diagnostics
identify orbital-gap, singular-gap fallback, or unchanged owner policy.

## Evidence and limits

The response-only local Ni experiment on parent 0434fe03e used the same physical
operator, RHS and default restart30/max200/atol1e-11/rtol1e-10. Injecting the
existing diagonal preconditioner converged in 17 iterations / 34 actions at a
true residual of 8.110805132056192e-12, about 29.54 s for the response phase.
The independent PySCF 2.14.0 reference matched the accepted native state before
this experiment; the fixed 5760-point grid is not default-grid qualification.

Local receipts are `/tmp/gqc-2843-2849/ni-hvp-symmetry.log`,
`ni-hvp-large-restart-v2.log`, and `ni-preconditioned-response.log`.
The restart100 attempt terminated with signal11 in later one-electron
second-source assembly; its cause is not established. The isolated second-source
probe instead exceeded the finite 300 s compiler deadline on a d/f nuclear
attraction tile. Response convergence is not a complete HVP, installed/public
endpoint qualification, or an endpoint speedup claim.

Executable tests compare the conditioned and ordinary solutions with an
independent dense solve, preserve explicit packing and signed gaps, reject
invalid inputs, and check the unchanged true residual. These and the shared
response/weight regressions pass 66 cases in 2.40 s. All eight independent
B3LYP/PBE0 canonical/custom complete curvature/raw-matrix cases pass in 157.99 s
(eight JUnit warnings). The full WB97M-V direct/full-rank-DF RKS/UKS Hessian
files pass 39 cases in 359.62 s. The molecular runs reuse an explicitly retained
JIT cache through a local-only fixture; source and artifact identities remain
authenticated. They are warm-cache scientific regression, not cold compilation
or installed Ni endpoint receipts. All applicable staged hooks pass.

## Revisit

Revisit stronger shared conditioning only with complete endpoint evidence and
the same physical residual gates. Do not loosen tolerances, enlarge all caps,
regularize physical denominators, or add a B3LYP-specific solver to obtain a pass.
