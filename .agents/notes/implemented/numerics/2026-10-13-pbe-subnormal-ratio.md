# Decision: share accurate subnormal PBE ratios and directions

Status: implemented
Date: 2026-10-13

## Problem

Ordinary full-spin and equal-spin PBE exchange divided a nonzero gradient by
`rho*cbrt(rho)` even when that product was subnormal. Quantization of the
denominator corrupted otherwise representable potentials. The separate bound
PBE0 consumer already factored that division, but selecting its fixed 0.75/1
weights cannot repair arbitrary PBE/MethodIR consumers. This defect predates
the opt-in point-family work in #2250: its head `61e2c3e4` and current master
`f62734fb` share the same original numerical-header blobs.

Factoring the value alone is insufficient for signed response. The ordinary
jet quotient rule subtracts directional terms before division; that numerator
can itself be subnormal. A response may then be wrong even after its physical
point value is corrected.

## Decision and invariants

Use the same `(gradient/rho)/cbrt(rho)` expression in every consumer only when
`rho*cbrt(rho) < DBL_MIN`. In that branch, differentiate the factored ratio by
dividing directional terms before subtraction. Multiply the two relative
density terms separately, since their sum can overflow even when the final
directional coefficient is finite. Remove the private restricted-only switch.

Retain the ordinary normal-denominator operation order, the reciprocal
large-gradient branch, X/C weights, zero-gradient Hessian, empty-spin and
interior admission, correlation spin extension, and all default policies.
There is no density floor, clipping, fixed-weight substitution, or whole-XC
invariant rewrite. No response capability or device qualification is added.

## Evidence

With GCC 14.2, verified ccache 4.14.1, strict FP64 host compilation and existing
reference dependencies, the three focused Python modules pass **106 tests**:
`test_pbe_subnormal_ratio.py`, `test_scaled_xc_point_response.py`, and
`test_stationary_restricted_point.py`.

- All 87 frozen PBE0 physical points pass for full-spin, equal-spin and bound
  consumers. The emitted ordinary consumer now passes the same 87-point gate;
  its historical six-row expected-failure list is replaced by positive checks.
- All 97 frozen SCF points pass, including 24 independent Libxc and 73 original
  high-precision references. Both physical fixture gates remain exactly
  `5e-10*abs(reference)+1e-322`, with no dropped inputs.
- The existing 30 RKS and 48 UKS independent response directions retain their
  original component-conditioned gates. Twenty weighted Libxc response cases,
  signed finite differences, interior/empty/minority-spin, zero direction,
  zero gradient and extreme-finite controls also pass.
- Twelve additional weighted RKS/UKS tail directions and two zero-gradient /
  extreme finite exchange directions agree with the independent original
  energy differentiated at 450 and 550 digits; the two precisions round to
  identical FP64 references. Signed responses are distinct from physical E/V
  qualification and from same-consumer finite-difference consistency.
- Original source fails six physical rows in each ordinary spin layout and
  48 of 72 newly weighted physical vectors. Restoring only the old jet
  quotient arithmetic passes physical points but fails 16 signed-response
  vectors. Final source passes all of these controls.
- All 97 SCF outputs and 1,000 additional normal-denominator physical outputs
  are byte-identical before/after. Another 1,000 signed-response outputs and
  admission decisions are byte-identical, including 41 unchanged rejections.

### Newly extended PBE cancellation probes

The new arbitrary-weight corpus uses a separately declared roundoff allowance
`32*epsilon*(abs(weighted_X)+abs(weighted_C))` in addition to its relative term.
The independent original formula supplies both component magnitudes. This
follows the existing response-corpus error model: rounding in the component
evaluation and weighted sum scales with the terms, not their canceled result.
It is an explicit tested FP64 allowance, not a formal libm error bound. The
original 87/97 gates above are untouched.

For pure PBE at row 65, X=-3.321338964678348e-6 and
C=3.321338971963299e-6 leave 7.284950757499163e-15, with condition number
9.12e8. Observed absolute error 2.72e-22 is below
epsilon*(abs(X)+abs(C))=1.47e-21. At row 66 the y terms are approximately
+9.964016282099209e-6 and -9.964016915889875e-6; their result is
-6.337906656321792e-13, with error 2.43e-21 below the 4.42e-21 component scale.
A pure relative 5e-10 gate on those nearly canceled totals is not an attainable
general FP64 claim. These newly designed stress-test limits are distinct from
the inherited quantized-ratio and signed-quotient defects; neither motivates
a broader cancellation rewrite.

## Limits and reproduction

Run the three modules above with the `reference-test` dependencies and an
available verified compiler cache. The new module records input directions,
450/550-digit references, X/C contributions and measured outputs in JUnit
properties when requested. This is source-matched CPU/header and emitted-host
qualification, not a full native-library build, CUDA execution, complete E+F
endpoint campaign, performance measurement, or #2250 promotion approval.
