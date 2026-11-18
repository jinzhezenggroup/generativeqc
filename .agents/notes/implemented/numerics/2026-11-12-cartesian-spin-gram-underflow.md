# Decision: scale Cartesian gradients for Gram admission, not XC evaluation

Status: implemented (native Cartesian GGA point admission)
Date: 2026-11-12

## Diagnosis

The new composed nuclear-response control uncovered a cold native UKS failure
on an actual Na-like synthetic residual-ECP/H/H fixture. The original point
guard reported these raw sigma values and Cauchy bound in hex FP64:

```text
sigma_aa = 0x0.1aa5d6cf1af84p-1022
sigma_ab = 0x0.0372cdd6f43e5p-1022
sigma_bb = 0x0.00723e0b6c8c3p-1022
bound    = 0x0.0372cdd6f43e3p-1022
```

All inputs were finite Cartesian gradients; their exact Gram matrix is
necessarily positive semidefinite. Two minimum-subnormal ulps in a rounded dot
product exceed a relative-only 16-epsilon allowance at this scale. Rejecting
the state is not evidence of unphysical chemistry or invalid spin response.

## Correction and invariants

For admission only, divide each Cartesian gradient by its own largest absolute
component and reuse the same generated sigma arithmetic. The nonzero row's
squared norm is then between one and three, so the same 16-epsilon Cauchy
allowance is meaningful. Exactly zero rows remain zero. Keep all original
density, same-spin sigma, finiteness and representability checks. Preserve the
original UNscaled sigma passed to generated XC energy/potential arithmetic.
No clipping, density/sigma floor, functional change, reference-engine fallback
or relaxed response residual is introduced.

Apply this shared validator to the Cartesian interior and B3LYP production
point checks. A free-standing sigma-feature API must still validate its own
independent physical domain; this reasoning depends on the actual finite
Cartesian gradients being present.

## Evidence and limits

`test_subnormal_cartesian_gram_against_independent_point` uses parallel
gradients `[1,.3,-.2]*1e-154` and 0.13 times that row. Their rounded raw sigma
fails the old guard, despite representing physical vectors. Compare actual
native B3LYP energy, density potentials and all six Cartesian gradient
potentials to independent Libxc at the same input. Invalid density/nonfinite
inputs must still fail without publishing output. The composed nuclear test
retains the actual UKS/ECP fixture rather than skipping or changing it.

This fixes an input-admission false positive. It does not qualify a complete
Hessian, representative ECP endpoint, molecular saddle or final frequencies.

## Rejected alternatives

- Increase the relative allowance: no fixed relative tolerance handles all
  subnormal dot products, and this would relax normal-scale admission too.
- Clip sigma_ab or floor the same-spin sigma: changes the fields consumed by XC.
- Remove the failing UKS/ECP control or shrink its residuals: hides the actual
  cold production failure instead of correcting its numerical cause.

## Revisit when

A new point contract accepts raw sigma features without Cartesian gradients,
or the generated dot-product precision changes. Preserve independent point
potentials, invalid-input refusal and the actual cold-SCF regression.
