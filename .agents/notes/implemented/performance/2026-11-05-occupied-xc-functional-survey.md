# Decision: qualify occupied-XC gains by endpoint population

Status: implemented
Date: 2026-11-05

## Problem

The recovered bounded initializer is generic across five-feature strict-FP64
meta-GGA providers, not intrinsically r2SCAN-specific. Numerical admission does
not establish an endpoint gain for every functional, spin, size or trajectory.
The follow-up must retain negative observations rather than advertising only
the cheapest occupied contraction.

## Decision

Add a reproducible multi-functional E+F runner, independent oracle generation,
offline replay and a source-bound survey capsule. Keep production science,
default selection, final original-D acceptance and all existing tolerances
unchanged. Actual absent environment controls, not explicit `1`, measure the
current default. Traces establish frame admission separately from timings.

The existing generic implementation already benefits a M06-2X RKS population:
48-atom spherical def2-SVP Standard/Direct cold E+F takes 2.727% less time in
three alternating pairs, and 3.048% less in an independent three-pair
confirmation. Both clear `max(2%, 2 * sum(relative MADs))`; all runs take
10 iterations/Focks. Do not pool the independently shifted baseline populations.
The 96-atom 1.300% difference and held warm timings do not establish gains.

MN15 48-atom default cold/displaced E+F regresses 2.351%/4.911%, accompanied by
16-to-17 and 13-to-14 Fock increases. Tight small M06-2X UKS also adds Focks
and regresses. These populations retain accurate final states under independent
and paired gates. Default-solver M06-2X RKS at 12 atoms passes common independent
gates but fails the retained tighter paired force gate; it stays unqualified.

## Invariants and rejected shortcuts

- Never equate a valid canonical frame or lower per-tile arithmetic count with
  fewer complete-endpoint seconds. Count SCF iterations and complete Focks.
- Do not retune convergence tolerances to hide an extra Fock or accept a failed
  paired force comparison. Preserve unsuccessful and neutral populations.
- Do not add a functional-name or molecule-size whitelist from these examples.
  A useful future profitability gate should reflect actual plan/initializer
  costs without adding full original-D recomputation or a production oracle.
- Do not simply broaden `feature_terms == 5` for GGA/LDA. The current native
  layout does four density work jets for tau-dependent meta-GGA, but only one
  for LDA/GGA; their projection work and scratch contracts need separate
  compiler-owned specialization and complete-endpoint qualification.
- Initial grid-3/incremental large-reference attempts did not converge. They
  cannot certify a native result. Successful MN15 full-Fock grid-5/6 references
  use the established reference controls and pass grid-stability and CPU/GPU
  spot checks; the common observable gates are unchanged.

## Evidence and consequences

The [survey capsule](../../../../benchmarks/results/occupied-xc-functionals-20261105/README.md)
retains 204 untraced complete native E+F observations plus 16 traced diagnostic
vectors. All use immutable scientific source
`ddadce1e29453e0f5cabc4c12948c7c798e8d66c` and its existing qualified native
image. DFT owner/producer bytes at delivery snapshot
`6ba9f19af11f8d7e10db28149c6530d8628b1c20` are identical. Subsequent master
admission/stopping/warm-anchor changes are not covered: this is not full
merged-target or all-functional qualification. The generic runner forbids
production CPU scientific derivative consumers and offline gates remain active
under optimized Python.

This establishes another profitable population, not an all-hybrid promotion.
The current default can still regress on admitted inputs: numerical safety and
profitability are distinct. The explicit matrix rollback remains necessary.
Revisit admission/strict-handoff scheduling when a cheap non-oracle cost model
or initializer policy can preserve the M06-2X/r2SCAN wins while removing the
measured MN15/UKS extra work. Changing method-name guards alone would not solve
the observed within-functional variation.
