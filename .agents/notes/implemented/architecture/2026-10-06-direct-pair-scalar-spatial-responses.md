# Decision: one scalar Coulomb preparation for all dddd atom responses

Status: implemented
Date: 2026-10-06

## Problem

The first stationary dddd consumer shares a Dual3 recurrence across all
components and J'/K' channels, but repeats Coulomb preparation for every
independent atom. A four-atom, two-primitive quartet therefore prepares 48 times.

## Decision

Use the existing Coulomb IR's spatial-derivative definition: differentiating
`R^n_tuv` along an axis requests its next Cartesian state. One scalar order-9
simplex supplies both the order-8 primal component and every first response.
The view multiplies each raised state by the retained Gaussian product-center
response to the current atom. Pair Hermite derivatives continue to use Dual3;
all independent bra responses survive the complete ket loop.

The authoritative Cartesian contraction body now accepts a state view. Its
typed compatibility entry retains all existing scalar/AD callers and the exact
loop/prefactor order. The new force schedule retains each response's primitive
traversal and individual coefficient order, bounded component sums, independent
density/source gates, repeated atoms and translation recovery. Storage stays
below 32 KiB, with no global derivative tensor or production counters.

Scope stays full-range dddd, with the same default-off frozen selection and
cache/layout/recurrence fallbacks as #2003. This does not raise the public basis
limit or admit through-f order-13 scalar storage.

## Rejected alternatives

- A second raised/lowered basis recurrence: the existing pair AD and Coulomb
  derivative IR already supply both parts of the geometry chain rule.
- Recomputing bra responses inside every ket product: three bounded bra tables
  fit the existing storage allowance and preserve primitive-pair reuse.
- Scattering each primitive separately: bounded per-component derivative sums
  retain the previous accumulation order and avoid extra atomics.

## Evidence and acceptance

The IR AD gate checks all 495 axis responses of the degree-0--8 prefix, using
the existing Boys leaf derivative rule and independent symbolic differentiation.
Finite n1 Slurm job 6177 passed the complete retained value controls and new
scalar derivative controls in 30.59 seconds. Native work gates require 16
Coulomb preparations for 1,296 four-atom/two-primitive components and both
sources, instead of 48. Raw Dual3 components and independent host orbit forces
agree for RHF/UHF, each source channel, repeated atoms and same-pair domains;
inactive, screened and zero-density domains prepare nothing.

The exact release library, independent Libcint responses, sanitizer lifetimes
and complete endpoint performance remain qualification gates. No default
promotion or endpoint improvement follows from this work-count reduction.

## References

- Issue #1892; stationary-source foundation #2003.
- Supersedes `2026-10-06-direct-pair-materialized-full-source-derivatives.md`.
- `docs/developer/direct_pair_recurrence.md`.
