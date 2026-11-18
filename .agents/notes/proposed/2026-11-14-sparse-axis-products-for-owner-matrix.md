# Proposal: qualify sparse Cartesian-axis products for physical-owner matrices

Status: proposed; no candidate compiled and no production implementation
Date: 2026-11-14

## Actual consumer diagnosis

After the separately qualified gradient-only first-source change, the remaining
frozen-integral matrix source still needs its own diagnosis. Do not infer its
cost from first gradients or restart an unchanged molecular endpoint.

`build/qualification/host-frozen-matrix-prefix-20261114` calls the actual
ProjectedMatrix/SymmetryUnique consumer with original saved native C->D,
six atoms/81 AOs, exact linked method/strategy and immutable validators. W is
explicitly zero: the saved state has C/occupancy but no orbital energies/W.
This isolates an ERI component, not the complete physical frozen/stationary H.
No reference SCF, XC, residual ECP, D3, response or Hessian substitution occurs.

It completes 8,000,000 primitive calls then interrupts before the next leaf;
native counters correctly include 8,000,001 attempts. The original combined
first/frozen admission is 882,874,977 ERI visits under the unchanged 1e9 cap;
the frozen plan is 645,136,023 ERI, 9,548,100 one-electron primitives,
656,462,280 map terms, 188,096 numeric bytes. The diagnostic separately bounds
capture storage to 6,942,720 bytes. The output matrix remains entirely canaries.

Complete interrupted apply is 25.329919118 s; successful primitive leaf scopes
total 10.735467950 s (42.38%). The remainder includes the complete one-electron
prefix, density congruence, maps/scatter, and wrapper work; it is NOT proven
to be entirely S/T/V. Actual one-electron work is 9,548,100, map terms26,496,888,
and density-congruence contributions9,216.

The prefix has 2,858,190 scalar one-physical-owner calls (1.639294635 s,
25 classes) and 5,141,810 matrix calls with exactly ONE non-origin owner
(9.096173315 s,216 angular/incidence classes). Native matrix counters include
the interrupted call: 5,141,811 calls, 15,425,433 coordinate fields and
46,276,299 ordered second partials. Maximum observed angular momentum is d.
This early prefix does NOT represent later two/three-non-origin-owner work,
all Cartesian components, f, or the entire earlier599-second frozen phase.
Never linearly extrapolate it to that complete phase.

All482 actual samples pass independent analytic libcint value/gradient/Hessian
and physical-owner projection checks. Original gates remain 2e-11 abs/rel for
value, 2e-10 abs/rel for gradient, and 2e-9 abs/2e-8 rel for Hessian. Additional
norm-scaled checks use the same coefficients and a64-min-subnormal-ULP floor;
the independent oracle's exponential cutoff1000 prevents default screening.
Maximum errors are6.78e-20,4.14e-19,2.19e-19 respectively; max raw asymmetry
1.30e-26. Largest sampled norms are8.39e-5,1.56e-5,5.62e-6, respectively:
not every sample is trivially zero, but broad ordinary-scale controls remain
mandatory for a production candidate. These are primitive/consumer-prefix
checks, not a completed frozen matrix or raw molecular-H gate.

## Narrow hypothesis to test privately

The compiler's `integral/factorized_matrix.py` builds three dense Cartesian-axis
jets for each Hermite term. Each axis coefficient has derivative support on
only that axis, yet the term composes them with three generic dense products:
`multiply(multiply(multiply(axes[0],axes[1]),axes[2]),coulomb)`.
The source shape suggests avoidable known-zero work; it does NOT prove the
compiler leaves all of it in machine code. A private bounded candidate should
measure this hypothesis before a production change or speedup claim.

Have the existing scalar Graph/AD owner lower a structured product of the three
axis factors, preserving their sparse support before materializing the dense
physical-owner jet for the Coulomb product. Share existing coefficient,
Coulomb-state, Gaussian-prefactor and incidence owners. Do not author another
Hessian/product-rule algebra in native C++, copy a matrix transpose, or fake
zero scalar fields. In particular, record/coefficient nonfinite and signed-zero
behavior needs explicit controls when eliminating nominal zero products.

Require independent analytic value/gradient/ordered-H gates across every
physical-owner partition, separated/coincident-distinct owners, diffuse/tight,
translated, all supported s/p/d/f classes and nontrivial scales. Keep the fixed
gates, finite compiler/cache ownership, original primitive/partial accounting,
immutable-state validation and transactional outputs. A qualified candidate
then needs the real consumer prefix, normal scalar/first/curvature regressions,
and eventually complete endpoint evidence. A leaf win alone is insufficient.

Record-local geometry reuse is a separate possible dataflow optimization; see
the earlier implementation note. Do not silently introduce a persistent cache,
unchecked frame identity, extra resident storage or an unbounded resource path.

## Current authority and limits

The production pair remains source
`8ce495a15ae52d1b293013f577a24314373b3d2ceced384c77df67275e0530a6`, core
`e5a0e54408d57ff5a119b724819cd3e7136e4ed6545bcb45f0aa0528f7b3cb61`.
The diagnosis does not change production. Both public ECP saddle guards remain
closed. Complete H/response/unscaled modes, mixed Ni/I/Cs, stationary Ni/Cs,
25–40-atom practicality and the native Ni TS with all five criteria within
100 attempts are still missing.

## References

- [Gradient-only first-source decision](../implemented/performance/2026-11-13-gradient-only-packaged-first-source.md)
- [Physical-owner matrix lowering](../implemented/performance/2026-11-13-physical-owner-host-matrix-sources.md)
- [Earlier record-local geometry reuse](../implemented/performance/2026-11-11-factorized-host-record-geometry-reuse.md)

## Subsequent private qualification and implementation

The private hypothesis now passes independent analytic, extreme-domain,
actual-consumer and sanitizer gates and has been migrated into the generator.
The original proposal and diagnosis above are retained as historical evidence.
See [the implementation decision](../implemented/performance/2026-11-14-sparse-axis-products-for-owner-matrix.md)
for source-bound measurements and the still-pending production qualification.
Production component qualification subsequently passes; the implementation
note records the matching source/core, complete regression sets and remaining
endpoint limitations. The proposal's original status above is historical.
