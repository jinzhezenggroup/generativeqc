# Decision: exact raw-feature kinetic predicates for unchanged work inputs

Status: proposed (scoped point qualification complete; no SCF/force promotion)
Date: 2026-10-12

## Problem

The physical-coordinate correction in #2279 removes a fractional-density
round trip, but separately rounded spin sigma products can still lose the
physical tie. Packed inputs `[.1,.4,.2,0,0,.8,0,0,.05,.2]` and twice that vector
are exact rational rank-one ties of the original binary64 Cartesian inputs.
Rounded total sigma exceeds its cap by one ulp. Both native half-vtau values
are zero, while independent legal inward energy derivatives are approximately
`-.01296996635` and `-.0200774970121`. Libxc also returns zero for the doubled
case, so equality to its raw derivative is not an adequate oracle.

## Decision

Bind only the canonical total-sigma/cap predicates in physical-spin point
lowering. Bind every energy and already-differentiated derivative occurrence
consistently, before optional equal-spin sharing. Retain branch payloads, work
floors, FHC and cross-spin clamps, and the original normal-product guard.
An input whose work coordinates change keeps the incumbent work comparison;
this change does not invent a convention for empty/floored-spin coordinates.

For unchanged work inputs, classify the exact original binary64 quadratic
`sum_axis((gradient_a+gradient_b)**2) <= 8*(rho_a+rho_b)*(tau_a+tau_b)`.
Outward nextafter intervals certify strictly separated sides. Overlap invokes
a fixed 4352-bit signed accumulator using 13 exact products, each expanded
into four 32-bit limbs. frexp extracts integer significands without host-only
128-bit arithmetic. Binary64 significand exponents range from -1126 to 971;
the accumulator's unit is 2^-2252 and its high end exceeds every finite-input
sum, with spare sign/carry space. Negative cross products and subnormals are
included. No tolerance, dynamic allocation or runtime reference is involved.

Compiler-owned emitted predicate bytes enter both backends' semantic identity.
Unrelated registrations retain their previous point mathematics. Provenance
identities continue to follow the complete compiler source closure; even a
documentation-only compiler edit can legitimately invalidate that provenance.
Scalar-feature bulk programs retain their existing work-coordinate contract;
only physical-spin lowering has the original Cartesian inputs needed here.

## Rejected Alternatives

- Another epsilon around a rounded sigma cap admits actual exterior points.
- Simply summing gradients first can still lose a multi-axis tie in its norm.
- Changing the whole normalized ratio perturbs retained empty-spin payloads.
- Copying raw Libxc derivatives fails the doubled-vector false-negative control.
- Quietly updating historical source fingerprints is not a scientific oracle.
  The old full-shard migration audit is explicitly opt-in; existing independent
  compiled point tests and new rational/energy-FD controls remain required.

## Evidence and Boundaries

The exact comparison passes deterministic independent Fraction checks across
512 random finite binary64 vectors, 41 scale-distributed exact ties, adjacent
raw branch sides, cancellation, subnormal squares and overflow-range products.
Compiled point tests pass the two decimal inward energy stencils against
pinned PySCF 2.14.0 / Libxc 7.0.0 energy, with no raw-vxc equality requirement.
The original dyadic/full-coefficient and empty/floored-spin gates still pass.
Initial local attempts, including a missed n-ary predicate match, are retained
under `.cache/1118-runtime/physical-kinetic*`; they are not passing evidence.

These point controls do not qualify every molecular empty-spin inward path,
the retained four raw-Libxc molecular failures, CUDA KS endpoints or forces.
The #2276 numerical/merge hold remains until those open directions are resolved.
The master-based independent phase passes 243 host cases, with its historical
source-migration fingerprint audit explicitly skipped. Slurm 7642 on node1/main,
one RTX 5090, finite ten-minute allocation and scheduler visibility 1 passes ten
CPU/CUDA point controls, including both-spin pinned E/vxc and invalid-input
controls plus the two decimal inward derivatives on both backends. Repeating
the same controls under memcheck passes with zero errors. These are repeated
controls, not twenty independent qualifications. Exact phase source manifest
SHA256 is `39b81534f1b1572fec80c6db9da866be3a7f7b0c41c1ca87cb78b745d208525d`.
The actual sm_120 point probe uses 255 registers and a 720-byte stack frame,
with 192/208 bytes of spill stores/loads. These are complete point-probe compiler
resources, not isolated-comparator or complete-endpoint performance results.
No endpoint speedup or default promotion is claimed. The final documentation
refresh must retain byte-identical emitted point sources and is qualified
separately; earlier source identities are not relabeled.

## Revisit When

Transformed work-coordinate consumers acquire an independently validated
physical-input boundary contract, or a lower-storage exact comparator preserves
the full exponent-range, adjacent-side and inward-derivative gates.

## References

- #1118, #2276 and #2279.
- [Earlier work-coordinate predicate correction](../implemented/numerics/2026-10-11-physical-kinetic-predicates.md).
- `tests/python/test_exact_physical_kinetic_comparison.py`.
- `tests/python/test_libxc_physical_kinetic_predicate.py`.
- `tests/python/test_physical_kinetic_point_cuda.py`.
