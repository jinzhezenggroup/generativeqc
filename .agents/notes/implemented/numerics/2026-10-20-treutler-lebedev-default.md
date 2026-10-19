# Decision: default to Treutler/Lebedev with NWChem pruning

Status: implemented
Date: 2026-10-20

## Problem

The previous tensor-product prescriptions did not reproduce the default
GPU4PySCF/PySCF quadrature. Matching only the radial and angular orders also
misses NWChem's element-dependent shell pruning and Treutler's heteronuclear
Becke radius adjustment. Introducing this family changes grid ownership,
resource admission and derivatives as well as the quadrature values.

## Decision

All KS compositions resolve omitted grids to `GridSpec.treutler(3)`; tight
accuracy resolves to level 4. Force requests keep the same level as energy
requests. An omitted VV10 grid inherits the XC prescription. Explicit v1--v3
grids remain available with their previous semantics.

Version 4 uses the existing descriptor layout: `radial_points` carries level
0--9 and both angular fields are zero. The scientific Python descriptor names
the radial, angular, pruning and partition rules explicitly. This preserves
C and snapshot structure sizes while making accidental tensor interpretation
fail validation. Point counts require actual atomic numbers.

Pinned PySCF 2.14.0 numerical inputs cover H--Lr. The importer records upstream
file hashes and retains the Apache license and NOTICE. Production and source
generation read repository-owned tables without importing PySCF. The native
radial transform and partition adjustment derivatives use the common scalar
Graph; a second symbolic algebra is not introduced.

Each element retains its real shell sizes. Atom-major prefix offsets replace
uniform `point / points_per_atom` ownership for this family. Legitimate signed
Lebedev weights are preserved, and no zero-weight alignment points are added.
Default water has 33,698 points, compared with 33,704 including PySCF padding.

The Treutler correction is `mu + a * (1 - mu**2)`, after clipping the unadjusted
coordinate. With `r = sqrt(radius_b / radius_a)`, the fixed elemental parameter
is `a = clip((r - 1/r)/4, -1/2, 1/2)`. These radii affect the partition only;
radial nodes and NWChem pruning retain their pinned elemental inputs.

CUDA builds immutable element-local input rules on the host, then translates
points and computes molecular partitions on device. Retained quadrature owns
five point arrays plus `2*natom + 1` words for prefix offsets and radii. Shape-only
preparation bounds conservatively include unique-element input rules, uploads
and pair adjustments. Stationary owners copy topology once; mixed geometry
consumers borrow metadata under the existing authenticated grid lifetime.
Independent XC/VV10 owners retain independent topology.

Adjusted partitions currently use the bounded serial/cooperative reverse
traversal. The equal-radius phased/binary-scaled optimization is not admitted
for adjusted partitions. Its reserved storage remains part of conservative
capacity planning; diagnostics report the route actually eligible to execute.

## Rejected alternatives

- Importing PySCF during runtime or generation would violate the production
  dependency boundary and make installed behavior depend on a reference package.
- Padding every atom to the same point count would add artificial scientific
  work and hide incorrect uniform-ownership assumptions.
- Reusing unadjusted partition derivatives would give incorrect heteronuclear
  forces even when energy-only quadrature agrees.
- Changing old descriptor versions would break explicit reproducibility.
- Inventing parameters beyond Lr would lack an independent upstream source.

## Invariants and evidence

`test_grid_treutler.py` compares independent PySCF angular rules and moments,
radial transforms, pruning, and molecular weights. Coverage includes levels
0, 3 and 9, light/heavy elements through Lr, and signed low-order rules.
Generated native first responses and mixed responses are checked against
independent values/finite differences. Legacy grid tests remain in place.

`test_treutler_endpoints.py` qualifies default-grid energy and analytic forces
against independently generated PySCF grids (2e-8 Hartree energy and 3e-7
Hartree/Bohr force gates), including Direct/DF and separate VV10 prescriptions.
`test_native_grid_geometry.py` checks first/mixed moving-grid responses against
long-double finite differences with ragged ownership and signed weights.
Native quadrature tests check allocation bounds and release of retained storage.

This change makes no performance claim and does not refresh README benchmarks
or CBS results. Before publishing timings, qualify complete endpoints with the
new discrete quadrature and report their actual work counts.

## Revisit when

An adjusted phased derivative is implemented and independently qualified, an
upstream source supplies heavier-element parameters, or a public descriptor
revision warrants replacing the level encoding with a dedicated field.
