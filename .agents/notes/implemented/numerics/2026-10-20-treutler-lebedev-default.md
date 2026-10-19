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
The complete retained table content participates in grid identities and
provenance, so stale version labels cannot preserve a cache across a table change.

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
CUDA checks complete forces for PBE, PBE0, r2SCAN and WB97M-V; CPU checks
energies for all four and public forces for PBE/PBE0. Existing CPU meta-GGA
force admission is not expanded. The WB97M-V DF oracle retains exact long-range
exchange and zero long-range auxiliary response, matching the existing native
operator; PySCF's ordinary all-range DF default is a different approximation.
`test_native_grid_geometry.py` checks first/mixed moving-grid responses against
long-double finite differences with ragged ownership and signed weights.
Native quadrature tests check allocation bounds and release of retained storage.

Qualification on n1, using Slurm-assigned RTX 5090 devices, passed all 18
endpoint oracle cases, both native quadrature executables, 36 native geometry
checks, 202 API/resource regressions, and 14 CLI checks. Four separate
WB97M-V prepared-owner geometry probe tests were not enabled; the generic
moving-grid probe covered adjusted ragged/signed rules. This is not a complete
Hessian endpoint qualification. Local grid, response, ownership and packaging
checks also passed; the compiler audit checked 571 modules with no dependency
errors. The installed Python wheel payload retained the JSON and licenses and
generated native quadrature without importing a reference/runtime package.

The measured code is `57f1b7d80`, based on #2609 at `0721ef3f6`, with a CUDA
12.9 sm_120 build and the explicit development Hessian leaf inventory used
for the standalone geometry probe. Native library SHA-256:
`ec278e2d771b534a4d34ba5db1ee8bee7684e7c291f3336e3fa83d9a9ab7b381`.
Build and JIT compilers used verified sccache 0.16.0; shared-cache statistics
were retained before/after compilation. Raw qualification logs are retained
locally under `.artifacts/receipts/`.

This change makes no performance claim and does not refresh README benchmarks
or CBS results. Before publishing timings, qualify complete endpoints with the
new discrete quadrature and report their actual work counts.

## Revisit when

An adjusted phased derivative is implemented and independently qualified, an
upstream source supplies heavier-element parameters, or a public descriptor
revision warrants replacing the level encoding with a dedicated field.

## 2026-10-19 review addendum: independent reference integration

The earlier qualification above is retained as originally scoped. The old
`c99a43fb8` CI run exposed four additional energy-oracle failures, reproduced
against the exact `33db6b018` source composition on a freshly built CPU core.
These were reference-setup/branch-selection defects; no production numerical
formula or acceptance tolerance was changed to repair them.

The DF force oracle assigned explicit coordinates/weights before changing
PySCF's `radii_adjust`. PySCF 2.14.0 invalidates those arrays on that assignment,
so it rebuilt an unadjusted partition instead of the native v4 prescription.
The oracle now sets the sourced radii/adjustment first, then installs the exact
quadrature and checks that SCF retained both arrays. Replaying the old order
still fails the 1e-8 Hartree energy gate (PBE: 8.07e-7; r2SCAN: 2.55e-6).
Correcting the order while omitting the v4 adjustment still fails the unchanged
3e-7 Hartree/Bohr force gate (PBE: 1.25e-5; r2SCAN: 5.24e-5).

Axial OH has continuum-degenerate pi orientations, but finite Lebedev angular
quadrature separates the axis and diagonal C2v stationary branches. The original
axis-only PBE reference differs from native by 2.098e-7 Hartree for STO-3G and
1.012e-7 Hartree for def2-SVP. Independently enumerating the predetermined 0 and
45 degree subgroups finds the diagonal branch within 1.14e-13 Hartree of native
for both bases. Both branches and their original differences remain evidence;
ordinary SCF convergence does not promise a lowest or unique root. Stability is
a separate public workflow, as documented in the SCF proposal/root contracts.

The repaired comparison converges independent `minao` and `1e` starts for each
predetermined branch, requires their agreement within each branch, retains the
full unrestricted AO residual gate of 1e-9, and requires native energy to match
one complete branch within 1e-8 Hartree. Neither reference start uses native
density/orbitals. A separate post-convergence PySCF E[D]/commutator check validates
the native PBE density after independently checked Cartesian normalization.
Off-branch energy and perturbed-density controls remain rejected.

Local CPU-only validation passed 35 tests across the two affected SCF/DF modules;
one existing CUDA-only allocation-budget case was skipped for its CPU parameter.
Both OH basis sizes and LDA/PBE controls, DF PBE/r2SCAN complete forces, translation
and warm replay, and explicit legacy global-hybrid controls passed. The four
negative DF controls above fail as intended. This addendum supplies no new GPU,
performance, whole-Hessian, or global-minimum qualification and does not replace
the original failed CI evidence or the earlier author receipts.
