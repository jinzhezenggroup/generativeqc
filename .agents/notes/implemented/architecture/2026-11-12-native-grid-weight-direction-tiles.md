# Decision: share native grid adjoint arithmetic with tiled weight directions

Status: implemented
Date: 2026-11-12

## Problem

Native metric/field response accepts abstract external Fock directions, but
an actual nuclear source must also differentiate moving molecular quadrature.
`MolecularGrid::contract_weight_derivative` exposed only the final nuclear
gradient contraction. Calling it once for every point would repeat the full
grid traversal, and storing a point-by-nuclear-coordinate Jacobian would
defeat bounded source execution.

## Decision

Factor the existing point traversal into a private partition-derivative visitor.
Keep its generated local arithmetic, exact branch conditions, signed pair
incidences, log normalization and adjoint accumulation order. The existing
contracted-gradient caller still multiplies each retained point weight by its
sensitivity before the same normalized-log tangent, preserving its arithmetic.
The new tiled direction caller contracts that same derivative with one nuclear
direction, stages the requested outputs and publishes only after success.
There is no second Becke derivative implementation or displaced production grid.

Callers own explicit numeric workspace. It contains the existing atom-by-
nuclear-coordinate log-derivative scratch and staged tile outputs, not a
full-grid nuclear Jacobian. The query includes this storage, but not retained
grid data, scalar stack or metadata. A separate shape-only work query counts
point and pair visits. Every requested point still visits every atom pair;
this is not a complete endpoint scaling or resource claim. Larger source
owners must impose their own total-work limits before starting.

## Invariants

- Retain v1/v2 unpruned, v3 radial-pruned and v4 Treutler/Lebedev prescriptions,
  including unequal sourced radii, iteration counts and ownership ordering.
- Preserve the existing coincident-separation and clipped-coordinate branches.
- Do not evaluate an inactive singular norm or replace physical derivatives
  with floors. Nonfinite completed directions fail without publication.
- The grid is borrowed and immutable; numeric workspace cannot alias any
  retained grid arrays/centers, directions or outputs. Outputs may alias the
  external direction because publication is staged.
- New directional work does not authorize frozen-XC, Hessian, installed Ni/ECP,
  large-molecule endpoint or TS capability promotion.

## Qualification

`test_host_grid_geometry.py` compiles the actual changed `src/dft/grid.cpp`
standalone with a freshly generated atomic-grid header and strict FP64 flags,
through a verified compiler cache. It does not rely on the older frozen core
library to exercise this change. Independent 60-digit direct-product displaced
oracles cover partition iterations 1/3/5 and all four grid prescriptions;
tiled/full direction equality, adjoint duality and translation conservation
retain separate gates. The oracle does not invoke Graph or production
derivatives. Failure injection tests cover late nonfinite weights, capacities,
shapes, aliasing, transactional publication and recovery.

Mixed Ni/I/Cs native-default grid controls and fixed-tile 25/40-atom capacity/
work controls exercise this source only. They are deliberately not reported
as the still-required mixed-element or 25--40-atom molecular endpoint gates.
Raw local evidence is under `build/qualification/host-grid-motion-20261112/`.

## Rejected alternatives and next boundary

Avoid repeated full-gradient calls per point, a dense full-grid Jacobian,
handwritten duplicate partition algebra, finite-difference production sources
and easier-grid substitutions for molecular endpoint qualification. Moving AO
jets and the fixed-density XC Fock derivative remain the next composition,
alongside independently owned one-/two-electron and ECP nuclear sources.

## References

- [Native metric/field response](2026-11-12-native-host-metric-field-response.md)
- [Current grid contract](../../../../docs/developer/dft_grid.md)
