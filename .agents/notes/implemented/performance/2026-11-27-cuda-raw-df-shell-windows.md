# Decision: Batch single-component CUDA raw DF shell groups

Status: implemented
Date: 2026-11-27

## Decision

The raw DF owner uploads contiguous group descriptors and admits the ordered
schedule by byte, primitive, projection, and reduction bounds. The Hessian
does not add a second fixed group-count ceiling. Bound public AO expansion
by its actual shell support, and authenticate that bound against the live
device source before using the schedule. This avoids charging each s-shell
projection as though it were a worst-case spherical expansion.

When every group has a single Cartesian/public component, its primitive tile
fits, and the linked factorized leaf accepts batch descriptors, reuse one
bounded window for records, outputs and projection. Prepare per-atom incidence
lists in group/center order, keeping repeated physical centers. The atom-owned
fold reads those lists in order. Neither the ordered DF weights and metric
response nor the independent analytic leaf changes. Angular, multi-tile, or
insufficient-capacity cases retain the scalar path.

## Evidence

Slurm 2685 passed host-only budget checks and 22 real-device cases including
all-s batched and scalar references from libcint, failure admission, and
public Direct/DF Hessian/HVP regressions. On the 96-H matched STO-3G auxiliary
probe, the old path rejected raw DF admission even when the group limit was
raised (Slurm 2683). The new path reaches the dense Hessian with 893,952 raw
groups and no group-limit override (Slurm 2686). This stage observation is not
evidence of a completed 96-H result. Grid, response and exact Direct budgets
are independent, and the pilot uses explicitly bounded diagnostic overrides.

## Revisit when

At larger sizes the current two passes still rebuild every ordered group for
every direction. Improve the incidence of exactly nonzero right derivatives
and preserve the moving-weight first derivative term before claiming that
launch batching removes this repeated work. Extend batching to angular basis
sets only with independent spherical and fitted endpoint tests.
