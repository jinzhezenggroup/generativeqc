# Decision: elide exact one-center molecular integral derivatives

Status: implemented (CPU physical contractions only; Ni endpoint not qualified)
Date: 2026-11-04

## Problem

After correcting the composite AO budget lease, the external spherical
NiH4(2-) def2-SVP B3LYP probe constructed its response owner but returned no
HVP within a finite 1,800-second cold-start limit. The process exited 124.
Its retained first-derivative cache contained 185 generated C++ translation
units and 184 shared objects; no second-derivative cache had been reached.
The runtime compiler invoked the verified sccache launcher. Shared-server
aggregate statistics are not evidence of hits for these separately owned
foreground JIT servers. No orphan compiler remained after timeout.

Many expensive high-angular-momentum programs concerned four shells on the
same physical Ni atom. Their independent-center derivatives are nonzero,
but the molecular contraction translates every integral center together.
That derivative is exactly zero, independently of density, energy-weighted
density, component, angular momentum or finite displacement direction.

## Decision and invariants

Skip these exact translation zeros in the shared CPU physical first- and
second-integral contractions:

- overlap and kinetic terms whose two AO shells share one physical atom;
- nuclear attraction only when both AO shells and the charged nucleus share
  that same physical atom;
- ERIs only when all four AO shells share one physical atom.

Use physical atom identity, not equal coordinates, a numerical threshold,
screening estimates or a named functional. Keep normal topology lifetime,
direction, weight shape/finiteness and output-budget checks. In particular,
validate shell-local ERI weights before discarding their zero contribution.
The raw independent-slot derivative provider and CUDA execution are unchanged.
Spherical variational coordinates still use the existing Cartesian cotangent
workspace; the decision does not replace the physical spherical basis.

## Rejected alternatives

Raising the timeout or resource caps does not remove mathematically redundant
work. Skipping independent-center derivatives would invalidate real primitive
derivative contracts. Dropping merely small integrals or coincident coordinates
would introduce different scientific assumptions. None is part of this change.

## Evidence and consequences

`test_stationary_one_center_integrals.py` uses the pinned spherical Ni f basis
and arbitrary PSD density/energy weights. An independent exact translation
invariant requires zero first Fock/overlap, weighted first gradients and
weighted second HVP sources. It also exercises malformed weights, insufficient
output capacity and a closed owner. This test is not a molecular Ni Hessian
receipt, a timing comparison or a complete-endpoint performance claim.

Retain the failed cold endpoint and its cache when testing subsequent changes.
A successful Ni HVP, independent raw Hessian, default-grid/installed endpoint,
resource/lifecycle qualification and the separate through-f public KS force
owner are still required. Further reduction of genuinely multicenter work may
need a shared directional contraction rather than component-specific programs.

## References

- Issues #2224 and #2843.
- `python/generativeqc/rks_hessian_integrals.py`.
- `tests/python/test_stationary_one_center_integrals.py`.
- `2026-11-04-composite-ao-budget-lease.md` under implemented architecture notes.
