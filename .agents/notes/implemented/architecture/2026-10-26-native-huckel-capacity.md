# Decision: share Hückel preparation topology and capacity ownership

Status: implemented
Date: 2026-10-26

## Problem

Native Hückel execution owned its memory-admission bound, but Python resource
planning independently selected occupied atomic channels and repeated the
capacity polynomial. A future change to either lifetime inventory could admit a
budget that disagrees with the actual preparation. This remains an R2/R6 ownership
gap under #2227 even though the seed equations already execute natively.

## Decision

The selected library exposes an additive private compact-topology query. Both
actual preparation and this query use `huckel_capacity_shape` and
`huckel_numeric_capacity`. The existing native atomic-configuration authority
selects occupied s/p channels; all target contractions still contribute to target
AO and primitive counts. The query never constructs coefficients, integrals,
density matrices, atomic SCF state or a CUDA context. Its temporary metadata is
one counter per atom; work is linear in the atom and expanded-shell counts.

The scalar allowance is unchanged: assembly and strict admission are serialized,
sixteen target matrices cover the larger admission peak, and atomic cache plus
one serialized atomic solver retain their allowances. The Python planner still
composes all retained seeds with the largest serialized workspace and the target
inventory. This does not claim full allocation/RSS authority or migrate global
prepared-owner lease lifetime.

## Compatibility and invariants

- No scientific equations, SCF controls, initial-guess choices or numerical caps
  change. An empty occupied subspace remains a valid capacity query and retains
  bounded Hcore fallback during preparation.
- Old libraries and explicit offline callers retain portable Python sizing.
  Query failure from a selected current library never retries portable math.
- Input validation and checked arithmetic precede transactional output
  publication. The numeric bound remains within portable signed-int64 metadata.
- Coordinate/coefficient arrays do not cross the sizing boundary. Shape success
  is not proof that a scientific provider admits the target.
- The SDK/CLI preparation owner uses the same bound directly. The new private
  query adds no stable public ABI field or provider capability.

## Verification

Behavioral tests cover H–Ar occupied-channel counts, Cartesian/spherical
polarization shells, DIIS bounds, missing-query compatibility, native failure
without retry, null/invalid/overflow ABI inputs and transactional outputs. Actual
prepared endpoints check exact and one-byte-below host/numeric caps, ragged cold
and warm execution. Existing independent atomic-oracle and live-allocation
probes remain part of qualification; passing scalar tests alone is not GPU or
complete-endpoint performance evidence.

## References

#2227 R2/R6; `docs/maintainer/resource_planning.md`;
`tests/python/test_native_huckel_capacity.py`;
`tests/python/test_huckel_initial_guess.py`.
