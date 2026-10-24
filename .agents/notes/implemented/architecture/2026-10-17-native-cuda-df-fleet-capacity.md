# Decision: share complete CUDA DF numeric fleet composition

Status: implemented
Date: 2026-10-17

## Problem

The CUDA HF DF Python planner composed bucket grouping, preparation/response
bounds, storage alternatives, retained caches and singleton retries itself.
The Python-free CLI separately implemented the same policy for uniform fleets.
Native source-upload, DIIS and dense/packed tile owners already existed; moving
one more leaf would not remove either frontend's numeric orchestration.

## Decision

Use `scf::cuda_df_fleet_resources` for complete numeric bucket/fleet policy in
both frontends. Compact expanded item records admit repeated topology for the
CLI without duplicating its batch metadata. The private versioned ABI publishes
numeric bucket/tile inventories after complete validation; Python only packs,
validates and renders them. Native metadata scratch is linear in items/buckets.
An isolated Python compatibility module is reachable only when the new native
symbol is absent, never after native rejection or malformed publication.

## Invariants

- Preserve all compatible ragged buckets, RHF/UHF semantics, sorted scientific
  bucket keys, resident/source choices and requested/default sub-budgets.
- Explicit packed/packed-single never silently falls back to a dense estimate.
  Packed-single keeps the historical two-factor conservative shape bound.
- Retained buckets coexist. Preparation/setup/force work is serialized; cold
  singleton numerical recovery coexists with the warm fleet and receives a
  complete ordinary eigensystem workspace and method-aware factor reservation.
- Reuse existing native source/DIIS/tile owners rather than reproducing their
  leaf equations. Keep response staging and final spin-frame bounds intact.
- Query output is transactional for invalid later buckets, insufficient budget,
  overflow or aliasing. No scientific work, CUDA initialization, new capability
  admission or hidden CPU fallback is introduced.

## Rejected alternatives

A homogeneous-only helper would leave the production Python ragged owner in
place. A leaf-only migration would retain the duplicate fleet policy. Treating
packed-single as a lower allocation without independent execution qualification
would weaken the retained contract. Retrying Python after a discovered native
failure would split authority and potentially admit an explicitly refused plan.

## Evidence

`generativeqc_hf_df_resource_tests` exercises repeated versus expanded topology,
ragged retained/serialized lifetimes, storage/exchange alternatives, ABI schemas,
transactional overflow/refusal and disjoint buffer rules.
`tests/python/test_hf_df_fleet_resources.py` exercises the actual compiled owner,
full publication parity with old-library behavior, native exact/below sub-budget
and global admission, compatibility guards and failure/corruption injection.
The native CLI runtime contracts compare its actual compiled inventory with
the Python publication without claiming synthetic checks are real GPU science.
These contracts are not a substitute for exact-head installed endpoint/GPU
qualification and independent DF energy/force/lifetime evidence.

## Consequences

Capacity-policy changes have one native implementation. Old binaries retain a
bounded compatibility path. Metadata allocations remain distinct from declared
scientific buffer bounds; full allocator/provider/lease/RSS authority is still
outside this slice. No response/HVP migration (R3) is included.

## Revisit when

The supported SDK compatibility floor permits removing the old-library module,
or independent allocation/endpoint evidence authorizes a different packed-single
bound or broader CUDA DF scientific inventory.

## References

- Tracking issue #2227, native resource-authority slice R2.
- `src/scf/hf_df_resources.hpp` and `src/scf/hf_df_resources.cpp`.
- `docs/maintainer/resource_planning.md`.
