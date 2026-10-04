# Decision: reserve packed projection capacity at the KS method boundary

Status: implemented
Date: 2026-10-02

## Problem

The #1686 full-driver GPU receipt on frozen `31fb7203` completed ordinary
PBE0-DF forces but never borrowed the final occupied projection. Packed-single
preparation used rank capacity zero: the generic prepared Fock owner could not
promise an occupied determinant. Dense producers could retain U, but the force
consumer deliberately admits only packed-single. Numerical parity therefore
did not demonstrate the new producer/consumer route.

## Decision

The KS method explicitly requests `FockOccupiedProjectionReservation` for its
validated integer restricted CUDA fitted-hybrid determinant. It uses the same
spin-occupation resolver as CUDA KS, rejects invalid or out-of-basis occupations,
and leaves unrestricted, CPU, semilocal, exact and range-separated routes empty.
The prepared owner validates the request and includes it in cache matching.
Generic fixed-density callers retain the default empty request, even for the
same geometry, electron count and AO dimensions.

Only the packed-single tile planner receives the requested U rank. Its resolved storage
contract, including a possible zero-rank fallback, reaches native allocation.
The existing planner charges complete U against the provider's value subbudget;
under automatic exchange policy it may retain packed B while dropping optional
U. Infeasible budgets and explicit diagnostic policies retain their existing
rejections. Packed-raw keeps rank zero: its force consumer is unsupported and
its planner cannot drop optional U. Reserving U there rejected a previously
feasible rank-zero budget in the host planner regression. Dense and automatic
layout selection are unchanged. No RHF-owned
factor reservation is requested because KS already owns Cocc.

## Invariants and rejected alternatives

- Capacity is not density provenance. The current completed occupied K, exact
  density-generating Cocc, method owner, solve epoch, density generation, metric
  rank, provider scratch generation and consumer storage checks remain required
- Do not infer occupation authority in a generic provider from AO dimensions or
  electron count, or promote automatic packing to make this test pass
- Do not weaken the packed-single consumer admission or retain U outside the
  existing budget. K still consumes the borrowed scratch before J invalidates it
- The reservation can increase persistent U capacity when bounded panels were
  smaller; it is optional charged storage, not an allocation-free optimization
- The separate #1661 LDA history gate is unchanged and unresolved by this repair

## Evidence

Host probes execute the real method initialization, constructor forwarding and
validation, cache matching, packed tile planning, source allocator arguments,
provider admission and dense/packed lease lifetime guards. The budget cases
include packed B with no complete U, a roomier U reservation and an infeasible
cap. These probes do not execute CUDA allocation or projection arithmetic.

`test_public_pbe0_packed_projection_hit_and_fallback` requires the completed
`response_final_fitted_projection_reused` arithmetic counter and terminal
`atom_coordinates` count in the same validated response row, alongside a
successful public packed-single CUDA E/F endpoint. A partial arithmetic attempt
followed by a successful ordinary retry cannot qualify reuse. The earlier
`response_reused_final_fitted_projection` admission counter precedes allocation
and the force trace scope; it may be absent or followed by ordinary OOM fallback
and cannot establish a completed hit. The test compares
against reuse-off ordinary response, unsupported dense storage, and an actual
8-MiB public DF budget on a 24/24-AO water fixture. The original energy/force
parity gates remain `1e-8` Ha and `3e-7` Ha/bohr. This test was not run on a GPU
for this repair; no new device qualification or speedup is claimed.

## References

- [Fixed-commit non-hit receipt](https://github.com/jinzhezenggroup/generativeqc/pull/1686#issuecomment-5947873701)
- #1661 (final projection producer), #1686 (method/force integration)
