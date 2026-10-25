# Decision: expose native DF policy targets for future owner reservations

Status: implemented
Date: 2026-10-26

## Problem

The remaining #2227 R2 native-admission work must reserve future mandatory
owners before spending capacity on optional XC storage. A prepared DF owner's
current retained bytes omit response storage and geometry-rebuild setup. Its
generic workload target also omits a later resident promotion that can spend
part of the live fractional headroom to retain complete B/U storage.

## Decision

Expose the existing workload/resident policy arithmetic through
`df_budget_targets`, which the base resolver now consumes directly. Expose a
separate `method_owned_df_resident_capacity` query using the existing packed
planner's uncapped sizing mode and the same response-floor helper as the live
promotion. The ordinary panel's capacity bounds the smaller minimum found by
promotion's bisection. Queries perform shape arithmetic without CUDA probing,
numerical execution or invented free-memory observations.

The maximum of the base targets and promotion target covers these allowance
decisions. It is deliberately not advertised as a complete Fock or KS fleet
bound. Initial source construction temporarily retains its generated metric,
and one-electron/grid setup, retained force consumers, additional providers and
serialized geometry rebuilds still require composition by their native owners.
This change does not activate optional XC storage or the resource cutover.

## Invariants

- Explicit positive budgets remain hard caps; probe-failure fallback, response
  splitting and live headroom policy retain their existing behavior.
- Capacity queries share the native planner and response floor. Do not copy
  scientific equations or packed-panel formulas into a frontend or caller.
- A future complete promise must reserve all relevant owner lifetimes. Observed
  current free bytes, a selected inventory row or one policy target is insufficient.
- Overflow or an unsupported packed shape must remain explicit; a failed query
  does not authorize an unbounded path or a fake zero-capacity plan.

## Evidence

The compiled host tests cover tiny/large shapes, explicit/probe-failure/live
envelopes, saturated dimensions, occupied and rank-zero promotion, overflow,
and the original packed B/U and response-floor acceptance cases. They perform
no GPU execution and make no endpoint performance or scientific promotion claim.
An opt-in comparison compiled baseline `59b6161c59718c1f8ac6f4fcc68c7e6a2f82621b`
and the changed policy in separate executables: all 19,200 complete resolver
records matched byte-for-byte across tiny/large shapes, live/unavailable probes,
explicit caps, response splitting and rank-zero/occupied promotion. This local
migration check is not a permanent source fingerprint or required-CI gate.

## References

- #2227 R2; #2798 historical optional-XC regression diagnosis
- `src/scf/df_preparation_budget.hpp`
- `src/scf/density_fitting.cpp`
- `tests/python/test_df_preparation_budget.py`

## Prepared Fock phase composition

The additive `prepared_fock_device_capacity` query now composes those leaves
with the native source-metadata bound. For mixed Direct/DF, base resolution
already includes Direct, but resident promotion acts on the DF subbudget.
Therefore its ceiling must add the retained Direct bound back. Positive caps
remain shared caps and must not receive that extra charge a second time.

Initial source construction holds the uploaded metadata and auxiliary metric
before the value owner exists. Even a tiny explicit DF cap may fail only after
this setup, so clipping this phase to the value/response cap would undercount
an unsuccessful preparation. The KS fleet's additional Fock setup input composes
this phase with other retained items, provider allowances and lazy force owners;
its default zero preserves the existing v1 transport and Direct-only behavior.
The query deliberately excludes one-electron/ECP, SCF/XC and opaque providers.
It does not activate promises or optional XC admission.

New executable native tests compare these bounds to independently resolved
base/subbudget/promotion records across explicit and automatic caps, live and
unavailable probes, exact-provider sizes and rank-zero/occupied owners. They
also cover source-metric overflow, topology-only queries without CUDA, serialized
fleet setup and transactional overflow. Existing source-allocation doubles
continue to check that the shared metadata bound covers actual uploaded records.
