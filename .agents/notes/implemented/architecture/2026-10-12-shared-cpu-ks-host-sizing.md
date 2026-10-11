# Decision: Share CPU KS host sizing across production frontends

Status: implemented
Date: 2026-10-12

## Problem

After #2290 and the native Direct-force slice, CPU KS host capacities still had
separate Python planner and native CLI implementations. The native CLI was
already Python-free, but independence of the estimates did not establish one
resource authority and allowed their lifetime models to drift.

## Decision

The allocation-free `scf::cpu_ks_host_resources` owner supplies the native CLI
and an additive private shape-only C query. Existing calculator requests pass
their already loaded library, discover the query and use its complete named
host inventory. Query rejection is visible and never retried with Python math.
Schedule identity distinguishes native sizing from portable metadata sizing.

No-library dry runs deliberately remain provider-free and retain the existing
portable shape arithmetic, also used by old installed libraries. This is a
metadata compatibility path, not a scientific production fallback. The native
query preserves output on malformed shape/schema, unsupported angular domain
and signed-int64 overflow. Nonlocal memory limits remain explicit frontend
policy over the native capacity result.

## Invariants and boundaries

- Scientific execution, model identity and SCF/force behavior do not change.
- Per-item retained owners compose across the fleet; serialized setup/SCF and
  the optional restricted AO cache remain phase maxima, not fleet sums.
- The optional cache includes its exact 64 MiB eligibility boundary. Spherical
  transformation buffers use Cartesian and public AO dimensions separately.
- CUDA sizing, device execution and resource leases are not migrated here.
- Named numeric/value-metadata bounds exclude allocator/runtime overhead and
  are not full endpoint allocation receipts or completion of #2227 R2.

## Evidence

Native bridge tests cover exact capacities, restricted/unrestricted ownership,
nonlocal/ECP buffers, malformed topology/booleans/schema and overflow with
transactional output. Python effect tests poison portable arithmetic after
native discovery, compare independent portable bounds across through-g shapes,
cache boundaries, tiles/spins and nonlocal owners, and verify explicit legacy
identity and rejection without retry. Existing CPU resource execution tests
cover exact/below-limit budgets, preparation, retained/warm/moved execution and
batch scaling. Native CLI qualification remains Python-free.

## Revisit when

Complete endpoint leases/allocation receipts or a library-free installed native
shape service can replace the remaining portable metadata compatibility path.

## References

#2227 R2/R5/R6; #933/#934; `docs/maintainer/resource_planning.md`;
`src/scf/ks_host_resources.hpp`; `python/generativeqc/resources_ks.py`.
