# Decision: compose resident RHF values and compact metadata handoff

Status: implemented
Date: 2026-10-04

## Problem

The compact CUDA RHF metadata source (#1828) and resident interaction source
(#1870) share the reference-to-correlation boundary. Selecting either branch
wholesale loses the other's lifetime, admission, or replay contract.

## Decision

The normal cached and uncached adapters prefer the eligible exact resident
interaction source. It exclusively takes the RHF plan from the cache slot.
When the resident source is ineligible, the adapters may instead copy compact
immutable public-AO metadata. That source owns a separate allocation; a cached
RHF executable stays in its existing slot. Uncached execution destroys the
local RHF owner after compaction finishes. The explicit
`run_rhf_cuda_with_source` qualification entry continues to request compaction.

Both sources use one `reference_interaction_source` state slot. Temporary
handoff telemetry aliases are cleared before correlation, retirement, or
successful reclaim. The CPU prepared source is unchanged. DF correlation
passes the original orbital System directly and never constructs RawSource
just to recover it.

## Invariants

- Resident value dispatch and its multistream fences are unchanged. Reclaim
  still requires a successful complete endpoint, one source owner, no force
  views, and established completion of all submitted uses.
- Compact source admission charges the reference peak plus the retained compact
  allocation, source System including ECP capacity, and control storage.
- Cached compact execution has two distinct owners: the reusable RHF plan and
  the compact source. The existing executable-budget helper reserves the first;
  the provider/problem source reservation covers the second. Neither owner is
  hidden by aliasing it to the other.
- Provider, solver, triples and forces may retire optional owners on memory
  failures only. Other failures propagate. A rejected compact allocation keeps
  the completed physical reference and bounded RawSource fallback available.
- No integral equations, FP64 policy, screening, provider capability, ABI layout,
  or default-performance policy changes in this composition.

## Evidence

Host fault-injection uses the actual adapter and resident source to cover
resident preference, cached and uncached compact dispatch, replay, optional
rejection, exact/short budget telemetry, transport failures, and the existing
multistream exceptional lifetime and sole-owner reclaim cases. Correlation
fixtures execute the production reference prefix and downstream ownership
transitions for both source types. The actual adapter, compact implementation,
and correlation translation units also compile with ABI-shaped CUDA headers.
This is bounded host evidence, not CUDA numerical or performance qualification.

The compact implementation's actual translation unit is additionally linked to
host CUDA allocation/copy shims to test exact/one-byte-short admission, ECP
capacity, completed copying, allocation failure, transport error propagation,
and malformed-view rejection. The shim does not qualify device ERIs.

## Retained limitation

The pre-existing force planner can conservatively count a source twice when it
is already included in `problem.reference_retained_bytes`. That behavior is
unchanged; its cleanup is independent of this ownership composition. The source
charge is in `src/methods/rccsd_method.cpp` (the `exact_source_retained` block);
`src/cc/rccsdt_force.cpp:417-419` adds `source_bytes` again. Both input heads
contain the same force-file blob `ff1decf80aa520d5a3bd17d93a366611a2aeb9ab`.
The compact input adds its source at lines 509-515 and the resident input at
602-609. This can reject an otherwise fitting near-cap endpoint or retire its
optional source, but it cannot turn the bound into an under-reservation. No
tight-budget optimality claim is made.

## References

- #1828 and `2026-10-04-rhf-cc-source-handoff.md`
- #1870 and `2026-10-04-resident-reference-transfer-lifetime.md`
- #1869 and `2026-10-04-correlated-plan-capacity-fallback.md`
