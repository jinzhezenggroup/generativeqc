# Decision: qualify the existing combined Direct derivative for resident RKS

Status: proposed; implementation candidate, device qualification pending
Date: 2026-10-30

## Problem

The shell Direct derivative executor already supports two publication domains:
independent weighted J/K channels and a total channel whose cotangents are
combined before automatic differentiation. The resident source adapter always
requested independent channels, including for a total-only stationary gradient.
This repeated the derivative traversal for nonzero exchange. Pure J already
uses one traversal and provides no new work-reduction opportunity here.

## Decision

Thread the existing `separate_sources` request through the resident adapters.
Only the noncomposite, restricted, full-range hybrid total-gradient controller
requests the combined channel. It carries `FullRangeJkSum` and the actual
`Force` convention. The compiler-validated unit-source sum accepts either force
or energy-gradient transport, applying its explicit sign exactly once.

The receiver stores the total in its existing reduction arena and zeros the
other slot. Its combined-source mask continues to reject independent component
diagnostics, duplicate J/K submissions and incomplete inventories. Component
APIs, UKS, and range-separated adapters retain separate channels. A resident
combined UKS request is rejected before traversal or callback publication.

## Invariants

- Reuse existing coefficient conversion, screening, topology, recurrence,
  generated differentiation and bounded scratch. No new force equation,
  hand-factored joint expression or arithmetic precision policy.
- The callback borrows provider storage only until its copy is enqueued on the
  producer stream. Preserve failure drains, typed errors, capture refusal and
  all-or-nothing final publication.
- A combined channel cannot claim independent J and K diagnostics.
- Do not infer H100 numerical or endpoint performance qualification from another
  target's generated profile, compilation, or the standalone DF Rys probe.

## Evidence and admission

The candidate extends the native resident-provider test to compare combined and
separate outputs with independent CPU integral derivatives, and runs the same
callback failure/recovery and capture gates for both publication modes. The
native receiver tests exercise both transport conventions, duplicate refusal,
diagnostic refusal, reset and incompatible spin/range profiles.

Those CUDA tests have not yet executed on this candidate. Promotion requires a
source-bound SM90 build, actual work/dispatch attribution, independent complete
molecular E/F gates, moved/warm, constrained-memory, lifetime and batch checks,
and clean same-device complete endpoint comparisons. The fixed majority matrix
and unfavorable historical results remain unchanged. Host generation/contract
tests do not substitute for these device gates.

The source-bound automatic Libxc inventory is regenerated after preserving its
immediately preceding summary. The retained native campaign was already stale;
this change adds no executed qualification, stage demotion or public selector
restriction. Obsolete transition acknowledgements are removed rather than
carried into a different source transition.

The integration candidate also retains upstream `a0a375c3`'s
[molecular VV10 schedule](../implemented/performance/2026-10-31-native-vv10-molecular-schedule.md).
That change's measurements retain their original devices and source identities.
The earlier H100 matrix frozen at `000517f8` is an independent baseline; it is
not qualification of this newer combined candidate. The new integration still
requires its own SM90 build and device gates.

## Rejected alternatives

An earlier naive two-output common-subexpression experiment was rejected after
increased estimated liveness and a strict parity failure in a factored probe.
This candidate uses the already implemented combined cotangent traversal; it
does not revive that expression rewrite or assume its predicted performance.

## References

- [H100 Direct follow-up #2897](https://github.com/jinzhezenggroup/generativeqc/issues/2897)
- [H100 production tracker #2889](https://github.com/jinzhezenggroup/generativeqc/issues/2889)
- [Portable DF and unit-sum receiver #2888](https://github.com/jinzhezenggroup/generativeqc/pull/2888)
- `docs/maintainer/performance_engineering.md`
