# Combined forces consume the materialized dddd recurrence

Status: implemented; real-device qualification pending
Date: 2026-10-06

## Problem

#2030 selected `Full` for Combined total forces, while the optional materialized
pair derivative scheduler from #2020 admitted only `FullSources`. Combined
therefore used per-AO recurrence even when the materialized derivative switch
was enabled. The same mismatch affected bounded and angular scheduling.

## Decision

Template the compiler-owned materialized force consumer on the existing
`DirectForceOutputMode`. Its common `DirectForceSources` contract provides
one signed J/K weight and one output array for Combined, or two independent
weights/arrays for Separate. Both consume the same scalar order-9 Coulomb
simplex and incumbent Dual3 Hermite responses. There is no additional spin or
exchange factor, new derivative recurrence or precision change.

Both native launchers admit the materialized specialization for full-range
forces, and their generic drains skip exactly the dddd tasks that it consumed.
The original default-off preparation switch, resident-cache requirements and
reachable/convolution exclusions remain. Other shells and radial operators
retain their existing consumers. Connecting the candidate is separate from
promoting it based on complete endpoint measurements.

## Invariants and evidence

The native gate checks signed combined forces against an independent host
symmetry-orbit contraction of retained raw derivatives, alongside Separate.
It covers both spins, repeated atoms, same-pair triangular domains, Schwarz
screening, inactive/zero-density tasks and a zero second-output canary.
The existing public Libcint gate also exercises Combined and Separate at the
same final density with materialized derivatives enabled.

Host production-dispatch qualification covers Combined, Separate and long-range
requests, every resident/cache admission state, workspace selection, disjoint
order ownership and submission error propagation. The initial targeted host
run passed 11 tests; compiler structure checked 480 modules with zero errors.
Device results and any endpoint measurements will be appended separately.

## Consequences

The disabled candidate keeps its prior specialization and resource footprint.
The enabled Combined specialization has one density weight per AO component
instead of two. Materialized recurrence publication is still prepared by one
CTA lane, and component-level global force atomics remain; cooperative
publication and shell-level reduction are separate follow-up work.

## References

- #2020: materialized pair derivative consumer and fallback.
- #2030: Combined stationary total-force layout.
- `docs/developer/direct_pair_recurrence.md`: current contract and GPU gates.
