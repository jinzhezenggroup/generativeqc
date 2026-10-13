# Pure-Coulomb force task demand

Status: proposed (complete GPU qualification pending)
Date: 2026-10-13

## Diagnosis

The r2SCAN-3c direct 96-atom endpoint spends about 77.8 seconds per warm call
in stationary integral derivatives. Full-range force queues always request
both J and K density bounds, even when their authenticated exchange coefficient
is exactly zero. The existing requested-source screening helper already knows
how to exclude K-only tasks, but these callers do not forward the demand.

## Candidate

Forward pure-J demand into the existing scalar screening predicate on the
bounded generic/materialized, spatial-low, dedicated order-seven and mixed-f
class consumers. Do not change any screening threshold, signed coefficient,
radial recurrence, density-product formula, source layout or SCF schedule.
Fock-purpose compatibility callers and nonzero exchange retain both sources.
Missing resident-cache/topology guards keep the existing bounded routes.
The optional angular-force path remains its existing conservative superset.

This removes work demanded only by an absent exchange source. It is not a
relaxation of the requested Coulomb force threshold. Independent E/F checks,
finite differences, constrained/fallback cases and the complete 96-atom
population must pass before promotion. Host tests execute the actual source
predicate and dispatchers with live/absent J/K products and resource faults.

## Separate spatial-reuse diagnostic

Source `227c55c06` with explicit `spatial-reuse` reduces integral derivatives
to about 63.2 seconds and warm endpoints to about 90.9 seconds. It still times
out in the final moved replay at the original 1800-second limit. Different
cold/moved SCF trajectories offset the warm reduction (35/18 iterations versus
28/13 in the prior default run). That diagnostic's original reference gates
are null, since its fresh directory lacked references; do not replace those
nulls or present the run as a qualified default. No spatial default is changed
by the pure-J demand candidate. Full raw attempts remain in ignored artifacts.

## References

Issue #2416; `src/scf/cuda/direct_screening.cuh` and the existing
`tests/python/test_direct_force_source_screening.py` source-demand gates.
