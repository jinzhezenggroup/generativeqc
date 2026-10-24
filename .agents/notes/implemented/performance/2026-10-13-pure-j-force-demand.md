# Pure-Coulomb force task demand

Status: implemented (the large-population timeout remains unresolved)
Date: 2026-10-13

## Diagnosis

The r2SCAN-3c direct 96-atom endpoint spends about 77.8 seconds per warm call
in stationary integral derivatives. Full-range force queues always request
both J and K density bounds, even when their authenticated exchange coefficient
is exactly zero. The existing requested-source screening helper already knows
how to exclude K-only tasks, but these callers do not forward the demand.

## Decision

Forward pure-J demand into the existing scalar screening predicate on the
bounded generic/materialized, spatial-low, dedicated order-seven and mixed-f
class consumers. Do not change any screening threshold, signed coefficient,
radial recurrence, density-product formula, source layout or SCF schedule.
Fock-purpose compatibility callers and nonzero exchange retain both sources.
Missing resident-cache/topology guards keep the existing bounded routes.
The optional angular-force path remains its existing conservative superset.

This removes work demanded only by an absent exchange source. It is not a
relaxation of the requested Coulomb force threshold. Independent E/F checks,
finite differences and constrained/fallback cases qualify task-demand forwarding.
The complete 96-atom population remains a separate acceptance gate for issue
closure. Host tests execute the actual source
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

## Qualification and limits

Integrated source `9e21b8cb6` and native library SHA256
`d6cb650cb9f66e65282a0b68a2e503f3c24bc49abff70a486d03d777f9d37a1f`
passed 33 host screening/dispatcher contracts and six real-device independent
analytic, finite-difference and moved/fresh force checks (Slurm job 8167).
The complete 12-call 3/12/48-atom direct r2SCAN-3c populations passed all
1e-8 Eh / 1e-7 Eh/Bohr gates (job 8169). The 48-atom complete cold/warm/moved/
moved-warm times were 87.608/25.845/66.400/25.821 seconds (replay medians).

At 96 atoms, 11 returned calls passed, with maximum energy/force errors
3.7744e-11 Eh / 9.8063e-11 Eh/Bohr (job 8168). Complete cold was 491.255 s,
warm median 102.283 s, moved 320.359 s and four moved replays about 102.454 s.
The final moved replay still exceeded the original 1800-second whole-point
limit. Compared with the earlier default 105.539-second warm median this is
only a small improvement. It does **not** solve #2416 by itself. Preserve this
failed outcome and test further scheduling work separately; no deadline,
threshold, grid, precision or oracle was changed.

## References

Issue #2416; `src/scf/cuda/direct_screening.cuh` and the existing
`tests/python/test_direct_force_source_screening.py` source-demand gates.
