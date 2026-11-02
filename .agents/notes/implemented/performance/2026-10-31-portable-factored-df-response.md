# Decision: portable charge partitioning and factored pure-J metric response

Status: implemented
Date: 2026-10-31

## Problem

Long Hermite charge source classes were partitioned automatically only on
SM 12.0. The same bounded schedule improves complete A100 endpoints, so the
architecture restriction excluded useful parallel work. Pure-J forces also
expanded the solved auxiliary potential into a dense metric cotangent before
contracting derivatives, losing the factorization already available to the
response owner.

## Decision

Keep the 8192-source threshold, automatic request of 32 partitions, explicit
1..64 controls and existing dead-buffer capacity cap. Remove the GPU model
input. Short classes and insufficient capacity retain the original traversal.
This supersedes the device restriction in the
[earlier partition note](../../proposed/2026-10-24-bounded-df-charge-partitions.md).

For an admitted full-rank pure-J owner, pass each solved signed potential
directly to a metric derivative callback. The energy contribution is
`-cJ/2 * v^T(dM)v`. Auxiliary primitives already have the unit-times-AO
representation needed by the generated Hermite potential algebra. Refresh
their density, clear their raised-potential buffer, contract every angular
class pair through g/g, and project the physical target-center derivative.
Index symmetry accounts for both metric orientations and cancels the half.
No second derivative traversal or new recurrence implementation is needed.

The bridge checks the borrowed owner, stream, device, dimensions and full-rank
pure-J contract. It installs this consumer only for the ordinary single-sink
schedule. General weights, exchange, truncated metrics, diagnostic schedules
and absent owners retain the existing bounded response. The owner's existing
value-budget admission and explicit `GENERATIVEQC_DF_MD_COULOMB=0` rollback
remain in force. No additional allocation or primal screening enters response.

## Work and numerical invariants

- Each nonzero Coulomb term avoids forming and traversing an `naux^2` metric
  weight array; the response trace records the avoided elements and calls.
- For `N` auxiliary primitives, the new contraction has at most `N^2` ordered
  primitive interactions. It does not repeat these interactions per AO matrix
  component or materialize a derivative tensor. Exact zero source factors may
  avoid arithmetic; the bound is not an asserted measured active count.
- Charge partitioning changes summation grouping, not the source domain. The
  ordered FP64 reducer, capacity cap and source-density lifetime are unchanged.
- Every call refreshes signed potentials. Zero and changed graph replays must
  remain independent of previous charge, projection and three-center calls.
- Metric output adds to an existing atom gradient. General nonsymmetric metric
  weights must never be routed to this rank-one interface.

## Evidence and limits

The retained campaign in
`benchmarks/results/portable-df-response-20261031/` records source/binary
identities, scheduler allocations, complete cold/warm populations, independent
oracle qualification and setup failures. The final production source identity
is `1714f26e6f6bd4ecac0867f29744ebbf57ae717d3e7ceeda533a7a57365bca8b`.
The early A100 metric build had a different identity and is not relabelled.

All 98 complete populations in the final 101-record campaign pass the unchanged
independent gates; three incomplete controls remain unranked. Fresh A100 warm
PBE96 is 2.222298 seconds versus GPU4PySCF's 2.306539. Warm PBE0, B3LYP and
r2SCAN-3c still lose by 1.52%, 2.60% and 4.63%; cold performance remains a
separate unresolved problem. Instrumented A100 metric regions take 2.267 ms
(PBE96) and 3.525 ms (r2SCAN-3c96), avoiding 13,075,456 and 46,457,856 dense
cotangent elements. These counts describe eliminated materialization, not a
reduction in the enclosing response workspace reservation.

Independent libcint five-point finite differences cover Cartesian and spherical
s/p, d/f, f/g and g/g metric derivatives, signed factors, additive output,
zero/changed graph replay and exact-capacity refusal. Complete endpoint tests
cover moved geometries and the raw fallback. Host fault injection retains the
occupied-response failure and lease checks. PRO 6000 memcheck covers all eight
new metric cases; larger endpoint/budget checks remain separate from timing.

The principal timing population uses the public explicit force tile of 1024
points with the SCF tile left at its default. It is not a promotion of that
force tile to the automatic default. Static water geometries and three GPU
families do not establish universal accuracy or performance. Existing HF/DF
and small M06-2X accuracy failures are not relaxed by this work.

## Rejected alternatives and revisit conditions

Copying another architecture whitelist would leave the same portability
problem. Allocating a new split arena would weaken the existing bounded
fallback. A separate metric recurrence would duplicate scientific ownership;
reuse of the generated contraction makes the derivative convention auditable.

Larger 2048/4096-point force tiles regress under the unchanged geometry scratch
cap; copying a larger batch size without its complete capacity plan is rejected.
Turning off the native Huckel guess also regresses the measured PRO 6000 PBE0,
r2SCAN-3c and WB97M-V cold endpoints, increasing iterations from 9/10/28 to
15/15/34. A different seed requires independent qualification; changing solver
tolerances or importing a reference density is not an accepted shortcut.

Revisit the threshold or partition count only with complete endpoint and
independent numerical qualification. Further speed work should measure the
remaining three-center, XC and hybrid-exchange consumers; this change does not
claim that all README functionals beat GPU4PySCF.
