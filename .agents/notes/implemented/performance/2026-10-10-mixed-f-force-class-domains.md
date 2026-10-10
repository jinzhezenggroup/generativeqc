# Decision: isolate exact s/p/d force domains inside mixed f bases

Status: implemented
Date: 2026-10-10

## Problem

The existing whole-basis split requires maximum angular momentum two. Ethane
aug-cc-pVTZ contains f shells, so its two physical nuclear derivative passes
still used one mixed consumer, despite having qualified scalar, weighted,
cooperative and materialized s/p/d consumers. After RHF phase-value reuse,
these derivatives remained approximately 90 seconds of the complete cold E+F.

## Decision

For full-range force requests with maximum angular momentum three, compatible
resident primitive pairs, both resident derivative schedules, the same plan's
class-major topology and the standard launch dimensions, partition the 21
s/p/d quartet classes into five disjoint consumers. Scalar orders zero through
three, weighted order four/five, cooperative order six/seven and materialized
dddd retain their existing compiler-owned scientific functions.

Class pages borrow immutable offsets and physical pair IDs. Rectangles and
diagonal-inclusive triangles enumerate each owned task once, including ragged
and empty systems. Scalar workers use 128-ket pages and 128 lanes; shared
recurrence workers use 32-ket pages and all 256 coefficient lanes. Every
f-containing class, including low-total-order f classes, stays in the original
bounded recurrence with its qualified 128-thread nonmaterialized schedule.
There is one residual bounded traversal, not thirteen full angular scans.

The original whole-basis s/p/d split is unchanged. Missing caches/topology,
incompatible derivative recurrence modes, other angular bounds and custom
launch dimensions retain the complete mixed fallback. Existing explicit
`GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES=0` also retains that route.
No new public opt-in, resident numerical arena or derivative tensor is added.
Submission/reset errors propagate; partial outputs never justify fallback.

## Invariants

- Strict FP64, physical pair orientation, screening and generated-class ownership.
- Two D+W/D-W physical passes, not one fused or approximated force pass.
- The complete RHF/CC/(T)/Lambda/Z work, replay, stationarity and physical audits.
- Force-owner density bounds, never the Coulomb-owner bounds inside topology.
- Stream lifetime protects borrowed topology, caches, outputs and cursor.

## Evidence

The measured base is `e55dcd2b2ec4acd52d9f45d3f9fa2ebf709d6d7d`, including the
qualified RHF-auto implementation and master `742dff879`. The retained
reconstruction patch pins the uncommitted candidate; it is not labelled a clean
new Git revision. Library SHA256 is
`9b42eccb04df382615355f9338f12c1c0fa7ff8d9423590db708386021b03e8c`.

A final fetch on 2026-10-10 observes newer master
`7f342546d887796e6a92a005ba033029ed73ce2a`, including the RHF-auto squash and
subsequent canonical-pair/DFT changes. These observations are not relabelled as
measurements of that newer tree; integration onto it needs source-matched
requalification.

Three fresh n1 RTX 5090 UUID-stratified ABBA allocations, jobs 7143/7144/7145,
provide six complete samples per side. RHF values are default-auto on both sides.
Complete native E+F median is **349.802091 to 326.742221 seconds**, a **6.5923%**
reduction and **1.07058x** speedup. Two-electron derivative median is
**89.908148 to 67.233617 seconds**, a **25.2197%** reduction. Every stratum is
positive and the descriptive MAD noise-floor check passes. One-second sampled
total-device occupancy peaks are **31,151 MiB on both sides**, not exact owned
or allocator peaks.

Separate n1 graph-node profile job 7155 shows two physical force scopes, two
128-thread f fallback launches and two launches of each specialized domain.
Kernel sums are f fallback 55.027298 s, cooperative 9.832603 s, dddd 1.982722 s,
low order 0.261471 s, order four 0.177528 s and order five 0.244180 s. These sums
are not clean phase wall times. Linked scalar/f kernels still report 255
registers, versus 227 cooperative and 226 materialized: prelink caller-only
resource counts must not be presented as final kernel resources. NCU hardware
counters are not collected (`RmProfilingAdminOnly=1`); no achieved-occupancy or
spill-traffic explanation is claimed.

Host address/admission/fault tests cover the exact 8/5/3/4/1 class partition,
128/32-page boundaries, empty systems, monotonic worker claims, 32-bit pair
inventory limits, both source layouts/spins and every submission/reset error.
Focused host validation passes 248 tests (seven allocated-device skips), and
adjacent validation passes 230 tests. Allocated n4/job720 independently checks
Cartesian/spherical mixed-basis two-electron CCSD(T) energies and all six forces
against fresh PySCF FCI finite differences (four tests). Native arbitrary-density
J'/K'/combined gates cover both spins, source masks, opposing/zero densities,
unequal primitive lengths and two distinct geometries, including the old route.

Full mixed-provider memcheck on n4/job721 reports zero errors and zero leaked
bytes. The earlier job720 sanitizer could not attach before expensive CPU oracle
preparation; that failed attempt is retained. The native qualification initializes
its allocated CUDA context before CPU oracle work, and the retry uses a finite
allocation and finite 600-second attachment timeout.

Every clean ethane result and the separate diagnostic passes retained independent
energy/FD checks and the qualified 24-force-vector comparison. These ethane
oracles are retained, not newly recomputed full ethane force oracles. All raw
endpoint results, identities and source-matched recipes are retained in the
compact benchmark publication; raw profiler/build/sanitizer logs remain in the
local audit evidence directory.

## Rejected alternatives and follow-up

Do not merely remove the whole-basis angular guard: an order-seven/eight f
quartet is not a proved dddp/dddd task. Do not replace class-major pages with
repeated full-domain scans or retain dense derivatives near the device limit.

The shared-channel candidate is not applied or counted in this result. The
separated s/p/d low/order-four/order-five work now totals only 0.683180 device
seconds; f-containing low-order work is not separately attributed inside the
55-second residual. Any channel change needs its own numerical/resource and
complete-endpoint A/B qualification. Lambda remains about 87.74 seconds;
compiler-owned exact operator-work reduction is the next major candidate,
not further residual-cadence tuning or forced large core retention.

## References

- [RHF-auto predecessor](2026-10-10-rhf-phase-values-auto.md)
- [Source-matched publication](../../../../benchmarks/results/force-class-domains-20261010/README.md)
- Local audit: `/home/jzzeng/codes/qc-branch-audit-20260922/evidence-force-class-domains-20261010/`
