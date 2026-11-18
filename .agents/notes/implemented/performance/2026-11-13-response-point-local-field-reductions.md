# Decision: point-local response AO field reductions

Status: implemented; full molecular scale and TS acceptance remain separate
Date: 2026-11-13

## Measured problem

Removing repeated scalar XC evaluations did not remove most response cost.
Inclusive user-space instrumentation of the exact bank producer measures real
public Cs RKS/UKS response XC at56.93/97.09 seconds of104.24/185.75-second complete
raw-H endpoints. RKS AO evaluation of all matching first-order calls takes10.00
seconds, overlapping the containing phases. Nuclear first/curvature XC takes
12.77/23.97 seconds RKS and25.18/51.87 seconds UKS. Do not add inclusive timers,
subtract different-entry-point Fock scopes, or interpret skipped streamed integral
entry points as zero matrix-preparation work. The terminal audit is retained in
`build/qualification/host-endpoint-phase-audit-20261113`, seal
`373abe2d95aff9c580cbed75a9fca28d5564d5c53520e3c5d497ebe70eae5190`.

## Decision

Accumulate primal and tangent density/gradient fields in two four-double local
arrays for each spin/point, then publish them once to the existing tile panels.
This removes repeated panel reads/writes inside the quadratic AO loop without
changing a field's row/column reduction order, product arithmetic, generic full
density semantics, tile traversal, mandatory AO/grid reads, linked scalar IR,
exact-input authentication, response solver, output contraction or work counts.
No heap panel, optional admission priority, resource cap or public ABI changes.
The locals occupy64 bytes independent of AO count, point count and spin count.

The independent Libxc test helper now forms both terms of the gradient-density
contraction rather than assuming a symmetric input. New nonsymmetric-density and
direction cases exercise both spins and both streamed/cached execution at all
three original finite-difference steps and unchanged gates. This checks the actual
generic matrix contract, not a mandatory internal spelling or source fingerprint.

## Evidence and limits

Producer: `eaf9a93f08a47bbd88f53a4a3e0c2a0a765ab394809a9180f0f0dc4b5bf3d039`.
Frozen core: `553181540d69cbff836d248187a2b51f3bc910600f822c2784723a973f1aaa9a`.
Evidence directory: `build/qualification/host-response-local-fields-20261113`.
The48 XC cases and123 targeted Python cases (five deselected) pass, as do all
seven native/API/CLI contract groups. Both real public raw Cs matrices are
array-equal to the independent-qualified exact-bank predecessor, maximum
difference zero. Fresh H endpoints take84.09/147.45 seconds; inclusive response-XC
intervals take36.62/58.82 seconds. Other XC phase timings and semantic work remain
unchanged. These are particular observed local runs, not a controlled general
speedup experiment. Driver peak RSS is not a derivative-only storage bound.

The earlier183 source cases and independent Cs matrices retain their own
producer identities; this cohort does not relabel them as rerun. Its exact raw-H
preservation comparison uses their accepted frozen artifacts. Further composed
response/HVP terminal results belong in the local binding, not an inferred pass.

Final composed-response validation passes all16 cases in398.06 seconds, including
independent displaced-SCF D'/W' checks and exact prefix/streamed budget parity.
All four fresh real Cs public HVPs are array-equal to the same accepted parent;
the two HVP endpoints take41.21/64.47 seconds per spin case in aggregate. All
timers are particular observed runs. These checks do not rerun or restamp the
parent's independent central/displaced Cs SCFs or its183 source cases.

This scheduling change does not change O(points*AO^2) work. BLAS batching would
change summation order and require separately budgeted panels, full nonsymmetric
semantics and independent numerical qualification; it is not silently introduced.
CPU value preparation also retains an independent dense AO4 scalability boundary.
No25--40-atom endpoint, stationary mixed Ni/ECP modes, I/Cs molecular saddle,
requested Ni TS or installed ECP saddle qualification follows from these Cs runs.
The full #2844--#2848 goal remains incomplete and both ECP saddle guards stay shut.
