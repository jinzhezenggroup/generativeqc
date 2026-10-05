# Decision: keep stable owner grouping an explicit, bounded experiment

Status: correctness-qualified opt-in experiment; complete endpoint promotion rejected
Date: 2026-10-05

## Problem

The audited Slurm 6090 capture records 2.675511 aggregate kernel seconds in the
combined XC/AO/atom-gather/owner consumer. It does not isolate atom gather. The
original ordered scan has 37,261,541,376 source-domain membership tests on the
96-atom public-force invocation. Removing an already small density gather or
panel copy cannot explain a seconds-scale improvement.

## Decision

The compiler-emitted cooperative kernel accepts an explicit schedule: 0 retains
the production scan, 1 requests stable integer grouping, 2 requests independent
owner-coordinate writers, and 3 combines both. The private native diagnostic
`stationary_configure_geometry_ao_v1` bounds the schedule and rejects changes
after topology installation. Public Python execution does not select this API;
the production default remains 0. Qualification helpers explicitly configure
owners before topology, rather than adding a hidden environment-dependent policy.

Grouping appends local AO positions to per-atom head/tail/next chains in original
order, including arbitrary AO maps and noncontiguous atom labels. Three owner
coordinate writers keep the same AO order for each independent coordinate.
Neither uses floating atomics, associative reductions, new screening, retained
scientific caches, or another global AO inventory.

Integer metadata aliases the already charged Becke pair-state storage only
before its Becke use. The existing XC value and ordered AO gradient panel remain
disjoint from the optional metadata. A nonfitting metadata request keeps the
existing cooperative scan; it does not increase allocation or downgrade a
fitting cooperative AO panel to scalar execution. Every reader completes before
Becke overwrites this storage. Invalid indices are checked before chain writes,
and the original sticky producer/collective error gate remains in place.

## Qualification

Host thread emulation covers empty/tiny/full panels, reversed maps, interleaved
labels, both cooperative and scalar routes, integer-scratch fallback, tails,
external seeds, cached/direct centers, sticky failures and allocation canaries.
The initial combined host run passes 15 tests. Compiler generation and ABI
checks pass 32 tests; the real-device suite is explicitly skipped off Slurm.

The expanded real-device owner gate freezes cancellation-sensitive AO/work
inputs and checks all output source bits against schedule 0 independently for
bounded and phased Becke execution. Cross-Becke algorithm comparisons retain
their separate numerical tolerance. Full, shuffled subset and empty maps include
48/384, 96/768, and cooperative-but-grouping-nonfitting 96/900 domains. Unknown
schedules and post-topology configuration are negative gates.

The artifact event probe is intrusive and times the enqueued geometry/publication
pipeline, not pure atom gather or complete molecular E+F. These times must never
be used as clean endpoint evidence. Before promotion, require independent
LDA/PBE/R2SCAN RKS/UKS force gates, real memcheck/racecheck, and clean paired
48/96-atom PBE0 endpoints with actual cold and moved SCF histories. Gate passing
and fewer source membership tests do not complete #1893 P0-C.

## References

- Proposal: `../../proposed/2026-10-05-stable-grid-owner-gather.md`.
- Audited capture: `2026-10-05-force-ao-grid-census.md`.
- Frozen experimental helpers/receipts: `.artifacts/stable-group/`.

## Fixed-state device evidence

Slurm `srun` 6113 exits 0 on one n1 RTX 5090. All 63 expanded owner tests pass,
including all four schedules, exact same-state source bits, shared-space
fallback, geometry changes, invalid-input recovery and negative configuration.
The frozen scientific/native identity is
`9fd0a78b20ebbd85a0a2b1d296495780dd134f8f2b6e6a3e4029389d617d14b0`.
Latest combined host qualification passes 44 tests with 63 explicit device
opt-in skips; the compiler structure check covers 444 modules with zero errors.

The intrusive event probe's combined schedule-3 phased pipeline totals are
2.6503 vs 3.1646 ms for synthetic 48/384 full input, and 4.2929 vs 5.7376 ms for
96/768 full input. The 96/900 optional-scratch fallback retains the scan and
observes 5.9503 vs 6.2527 ms. These are six measured enqueues per case across
three geometries, executed in fixed schedule order. They are neither pure
gather timings nor counterbalanced clean endpoint speedups. They select
schedule 3 for the next experiment, not a production default.

Receipts: `.artifacts/n1-owner-probe-6113/`. The completed s/p/d library is kept
with that frozen probe; the expanded AOT package uses a separate n1
`qc-1893-p0c-owner-v7-aot-20261005` snapshot and the same scientific identity.

## Complete qualification and rejection

All three owned jobs terminate with exit 0: srun 6113 (s/p/d fixed-state probe),
6114 (expanded package gates), and 6115 (complete endpoint pairing). Job 6114
repeats all 63 device owner gates against the regular packaged entry, passes
12 independent LDA/PBE/R2SCAN RKS/UKS and public PBE0 force tests with 26
explicitly configured owners, and passes two high-occupancy/fallback cases each
under real memcheck and racecheck. There are zero memory errors and zero race
errors/warnings. These are correctness gates, not endpoint performance proof.

Job 6115 uses one GPU/allocation, five warm and five changed-geometry warm
repeats, candidate-first at 48 atoms and control-first at 96. Each variant passes
72 same-geometry numerical/reference pairings per size. All 12 actual AO/jet
source-domain records per size are exactly equal between candidate and v6.
The experiment removes repeated *membership scans*, not AO evaluations, jets,
density projection FMA pairs or the remaining restricted beta-panel copy.

| Atoms | Route | Cold s / Focks | Warm s | Moved s / Focks | Moved-warm s |
| --- | --- | --- | --- | --- | --- |
| 48 | v6 control | 100.571776 / 25 | 8.951451 | 51.403380 / 12 | 8.939826 |
| 48 | schedule 3 | 96.954515 / 24 | 8.840963 | 54.625284 / 13 | 8.833676 |
| 96 | v6 control | 234.546259 / 29 | 24.775303 | 129.035027 / 14 | 24.877414 |
| 96 | schedule 3 | 220.537262 / 27 | 24.053450 | 121.401496 / 13 | 24.142217 |

Warm changes are -1.2343% and -2.9136%; moved-warm changes are -1.1874% and
-2.9553%. The 48-atom moved endpoint is **6.2679% slower with 13 vs 12 Focks**.
Reject production promotion on that observed regression and the still modest
reference-gap closure. Do not normalize Fock counts, combine allocations,
attribute different SCF histories to integer grouping, or call P0-C complete.
The native maximum E/F errors over both variants/sizes are 1.078e-10/2.558e-11.

The public endpoints actually load JIT stationary runtimes, not the separately
qualified full-primitive packages. Their v6/schedule-3 binary hashes are
`00054bd3d26bcb29a529bfb64d6b90c6e4855df64b236e41c34ec6a1ff66181b`
and `bab61d31519b091e007f6409a007f8fdec51b54f67490574850c1d8b24132a29`.
The endpoint selector configures only during owner construction; steady-state
native calls are not patched or event-profiled. Actual-resource static dumps show
255 registers for both cooperative entries, with stack declarations 72/168
bytes and 1040 static shared bytes. These are not achieved occupancy or spill
measurements, and do not by themselves prove a default-route slowdown.

Receipts: `.artifacts/n1-owner-gates-6114/`,
`.artifacts/n1-owner-endpoints-6115/paired-clean-summary.json`, and
`.artifacts/stable-group/static-resources/`. The next distinct hypothesis is
`../../proposed/2026-10-05-vectorized-stationary-point-preparation.md`; it is not
yet implemented or accepted. The public production schedule remains 0.

## Current-master PR integration boundary

The GPU receipts above qualify the preserved scientific/native identity, not the
later current-master transplant. The separate
[integration decision](../compatibility/2026-10-06-indexed-ao-grid-master-integration.md)
records the retained shared-lowering behavior, host fixture reconciliation and
pending provenance-matched native/GPU requalification. This integration does not
change the experiment's promotion rejection or complete P0-C.
