# Candidate: restore the proved through-d long-range force call graph

Status: proposed; current-source qualification in progress
Date: 2026-10-24

## Decision and boundaries

Restore the through-d LR force specialization previously investigated in
`c7eec7ed780e2a72b449edf995c8041ec30806f7`, on the current isolated XC and
held-density-factor base. A host-owned maximum shell angular momentum at most
two proves total quartet order at most eight. The existing one-pass queue may
therefore omit unreachable orders nine through twelve from its device call
graph. Both restricted and unrestricted paths preserve omega, screening,
generated-shell selection, subtile ownership and force coefficients. Unknown
and f-shell metadata, and short-range exchange, keep the complete dispatcher.

The through-d initial stack request is capped at 16 KiB instead of 32 KiB.
The existing ensure operation only raises a context limit, never lowers one
belonging to another live owner. CUDA may still grow the stack to the linked
entry point's requirement; that actual linked resource must be measured.

## Rationale and prior evidence

The old generic LR force graph reached 52,352 bytes despite a through-d input.
The historical specialization reduced its linked frame to 23,016 bytes and
recovered enough memory for subsequent DF geometry preparation. These are
historical diagnostics, not performance claims for this source. The current
Stage27 WB96 DF pilot remains 491.602 seconds cold with explicit range caching;
its independently qualified warm endpoint is 44.409 seconds. Restoring the
held factor alone has not completed the cold target.

Do not replace this with a global stack-limit reduction or angular rescanning.
Do not import the separate complete-U budget policy without its own evidence.
The specialization changes no scientific settings or recurrence arithmetic.

## Required current evidence

The executable host selector test preserves exact forwarded arguments across
spin, angular bounds, full/SR/LR operators, custom omega and launch domains.
Real-device Direct and high-angular tests, independent fitted-WB composition,
sanitizers, linked resource reports and complete cold/warm/moved endpoints must
qualify the current built source. Historical timings cannot replace those gates.

## Current-source water96 evidence and larger boundary

Stage29 source `7dcfdc50eb2c0fcb47635e2094c9a00918f44e54` passes all
20 calls in five alternating native/GPU4 water96 WB DF pairs. Cold medians
are 330.262210 / 393.184624 seconds (16.0033% reduction); warm medians
45.183987 / 135.656251 seconds (66.6923%). Explicit validation GEMM and
range-resident-values controls are enabled. Moved geometry preserves rank160
and zero fitted-panel-call events across all eight force-response trees.
The full independent/reference, native regression and source/binary receipts
are retained in `benchmarks/results/fitted-range-cold-20261024/`.

The water108 endpoint fails before force allocation because the lazy derivative
lease still reserves an unused 128 MiB MD-J value allowance. This separate
capacity issue and its zero-sample failure remain visible; the dry-budget fix
is not yet a qualified larger endpoint. Do not generalize the water96 claim
or silently remove bounded fallbacks before that verification.
