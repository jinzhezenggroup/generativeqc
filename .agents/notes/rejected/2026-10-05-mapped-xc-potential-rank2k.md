# Decision: retain generated XC potential on the measured local AO domains

Status: rejected (production promotion, not the shared indexed candidate)
Date: 2026-10-05

## Problem

The earlier dense 384/768-AO crossover (#1958) used synthetic full-AO panels.
Actual PBE0 water grids use local AO maps, so the dense 768-AO win does not
establish the profitability of compact rank-2k plus indexed publication.
The shared indexed candidate from #1963 makes this domain executable without
restoring screened-out global AO work or introducing method-local vendor calls.

## Decision and evidence

Retain generated execution as the production incumbent. Measure both providers
on the retained 48/96-atom original and moved geometries, spherical def2-SVP,
the full v1 48×16×32 per-atom grid, and tiles of 128/256/512 points. All 12 paired
warm endpoint medians favor generated execution. At the production tile 256,
the original-geometry rank-2k routes take about 24.6% and 21.2% longer for
384 and 768 AOs. The structured receipt contains every cold/warm sample,
preparation time, both geometries and exact errors:
`benchmarks/results/xc-mapped-potential-1876/`.

These are complete ordinary-stream fixed-density XC endpoints, including the
input transfer, compact panel construction, potential publication, point totals,
synchronization and E/V downloads. The positive diagnostic density is not a
converged SCF state. The results cannot be reported as full PBE0 SCF/force
performance. Basis/grid preparation, AO discovery and provider preparation are
reported separately. No CPU/reference work enters the measured device path.

The two routes have identical maps and symmetric summands. Both geometries
reproduce the retained tile-256 census. Compact output writes and scatter
matrix/index accesses are counted explicitly; logical access counts are not
measured DRAM traffic. Cache capacity is conservatively charged at global AO
extent, with a separate 96 MiB provider allowance. This receipt measures their
combined endpoint cost and does not assign the regression to a specific kernel.

All 144 evaluations pass paired E/V/population gates. The large-system comparator
is generated execution, not an independent oracle. The independent scalar,
CPU local/discovered-map, capture and resource tests belong to #1963, and its
native gate passes in the benchmark binary before timing. That distinction must
survive future summaries of the evidence.

## Rejected alternatives

Do not transfer dense endpoint qualification to the indexed materialization,
restore global AO panels, or hide scatter/storage costs behind a kernel-only
claim. Do not promote a tile-size change from fixed-density component timing:
changed domains and composed SCF/force behavior need separate qualification.

## Revisit when

A different shared recipe demonstrably reduces total compact/scatter work, a
larger actual active domain is measured, or complete endpoint evidence establishes
a stable crossover. Keep the indexed candidate and bounded fallback available
for such qualification. Small-map negative results do not prove that every
future dense or indexed rank-2k domain will lose.

References: #1876, #1886, #1958, #1962, #1963; the implementation decision is
`.agents/notes/implemented/performance/2026-10-05-indexed-xc-potential-provider.md`.
