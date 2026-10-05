# Decision: preserve local AO domains in prepared symmetric potential assembly

Status: implemented
Date: 2026-10-05

## Problem

The actual 384/768-AO PBE0 baseline uses local AO maps, with maximum active
extents 342/536. Dense rank-2k qualification does not apply to that domain:
local panels must update an indexed global destination. The dense provider
therefore retained generated execution for every mapped invocation.

## Decision

Add a separately qualified indexed candidate to the same canonical symmetric
cross-product request. Existing compiler-owned compact panels continue to own
all functional coefficients. The provider computes a compact authoritative
triangle with DSYR2K, then scatters/mirrors it into the selected global entries.
This changes materialization and execution identity, not the scientific graph,
precision or local AO domain. Dense qualification cannot select this candidate.

One explicit cache is bounded by `spins * global_nao^2 * sizeof(double)`, in
addition to the existing 96 MiB opaque provider allowance and 16 KiB host
binding reservation. It is conservative capacity, not a claim of maximum-active
sizing. Every tile reuses the same handle/cache with compact spin strides.
DSYR2K overwrites the temporary triangle (`beta=0`); accumulation belongs only
to the mapped global destination. Unmapped entries remain untouched.

The numeric cache uses the shared allocation ledger. Device/ledger exhaustion
can retain generated execution; host ownership-bookkeeping failure propagates.
Completely empty domains retain no optional device storage. A dense prepared
owner still falls back on mapped invocations; callers discover maps before
admitting the indexed recipe. Production remains generated until measured
complete-endpoint evidence qualifies a replacement.

## Invariants

Sorted unique validated IDs give each scatter thread exclusive ownership of
both symmetric entries. Tiles are ordered on one stream, so scatter needs no
floating atomics. All global matrix initialization remains with the enclosing
XC owner. Both finite publication and sticky error state survive scatter, and
compact scratch is never read as a previous tile's accumulator.

Provider execution neither allocates nor selects an implementation. Graph
capture records the same work; physical publication owns accounting. Empty
maps use the existing generated zero-domain operation and deterministic totals.

## Qualification

The `--indexed-potential` native gate uses an independent long-double bilinear
oracle for one/two spins, active extents 1/17 within larger global outputs,
short/full reduction panels, overwrite/accumulate, and three captured replays
with changed operands and valid noncontiguous indices. It checks every global
entry, including untouched entries, alias rejection and nonfinite injection.

Seven XC routes exercise unqualified/qualified indexed recipes, missing cache
budget, unavailable provider, dense-only qualification, completely empty maps,
and actual numeric-ledger exhaustion. Independent CPU local-map WB97M-V E/V
checks changed density. Full-map LDA/PBE/meta-GGA, spherical/Cartesian, nonlocal
and capture regressions run with indexed preparation. Discovered sparse maps
also compare scaled-PBE RKS/UKS with independent full-AO CPU E/V at `1e-10`.

The build/qualification scripts, complete source manifest, ccache commands and
statistics, binary hash, native regression and memcheck/initcheck evidence are
retained in ignored `.artifacts/1876-indexed-potential/`. All native/GPU work
uses n1/node1 through finite Slurm `main/gpu:5090:1` allocations.

## Rejected alternatives and revisit conditions

Restoring global dense panels would reintroduce screened work. Uncharged output
scratch would violate the prepared-resource boundary. Neither is an acceptable
way to reach a library. Next qualification must include compact output traffic,
scatter, point totals and complete fixed-density XC on actual mapped grids,
followed by composed SCF/force endpoints before any production promotion.

References: #1876, #1886, #1953, #1958, #1961, #1962.
