# Decision: resolve Direct J and K fallback independently

Status: implemented; native qualification pending
Date: 2026-10-05

## Problem

The resident Direct provider selected generated J and K separately only after
an outer whole-request fallback gate. A missing generated range-K operator sent
both channels to canonical execution, even when full-range generated J was
available. A recurrence-only mixed-J request also excluded generated and
canonical strict K. Thus the arithmetic or coverage of J changed K's execution
algorithm despite unchanged scientific exchange identity.

## Decision

Resolve one route per requested channel from retained source capabilities.
Strict generated coverage has priority, followed by optional canonical storage
and the existing bounded public-AO fallback. Mixed J retains the qualified FP32
recurrence / FP64 product-and-accumulation consumer; K remains strict FP64 and
chooses its own route. Absent channels select no route. The dispatcher composes
these consumers on the existing provider stream and validates every requested
output after production. It introduces no recurrence or method-name selector.

This supersedes the mixed-J whole-request fallback retained in
[the original generated-J split](../performance/2026-09-25-generated-j-hybrid-k-split.md).
Canonical J/K still share a traversal when both choose that source. A split
range request can require another density transform for generated J; its
profitability must be measured rather than inferred from fewer canonical radial
evaluations.

## Observation and qualification

Port the optional generated J/K task-entry census from the earlier incremental
experiment (commit `80fa2844c`), without its incremental or density-screening
changes. Counters observe admitted streaming shell tasks, including native
DDDD. They do not observe rejected candidates, primitive recurrences, or bounded
higher-angular classes. Null pointers retain ordinary execution. Observers own
zeroing, charged storage and stream-ordered lifetimes; intrusive census timings
are not clean endpoint evidence.

Host gates exhaust independent capability/request/precision/fallback masks and
require exactly one source per requested channel. Native gates compare strict K
and mixed J with independent CPU ERIs for s/p/d/f, signed nonsymmetric spin
densities and both public AO representations. The SPD range fixture verifies
that generated J actually executes while canonical K evaluates only one radial
source per admitted quartet; its constrained-capacity fallback remains covered.
Native execution, sanitizers and complete-endpoint evidence are still pending.

## Boundaries and follow-up

This is the dispatch prerequisite for #1892, not completion of its independently
prepared CoulombJPlan/ExchangeKPlan schedules or 48/96 profiling requirements.
The strict full-range composed PBE0 baseline already uses generated J and K;
this patch alone does not predict a speedup for that baseline. No screening,
SCF closure, force coefficients, precision admission, or resource limits change.
Independent worker/order/density policies and endpoint profitability remain
separate qualification work under #1892 and #1895.
