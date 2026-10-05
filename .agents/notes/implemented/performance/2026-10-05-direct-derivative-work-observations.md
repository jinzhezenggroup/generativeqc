# Decision: retain class admissions separately from angular replay timings

Status: implemented (diagnostic evidence only)
Date: 2026-10-05

## Problem

The composed PBE0 exact-direct endpoint remains above the #1895 warm target.
Aggregate J/K derivative time does not identify whether the relevant work is
large admitted domains or expensive high-order derivative evaluation.

## Decision

Retain the existing optional shell-class observer's work ledger and a separate
intrusive thirteen-pass angular replay. Both use final native SCF densities
and compare their source-major derivatives against the normal production path.
Do not turn the rejected angular schedule into a production default on this
evidence. Keep its timings distinct from the production mixed-order traversal.

## Invariants

- J/K derivative counts represent union work from their shared recurrence.
- AO and primitive counts are admission upper bounds, before inner screening.
- Separate solves have separate histories; do not borrow counts across jobs.
- Observer replay agreement is not an independent numerical oracle.
- No endpoint or parity claim follows from these diagnostic timings.

## Evidence and consequences

Slurm 5857 retains original/moved 48/96 observations. Slurm 5859 retains the
angular replay and exact profiler event projections. At 96 atoms its orders
4–6 account for about 9.651 seconds across the separate diagnostic passes;
this motivates studying generated high-order consumers, but is not their
production time share. All replay errors pass `1e-9`.

The unsuccessful 5856 probe tried using a derivative-order-one PreparedFockPlan
as the SCF resident value provider. It lacked the required binding. The fixed
diagnostic separates SCF and force owners; its preparation/upload is explicit
and does not represent a production endpoint timing optimization.

## Revisit when

A shared compiler-generated force consumer can reuse expensive derivative
work while preserving separate J/K source weights and bounded fallback. Any
candidate still requires independent scientific gates and complete cold, warm,
and moved endpoint qualification before promotion.

## References

- #1895
- `benchmarks/results/pbe0-derivative-work-20261005/`
