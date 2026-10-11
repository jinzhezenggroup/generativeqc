# Decision: bounded default-auto Global-Hybrid source reuse

Status: implemented
Date: 2026-10-11

## Problem

The screened phase source and fixed-density warm-energy handoff now support
complete PBE0/def2-SVP E+F for 96 atoms without changing the FP64 mathematical
contract. Requiring every caller to set an execution-only environment control
prevents otherwise eligible native cold guesses from using this shared owner.

## Decision

Unset `GENERATIVEQC_KS_RESIDENT_VALUES` uses the existing `auto` parser and
admission policy, not forced `1`. Explicit `0` restores the incumbent schedule.
There is no change to the cold-domain, value ceilings, complete resource
admission, live-memory reserve, source lifetime or capability checks. Warm and
external-seeded solves refuse setup before provider cloning and row sorting.
Native cold guesses carry explicit provenance; pointer presence is not a seed
identity. Public finite resource ledgers still refuse optional quartic storage.

The preceding opt-in decision remains historical in
[screened cold auto](2026-10-11-screened-hybrid-cold-auto.md). The separate
[warm-energy handoff](2026-10-11-hybrid-warm-energy-handoff.md) explains why one
incumbent fixed-density action is charged before publishing a warm baseline.
Neither final resident density/energy nor convergence tolerances are altered.

## Rejected alternatives

- Forced-on default: risks setup on warm/seeded or unprofitable small solves.
- Removing memory admission or public-ledger refusal: would silently bypass the
  complete consumer plan and future gradient owners.
- Reusing the resident terminal energy as an incumbent warm baseline: fails the
  unchanged 1e-12 Eh energy gate because reduction schedules differ slightly.
- Claiming universal speedup from the 96-atom case: eligibility is a bounded
  resource heuristic, not predicted SCF reuse or broad crossover evidence.

## Invariants and evidence

Preserve strict FP64, frozen geometry-only Schwarz extents, signed raw J/K
parity, source finite/census audits and graph invalidation before owner release.
Charge source construction and the warm handoff to complete endpoint work.
Qualification must genuinely unset the control, not relabel explicit `auto`.
Retain all ABBA samples, independent energy/force gates and physical Fock counts;
different moved-geometry iteration counts must not become kernel-speed claims.

Host policy and mutation guards cover unset/auto/0/1 and exclusions. Native
probes cover default cold/warm/external seeds and exact budget fallback.
Source-matched endpoint receipts are retained under
`benchmarks/results/hybrid-phase-source-default-20261011/` after qualification.

## Revisit when

Broaden hardware, resource-cliff and rapidly converged cold coverage before
widening eligibility. Reserve optional source storage in the complete public
planner before admitting finite resource-ledger requests. Retain explicit
rollback even when additional endpoints qualify.
