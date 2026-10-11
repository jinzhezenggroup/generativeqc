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
`benchmarks/results/hybrid-phase-source-default-20261011/`.

The clean measured revision is `1cef1189224c6ddcb29d684f9ef7eaa0aa0e34f8`,
rebased onto master `637466c624c4c4c4dd4eb7c511a2c88675744325` including the
resident final-state proof and native force-owner changes. Jobs 7338/7340 run
two actual-unset ABBA cohorts on separate RTX 5090 UUIDs. All 32 water-96 and
16 generated-small complete E+F oracle gates pass; maximum large energy/force
errors are 9.5497e-12 Eh and 3.8277e-11 Eh/Bohr. Every automatic cold phase
constructs 1,873,221,716 values, audits 17 source Focks and charges one handoff;
every warm phase has one physical Fock without source preparation.

Cold complete medians are 97.8693 s disabled and 76.2461 s unset. Baseline
cold work varies across 17/19/21/22 Focks, while default work is consistently
17 resident actions plus one incumbent handoff. This is a descriptive endpoint
ratio, not a work-matched integral-kernel gain. Warm/moved/moved-warm work is
1/12/1 in every arm. The resource heuristic is still not a profitability model.

Job 7345 passes four independent signed J/K and same-grid PBE0/B3LYP probes,
including default admission/refusal and chunk widths 1/2, plus three independent
RHF original/displaced tests for the shared owner. Job 7347 reports zero
memcheck errors. Final host suites pass 994 and 336 cases; four unrelated
offline GFN1-D3-asset cases remain explicitly deselected. Initial fixture-only
missing-LibXC and missing-SONAME-alias failures are retained separately, not
deleted or pooled with accepted measurements. Production source/binary bytes
remain unchanged when those fixture omissions are corrected.

## Revisit when

Broaden hardware, resource-cliff and rapidly converged cold coverage before
widening eligibility. Reserve optional source storage in the complete public
planner before admitting finite resource-ledger requests. Retain explicit
rollback even when additional endpoints qualify.
