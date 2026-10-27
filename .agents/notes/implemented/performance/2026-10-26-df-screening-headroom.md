# Decision: reserve DF shell-screening headroom during response planning

Status: implemented
Date: 2026-10-26

## Problem

The #2786 default-grid README campaign exposed a DF PBE0-96 cold endpoint
slower than GPU4PySCF despite earlier successful qualification. Identical native
source, binary, inputs and SCF work produced warm times near either 8.2 or
10.6 seconds on RTX 5090 cards. Live clocks were comparable. Lightweight
timers isolated the difference to the final Coulomb force response; native
traces showed optional whole-shell screening falling back for capacity.

The planner maximized response-panel width before admitting 1,189,120 bytes of
norms. A roughly 9 MB headroom difference changed the remainder enough to
disable screening. A 412-column panel reduced to 411 columns still needs nine
panels and restores screening. The problem was resource ordering, not a
different SCF stopping rule or intrinsic card throughput.

## Decision

Keep the chosen response algebra and budget. Before allocating workspace,
choose the largest tile that leaves space for norms, optional counters and the
existing logical reservation for a borrowed density. Account for owned density
uploaded since querying free device bytes. Binary search relies on workspace
size being nondecreasing in tile width for each retained algebra.

If even minimum scratch cannot coexist with optional storage, retain the
original strict tile. Physical allocation may still fail because of granularity
or another consumer; retain the existing strict fallback in that case. This
supersedes the norm-placement policy in the original whole-shell screening
qualification, without changing its numerical allowance or geometric bounds.

## Rejected alternatives

- Replacing the slow README sample with a fast-card result hides a reproducible
  planner cliff. The original full cohort and figures remain intact.
- Increasing the allowance or relaxing the screening target obscures the
  resource-ordering cause and changes a different contract.
- Nsight alone was insufficient: attaching it made the slow-card symptom
  disappear. Lightweight timers and native traces preserved the symptom;
  same-card complete-endpoint repeats established the benefit of the fix.

## Invariants and consequences

Norms remain per-force-call, geometry-dependent and arena-owned. Precision,
force target, grid, SCF convergence and output boundary stay unchanged.
Smaller panels may revisit auxiliary-shell fragments; counters from different
tiles need not have identical considered-work totals. Retain numerical gates,
actual executed/skipped work, resource ceilings and minimal-capacity fallback
tests instead of imposing a fixed private tile or call count.

GPU telemetry must use the active CUDA PCI identity, not an interpretation of
CUDA_VISIBLE_DEVICES as an NVML index. Instantaneous snapshots must describe
their actual collection boundary and cannot establish clocks during a call.

## Evidence and revisit conditions

The [retained investigation](../../../../benchmarks/results/df-screening-headroom-20261026/README.md)
contains all repeats, independent observable qualification, changed-geometry
and constrained-memory checks, work traces and incremental-build provenance.
On the affected card, cold median is 26.7521 → 24.3199 seconds versus 26.3067
for GPU4PySCF; warm is 10.5745 → 8.1910 seconds. The other card retains its
original speed. Revisit if a response algebra has nonmonotonic scratch, if
extra panels outweigh screening, or if storage ownership changes the mandatory
minimum or borrowed-density reservation.
