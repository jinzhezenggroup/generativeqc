# Decision: retain a bounded complete WB97M-V force gate

Status: implemented
Date: 2026-11-04

## Problem and decision

Removing redundant finite-difference matrices must not remove the only live
complete WB97M-V reconverged-energy/force check. Component finiteness and force
translation invariance cannot detect every coupled SCF/nonlocal force error;
historical qualification does not test the next source revision.

Retain RKS water and UKS OH on one representative grid, with two predetermined
fine central-difference steps. Both steps must meet the existing 2e-6 absolute
directional-derivative tolerance; never accept only the best sample. Preserve
strict convergence, finite outputs, zero total force and the complete report.

The module remains on scheduled/manual CI, not routine PR/merge-group shards.
The previous two-spin/two-grid/three-step matrix made 28 complete public endpoint
requests; this two-spin/one-grid/two-step gate makes 10. These are explicit
endpoint invocation counts, not a measured internal SCF/kernel census or a
wall-clock speedup claim. Redundant coarse-step/grid combinations are omitted,
not the independent complete-endpoint gate.

## Revisit when

A separately validated complete independent force oracle covers both admitted
spin domains, or measured endpoint timing justifies another representative grid.
Do not substitute archived receipts or component smoke tests for this live gate.
