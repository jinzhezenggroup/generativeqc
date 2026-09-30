# Decision: retain bounded nonlocal force feature collection

Status: implemented on the #1557 branch
Date: 2026-09-29

## Problem

The single-pass force join requires the final CUDA KS rho/grad-rho binding.
Host-unfused WB97M-V remains a supported public energy/force schedule, but its
native owner does not retain that optional device binding. Requiring it
unconditionally turns a scheduling optimization into a force regression.

## Decision

Attempt the exact-token snapshot seed on the first live grid lease. Only a
native NOT_IMPLEMENTED capability response selects bounded feature collection.
The bridge validates the token before that response and has not submitted a
seed copy. Verify that the nonlocal owner is still empty before collecting.

The fallback collects device features alongside the first semilocal geometry
pass, finishes that accumulator, and consumes pair seeds in a second bounded
AO/grid pass. Both schedules use the same native pair owner, independent
stationary accumulators, stream checks and device geometry consumers. No
scientific equations or host feature/seed arrays are added.

## Rejected alternatives

Rejecting public host-unfused forces would unnecessarily remove existing
capability. Catching every error would hide stale tokens, device mismatches,
allocation failures and CUDA/numerical faults. Retaining full-grid AO jets
would remove the bounded tile-storage guarantee.

## Invariants

- Capability fallback is limited to the snapshot seed call, not pair execution.
- A failed or partially seeded owner cannot be reused for collection.
- Each geometry source is accumulated once; only AO collocation repeats.
- The default resident route still borrows each grid tile once.
- Fallback work reports two AO passes and its actual feature source; it does
  not claim snapshot D2D copy enqueues or a single pass.
- Host-wall intervals remain exclusive and additive on both schedules.
- Components are published only after both accumulators finish successfully.

## Evidence

The device-free join regressions reproduce the missing-feature failure and
cover collection order, partial tiles, repeat execution, invalid seeds,
error propagation, partial-owner rejection and exclusive timing. The real
CUDA RKS/UKS test compares both public schedules with GPU4PySCF and checks
cold/warm force consistency. That test requires the existing Slurm/CUDA opt-in;
its presence is not evidence of a completed NVIDIA run.

## Consequences and revisit

Host-unfused force composition retains two bounded passes without restoring
host feature transfers. Revisit removal of this fallback only after every
supported schedule exports a validated resident binding, or after an explicit
public capability change. The native scientific owners remain unchanged.

## References

- #1557
- #1482
- `tests/python/test_stationary_resident_nonlocal_join.py`
- `tests/python/test_wb97mv_host_unfused_force_fallback.py`
