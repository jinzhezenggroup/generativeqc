# Decision: explicit CUDA force tiles have independent admission

Status: implemented; automatic default selection remains unpromoted
Date: 2026-10-31

## Problem

SCF and stationary-force grid consumers shared `KsOptions.tile_points`, despite
different launch overhead, geometry scratch and retained-response lifetimes.
Force-only experiments at 1024 points improved complete README endpoints while
keeping the SCF tile at 256 and the production geometry scratch cap at 16 MiB.
Increasing both tiles would confound that result and replace explicit SCF policy.

The earlier 2048-point/64-MiB experiment also exposed a warm-only DF response
fallback cliff. The metric-workspace fix removed the demonstrated cliff, but a
grid-only memory test still cannot authorize a universally faster default.

## Decision

Expose an optional `KsOptions.force_tile_points` bound for CUDA stationary
consumers. Native dry planning, fleet reservation, prepared-item admission and
controller construction all resolve it through the same internal accessor.
Zero in C/C++ and `None` in Python preserve native inheritance from the SCF tile;
generated CUDA fallback owners preserve their incumbent planner unless the new
control is explicit. CPU execution continues to use its existing SCF tile.

The bound changes batching only. It leaves the geometry lane/scratch caps,
quadrature, precision, screening, equations, default payload and SCF tile intact.
Explicit requests still undergo complete force-arena and nonlocal-budget
admission. Cached and minimum consumers remain available. No GPU name, method
name, environment override or measured timing is consulted to choose the tile.

The pre-release native descriptor grows directly; C, C++, ctypes and private
precision/nonlocal extensions migrate together. Do not mix an old native library
with the new Python descriptor, or retain a compatibility alias with ambiguous
extension offsets. Exact size/offset gates and catalog publication controls
cover the current layout.

## Evidence and limits

The retained [force batching campaign](../../../../benchmarks/results/independent-force-tiles-20261031/README.md)
separates instrumented pilots, public API runs and fresh GPU4PySCF observations.
Every timing population retains cold and warm complete endpoints, SCF/Fock work,
source/binary/scheduler identities, and unchanged independent observable gates.
Moved-force finite differences, repeated calls, planned/prepared capacity parity
and minimum-budget refusal exercise the actual public request separately.

The 78 pilot populations qualified on A100, RTX 5090 and RTX PRO 6000. These are
sequential controls, not default-promotion evidence. Some A100 methods remain
slower than the fresh GPU4PySCF controls; neither the option nor pilot successes
establish an all-README win or qualification on unmeasured devices.

## Revisit when

An automatic policy jointly accounts for grid scratch, retained density/factors,
projection leases and response fallback work across cold, warm and moved calls.
It must preserve explicit requests and authenticate the same selected capacity
at planning and execution. See the earlier
[stationary batching investigation](../../proposed/2026-10-31-stationary-grid-batching.md)
and [metric-workspace decision](2026-10-31-occupied-response-metric-workspace.md).
