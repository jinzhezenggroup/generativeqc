# Decision: account for actual CUDA launch resources before ranking schedules

Status: implemented; analytical resource validation, not performance promotion
Date: 2026-10-06

## Problem

`compiled_gpu_profitability` counted static PTXAS shared storage but had no
input for dynamic launch storage, a kernel's granted dynamic limit, its thread
cap, driver-reserved shared bytes or the launch grid. A high per-SM bound can
therefore coexist with a one-block grid or an unlaunchable dynamic-memory request.
The stationary Becke discussion exposed this blind spot. Its 606-point-lane
calculation also applies only when tile size and the remaining owner budget admit
that many lanes; it must not be described as a measured default-kernel occupancy.

## Decision

Keep one pure launch-assessment module with explicit device facts. Account for
static plus dynamic storage and charge driver reservation separately to residency.
Require an explicitly granted kernel limit for dynamic shared-memory opt-in.
Do not reinterpret a static 64-KiB array as a legal dynamic allocation merely by
raising the tuner's 48-KiB ceiling. Count partial warps as complete warp slots.
Expose separate per-SM and finite-grid whole-device upper bounds. Unknown SM
count remains unknown; the compiled-profitability adapter rejects a grid-aware
request without a probed SM count instead of ignoring its grid silently.

Existing callers need not supply launch metadata. Callers that do must use one
consistent launch contract per invocation. Device/kernel/SM facts belong to the
runtime; this compiler module never probes CUDA or imports the public runtime.
CUDA's occupancy API and measured achieved occupancy remain distinct from this
analytical model; register/shared allocation granularity is not guessed.

## Evidence and boundaries

The independent pure-Python assessment suite passes 43 cases locally, covering
static/dynamic separation, missing opt-in permission, driver reservation,
partial warps, small/empty grids, unknown SM count, kernel caps, and malformed
counts. Python syntax checks cover the adapter. The local environment has no
NVCC, CUDA GPU or ccache; no CUDA/C++ compilation, full repository suite or new
endpoint performance result is claimed. The existing native source and CUDA
execution defaults are unchanged.

## Related work

Refs #1892, #1893, #1894, #1978 and #2026. Direct-force static-128 promotion is
already owned by #2026 and must not be duplicated. Joint Becke/tile/matrix tuning
must record actual selected routes, registers, dynamic storage, grid extent,
independent E/F gates and clean complete endpoints before promoting a winner.

Agent: ChatGPT
Model: GPT-6 Astra Pro
