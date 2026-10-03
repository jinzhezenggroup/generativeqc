# Experimental CUDA kernel timing

`generativeqc_compiler.common.cuda_time_estimator` turns compiler-visible work
counts into an engineering time estimate using **caller-supplied measured rates**.
It is an offline tool: it does not import the runtime, compile code, or probe CUDA.
It does not select production defaults. No validated GPU calibration is shipped.

The existing `common.cuda_cost_model` and `tools/analyze_cuda_cost.py` screening
reports keep their relative, non-timing contract. Timing is an optional layer.

## Calibration contract

`CudaTimingCalibration` requires all of the following:

| Field | Meaning |
| --- | --- |
| `device`, `architecture`, `sm_count` | Concrete device/SKU identity, canonical `sm_XX` target, and measured device topology |
| `workload` | Kernel family, arithmetic precision, operation-count convention, and traffic/cache regime |
| `provenance` | Retained calibration measurements and their software, clock/power, and timing conditions |
| `effective_compute_ops_per_second` | Achieved whole-device operation rate in the stated workload regime |
| `effective_memory_bytes_per_second` | Achieved bandwidth using the same byte-count convention as the candidate |
| `launch_seconds` | Positive incremental latency per serial launch, measured separately from body work |
| `saturation_occupancy` | Explicit occupancy threshold in `(0, 1]` at which both rates are assumed to saturate |
| `uncertainty_fraction` | Explicit engineering fraction in `[0, 1]`; not a statistical confidence level |

Peak specification rates are not calibration. FP32, FP64, tensor-core, and
special-function operation counts are not interchangeable. Likewise, semantic
bytes and observed DRAM bytes must not be mixed unless the achieved bandwidth
was calibrated with that same convention. Calibration workloads must avoid
counting launch latency twice in the body rates.

The estimator rejects architecture and known SM-count mismatches. Matching these
fields cannot prove the GPU SKU, precision, cache regime, or kernel family matches;
that remains the caller's responsibility. `workload` and `provenance` are required
labels, not automatic verification of their contents.

## Work and formula

One `StaticCudaCost` must describe one homogeneous kernel, possibly repeated with
the **same grid and resource shape**. Arithmetic operations, semantic bytes, and
explicit dynamic spill bytes are totals across all those launches. `grid_blocks`
is the grid size **per launch**. Work is not multiplied by `launch_count` again.

```text
parallel_scale = min(1, device_occupancy_upper_bound / saturation_occupancy)
compute_seconds = total_operations / achieved_compute_rate / parallel_scale
memory_seconds = (total_semantic_bytes + total_dynamic_spill_bytes)
                 / achieved_memory_rate / parallel_scale
launch_seconds = launch_count * calibrated_launch_seconds
estimated_seconds = max(compute_seconds, memory_seconds) + launch_seconds
interval = estimated_seconds * (1 +/- uncertainty_fraction)
```

This linear correction is a disclosed modeling assumption, not a fitted law.
The static occupancy bound is optimistic and can omit unknown resource limits;
all source diagnostics remain attached to the timing report. The estimate does
not resolve instruction dependencies, tail-wave scheduling, cache effects,
latency hiding, or concurrent kernels. Its interval is not a guaranteed bound on
real execution time. `bottleneck` can be `compute`, `memory`, `balanced`, `launch`,
or `none` for a known no-op; unknown estimates have no bottleneck classification.

PTXAS spill bytes are static resource evidence. They do not count executions
across threads, loop iterations, or launches and cannot be added to total semantic
traffic. If the compiler reports zero spills, the estimator assumes zero dynamic
spill traffic. Otherwise the caller must pass total `spill_traffic_bytes`, possibly
an explicit zero when those bytes are already included in semantic traffic.
Other traffic, such as non-spill local-memory accesses, belongs in semantic work.

Missing operations, bytes, launches, dynamic spill evidence, or whole-device
parallelism yield `None` for the total and interval. Independent known components
remain available. `allow_per_sm_fallback=True` explicitly permits an optimistic
estimate without global underfill evidence and is recorded in the report.
A zero grid or impossible resident-block count never becomes executable through
that fallback. A known zero-launch, zero-work request returns zero; zero launches
with nonzero work are rejected. Invalid numeric inputs raise, and overflowing
computed seconds remain unknown instead of emitting infinity or NaN.

## Offline CLI

Serialize a measured calibration with `CudaTimingCalibration.to_payload()` to
obtain a `generativeqc.compiler.cuda-timing-calibration.v1` JSON object, then run:

```bash
PYTHONPATH=python python tools/analyze_cuda_cost.py \
  --arch sm_120 --block-threads 128 --grid-blocks 680 \
  --ptxas retained-single-kernel.ptxas.log \
  --operations 1000000000 --traffic-bytes 100000000 --launches 2 \
  --calibration measured-device-calibration.json
```

The counts above illustrate the interface, not measurements. Calibration supplies
SM count when `--sm-count` is absent. `--spill-traffic-bytes` supplies dynamic spill
work; `--allow-per-sm-fallback` opts into the fallback. Both require calibration.
Without `--calibration`, the CLI emits the existing static screening report.
With it, `time_estimate` adds the model identifier, calibration, static evidence,
parallelism basis/scale, component times, engineering band, and diagnostics.
A timing report rejects multi-kernel PTXAS logs: resource maxima across different
kernels are only useful for screening. Headerless single-kernel logs retain an
explicit unverified-architecture diagnostic.

## Endpoint boundary and qualification

The API does not compose an SCF or force endpoint. Estimate heterogeneous serial
kernels separately and add their times: `sum(max(compute_i, memory_i))` differs
from `max(sum(compute_i), sum(memory_i))`. Concurrent work needs a separate overlap
model; summing component times does not establish wall time in that case.

An endpoint consumer must explicitly account for compilation, host setup, plan
construction, grid/geometry rebuild, transfers/synchronization, iterations, and
force/response work. Cold, warm, and changed-geometry scenarios must record which
of those stages execute or reuse state. Iteration counts and cache invalidation
are method/runtime policy, not device calibration. Unknown stages cannot be
silently treated as zero.

Before relying on estimates for decisions, retain held-out measurements spanning
kernel families, sizes, underfilled grids, occupancy/resource pressure, and cache
regimes; record prediction residuals separately from the engineering band. Any
production promotion still follows the complete-endpoint, work-count, and
independent numerical gates in
[Performance engineering](../maintainer/performance_engineering.md).
Nonlinear corrections, hierarchical calibration, residual quantiles, and learned
residual models require such evidence before becoming automatic behavior.

The scope and rejected alternatives are preserved in the
[calibrated timing decision](../../.agents/notes/implemented/performance/2026-10-03-calibrated-cuda-kernel-timing.md).
