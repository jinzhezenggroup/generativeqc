# Decision: use the molecular VV10 pair contract in the native force controller

Status: implemented
Date: 2026-10-31

## Problem

The native stationary controller applied the molecular density domain but created
its VV10 layout with the raw-row default. The standalone resident force owner
already requested the molecular layout. As a result, native forces unnecessarily
traversed density-screened rows and could not select the qualified admitted or
parallel pair schedules. An A100 DF96 diagnostic measured 14.677 seconds in the
VV10 force-pair operation out of 29.472 seconds in the complete stationary force.
These synchronized component traces are diagnostic, not endpoint ranking data.

## Decision and invariants

Both capacity planning and construction now request the existing molecular
force layout. Its producer marks density-screened rows with negative zero.
Positive-zero and finite negative quadrature weights remain observable through
potential and weight derivatives; they must not be erased. Invalid inputs still
poison publication. The density threshold, quadratures, FP64 arithmetic,
self-consistent state and numerical acceptance gates are unchanged.

The existing dispatcher is based on work extent, numerical admission and
nonoverlapping live outputs. Large admitted grids may use parallel packets;
small or rejected domains retain ordered execution. No GPU-model whitelist,
new allocation or enlarged memory allowance is introduced. Raw fixed-grid
consumers retain their original contract.

## Validation infrastructure

The phased molecular test previously derived a public method name by replacing
underscores in a compiler profile. This produces invalid aliases such as
`wb97mv-rks`. The test now constructs the exact profile MethodIR, retaining its
spin, moved-geometry energy-difference gates and bounded-capacity fallback.
The VV10 device probes use the existing verified compiler-cache runner and
recognize real Slurm or LSF allocations without fabricated scheduler IDs.

The source-bound Libxc inventory is refreshed mechanically. Every capability
identity and stage is unchanged; historical native evidence remains stale and
no native qualification is inferred from this curated WB97M-V campaign.

## Evidence and limitations

The associated benchmark bundle retains full-endpoint populations, independent
oracle gates, source/binary identities and controls. This integration fix does
not establish parity for the remaining README functionals or optimal scheduling
on unmeasured GPUs. Native and GPU4PySCF WB97M-V DF retain different long-range
fitting contracts, so their default-configuration latency is not a matched-DF
algorithm comparison.

An initial A100 source overlay retained timestamps older than the compiled
controller object. Incremental linking left the exact diagnostic baseline
library hash unchanged. The binary-identity audit caught this; those receipts
are retained separately as rejected candidate evidence. Touching the transferred
inputs and rerunning CMake produced a new identity and recompiled the controller
before the accepted tests and measurements. Never infer executed changes from
successful linking or source-file hashes alone.

The [evidence bundle](../../../../benchmarks/results/native-vv10-molecular-20261031/README.md)
contains 24 independently qualified complete populations across A100, RTX 5090
and RTX PRO 6000, plus separate cross-binary ABBA controls. DF96 matrix warm
medians improve from 31.930 to 25.939 seconds, 94.582 to 60.705 seconds, and
73.361 to 45.233 seconds respectively. Each device passes 99 VV10 pair/lifecycle
tests and two moved-geometry RKS/UKS force/fallback tests. Cold times and the
remaining A100 DF48/Direct48/Direct96 gaps remain visible.

The global 2048-point/64-MiB force-grid experiment remains unpromoted.
The [follow-up investigation](../../proposed/2026-10-31-stationary-grid-batching.md)
identifies insufficient response headroom and 4,489 repeated fitted-panel
GEMMs as the cause of the large warm regression. Larger grid tiles themselves
were faster. Production retains its original limits.
