# Decision: retain workload-specific XC scheduling and bounded VV10 proposals

Status: rejected
Date: 2026-11-01

## Evidence and decision

The [three-device integration bundle](../../../benchmarks/results/portable-integration-20261101/README.md)
keeps complete timings and independent numerical gates for the following
alternatives. None changes production defaults.

A single larger SCF point tile is unsuitable: B3LYP benefits on all three tested
GPUs, especially A100, while r2SCAN-3c warm calls regress at 1024. Preserve explicit
public tile bounds. An eventual automatic policy must distinguish an omitted
choice from a caller's explicit bound, compose force inheritance and resource
quotes, and retain the chosen execution identity. GPU names are not a cost model.

The compiled potential matrix tile 32 regresses all three sampled A100 methods;
retain 16. The original matrix32 computation succeeded, but upload failed; its
raw results were recovered through the scheduler rather than discarded.

Replacing bounded provisional VV10 updates with ordinary full-target SCF reduces
iterations but quadruples pair sweeps and slows the complete A100 WB97M-V cold
endpoint by 35.06%. Forced range residency changes in the same experiment, so it
cannot isolate solver and residency effects. Native-cold provenance is already
preserved in `phase_source_cold`; the hypothesized loss of that provenance was
not found. Keep both expensive operator counts and complete endpoints decisive.

Indexed GEMM plus symmetric publication improves isolated XC components, but
removes compact generated batching in the current prototype. Its component gates
and fixed-density timings do not qualify a default. Revisit with a correctly
identified shared lowering, bounded resource ownership, complete endpoints and
independent force gates. Its FP64-specific default cuBLAS math does not authorize
TF32 or a global provider math-mode change.
