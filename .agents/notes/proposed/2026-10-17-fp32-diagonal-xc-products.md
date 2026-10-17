# Decision: open the existing diagonal XC schedule to FP32

Status: proposed
Date: 2026-10-17

## Problem

Master already optimizes diagonal jet bilinears, but both the standalone and
compact-batch selectors require double arithmetic. Lowering the same products
to FP32 consequently loses a schedule already available to FP64. This candidate
only opens the existing schedule to float; it adds no new FP64 optimization.

## Decision

Parameterize both diagonal consumers by their existing arithmetic type and
select the schedule by the shared four-jet representation, not a functional
name. Retain canonical FP64 half-panel preparation, FP64 global storage and
ordered inter-tile scatter. Density and point programs are not changed.

The qualified FP64 operand bound remains2^128. Float uses a conservative2^30
AO/panel/weight bound so doubling a finite FP32 right panel cannot overflow
before an otherwise finite product. Unsafe tiles execute both original legs;
they are not clipped, repaired, or silently recomputed in double. FP32 inner
sums remain FP32, with unchanged FMA policy. In particular, this candidate
contains neither the rejected chained-FMA trials nor the separate pair-FMA
trial. The mathematical point program and any strict publication phase remain
independent decisions owned by the caller.

## Invariants

Both spins, signed weights, indexed maps, ragged/tiny point and AO tiles,
symmetry, explicit overflow rejection and strict default behavior remain
covered. Preserve every existing FP64 gate and the original FP32 product gate.
Add finite FP32 panel-doubling overflow, underflow-cancellation and hidden
preparation-underflow controls against the original float two-legged body.
Check the compact float consumer against independent unfactored bilinears,
including FP32 local results and FP64 scatter across overlapping maps.

## Evidence and promotion boundary

This isolated trial is based on master b81022468 and does not itself admit
ordinary Fast XC products. The separately qualified shared FP32 XC admission
must be integrated before full Fast performance qualification. Native CUDA
tests run only through finite srun with verified ccache and Slurm visibility.
Host simulation is not GPU or complete E/F evidence. Require independent
complete E/F and matched96-atom endpoint performance before promotion; neither
the historical v22 XC speedup nor a microbenchmark qualifies this new source.

The isolated production identity is
`2e560fc87580686bb9d98f1a4475e2d4cc3267836e9cc454882323ef89c65192`.
Host qualification passes23 cases, including the compact float CPU simulator;
the dependency audit passes565 modules. Scheduled numerical job3309 passes
all86 generated CUDA cases with FMA-on/off and memcheck0, no failures/skips.
Both old double gates and new float range controls pass. Node2's retained
NVML mismatch excludes every timing from performance evidence. This does not
qualify the full CUDA compact runtime, ordinary Fast admission, E/F or speed.

## References

- `.agents/notes/implemented/performance/2026-10-16-xc-diagonal-potential.md`
- `.agents/notes/implemented/performance/2026-10-16-xc-diagonal-range-guard.md`
