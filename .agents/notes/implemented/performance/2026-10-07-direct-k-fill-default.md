# Decision: default bounded cross-chunk exact-K filling

Status: implemented
Date: 2026-10-07
References: #1892, #2076

The default-selection decision is superseded by the
[2026-10-09 work-default decision](2026-10-09-direct-k-work-default.md).
The original fill qualification and rationale below remain historical evidence.

## Problem

The qualified fill schedule improves fixed-density K wall and observed complete
PBE0 cold energy-plus-force medians against the frozen parent. Keeping that
schedule indefinitely opt-in prevents ordinary prepared raw-K owners from
using it. The user explicitly requests default fill and accepts the completed
qualification rather than repeating GPU measurements whenever master advances.

## Decision

During raw-K preparation, an unset or empty
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE` now selects `fill`. Explicit `fill` selects
the same schedule. Explicit `incumbent` preserves the original per-chunk schedule
for rollback and comparisons; `primitive` remains opt-in. Invalid values still
fail closed, and prepared owners never reread the selector during execution.

This supersedes only the default-selection part of the
[initial queue decision](2026-10-07-direct-k-cross-chunk-queue.md), not its
algorithm, scientific ownership or historical evidence. No master update or
additional real-GPU performance run is required for this selector-only change.

## Invariants

The topology member initializer stays incumbent so owners that do not prepare
raw K retain their existing defaults. J/HF preparation, explicit Rys/block
lowerings, unsupported-class fallbacks, task identity, screening, precision,
recurrence, scatter mathematics and generated artifact identities are unchanged.
The queue remains bounded and flushes every admitted task exactly once.

## Evidence and consequences

The existing independent matrix, derivative and sanitizer gates are retained.
Recorded full-density K wall reductions are 8.10%/13.65% at 48/96 atoms; observed
complete cold E+F reductions are 2.31%/4.91%. These remain frozen-parent results,
with the original sample-count and SCF-trajectory caveats, not a new measurement
of current master. The native candidate still does not outperform GPU4PySCF.
Primitive grouping remains experimental because the recorded 96-atom cold
median regresses 1.61%.

The selector test covers unset/empty defaults, explicit fill, incumbent rollback,
primitive selection, invalid input and frozen selection after environment
mutation. It also preserves the non-K topology's incumbent initializer. The
existing concurrent emitted-worker tests continue to check all three schedules.
The focused selector/emitted-worker/ABI suite reports nine passed in 26.22
seconds; no additional real-GPU benchmark is run for the default switch.

Revisit the default if a reproducible complete-endpoint or numerical regression
appears; explicit incumbent remains the bounded rollback. K-block composition,
symmetric-density writeback and the remaining #1892 value/derivative work are
independent follow-ups, not prerequisites for this default switch.
