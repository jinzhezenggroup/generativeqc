# Decision: reject speculative Fock barrier changes

Status: rejected
Date: 2026-10-25

## Problem

Complete HF device-graph replay reported divergent warp barriers in generated
ssss and ddpp persistent Fock kernels on RTX 5090, driver 580.95.05 and CUDA
12.9.1. Ordinary numerical execution, memcheck and racecheck passed. Replacing
the single-warp packed queue's CTA barrier with a warp barrier moved the first
failure; adding warp barriers before the generic queue's CTA barriers did not
resolve the complete endpoint. The compiler lowered the latter additions to NOPs.

## Decision

Restore the original production emitters. The evidence does not establish a
Fock synchronization defect, and changing scientific production code to move
the first instrumented failure would conceal the actual diagnostic limitation.

A standalone two-kernel program, without any GenerativeQC dependency, reproduces
the failure. Its consumer executes an unconditional CTA barrier before writing
2,816 independently checked outputs. Ordinary launches and ordinary host Graphs
pass synccheck. Setting only `cudaGraphInstantiateFlagDeviceLaunch` causes the
same unconditional barrier to report divergence, even without a device-side
launch or persistent queue. Ordinary execution verifies every output in all
three modes. Memcheck and racecheck pass the device-launch-enabled Graph.

Both Compute Sanitizer 2025.2.1.0 and 2025.4.1.0 reproduce this distinction.
Both detect a separate intentionally divergent barrier in an ordinary Graph.
Kernel filters are not a dependable workaround: filtering the producer still
reports the unselected consumer's barrier. Keep the unrestricted failing logs.

This establishes a device-launch Graph synccheck limitation on this measured
stack, not a general assertion that production synchronization cannot be wrong.
Record full endpoint synccheck as instrumented-unqualified; do not report it as
passing or remove independent numerical, memory, race, or supported-launch
synchronization coverage. Instrumented observations are never benchmark timings.

## Separate provider test correction

Automatic MD-J can replace the overall schedule label while generated exchange
remains selected. Test the actual generated K admission for a value-only spec
(`derivative_order = 0`), retaining independent ERI/J/K comparisons, spin and AO
representation coverage, and the through-f fallback. Diagnostic failures include
angular order, representation and overall schedule to identify the case.

## Rejected alternatives

- A warp barrier in a 32-thread queue is semantically sufficient, but the
  observed instrumented failure does not require this production change.
- Explicit warp joins before a wider queue's CTA barrier did not fix replay.
- An intermediate Rys-emitter edit did not affect the selected generic ddpp
  kernel. Verify generated consumers before changing an emitter.
- A newer sanitizer did not resolve the minimal reproduction.

## Evidence and revisit conditions

[Retained diagnosis](../../../benchmarks/results/packed-fock-warp-sync-20261024/README.md)
contains the standalone reproducer, all minimal controls and the failed candidate
logs. The failed generic candidate was `f8edc52a6d7fe331be55eb7b35e106f9ea72f5fb`;
`92674610bac506d76b762bc30594e72aae0a1baa` restores the integrated production
emitters and changes only the provider test relative to `121b14ab59aa5ed94c2311de634e11e745e8923a`.

Rerun the minimal controls after driver, toolkit, GPU or sanitizer changes before
claiming full device-graph synccheck qualification. Reopen a production repair if
a supported isolated launch or independent numerical/race check demonstrates a
real ownership or synchronization failure.
