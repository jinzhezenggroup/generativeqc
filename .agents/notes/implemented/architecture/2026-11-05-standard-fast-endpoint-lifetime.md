# Decision: share Standard and Fast endpoint ownership

Status: implemented
Date: 2026-11-05

## Problem

Standard and Fast already shared native precision scheduling and candidate
classification, but scalar Standard discarded its workspace after every call
while Fast retained successful unaudited candidates. The default-comparison
and cold/warm benchmark factories also chose a raw `Calculator` for Standard
and the policy facade for Fast. This conflated precision policy with execution
overhead and resource lifetime.

## Decision

Use `PrecisionCalculator(policy=arm)` for both benchmark arms. Read resolved
settings and native binary identity from the selected candidate, and native
capacity/snapshot protocols from the prepared fleet underneath the policy owner.
Migrate benchmark consumers rather than adding forwarding attributes or aliases.

Retain a successful unaudited scalar candidate for either policy, not just Fast.
Validate result shape, finiteness, convergence and backend before retaining it.
Release failed or unpublishable candidates, including classification failures
and cancellation. Native audit scheduling still releases candidate residency
before independently prepared references; temporary fallback owners remain
short-lived. Explicit `clear_cache()` remains the scalar release boundary.

## Invariants

- Standard retains strict arithmetic and existing optimized SCF controls.
- Fast admission, bounded fallback, observable audits and independent reference
  requirements remain unchanged; shared execution is not shared accuracy.
- No reference density seeds a candidate, and no audit doubles resident owners.
- Prepared plans retain explicit/context-managed `close()` ownership.
- Existing receipts retain their historical identity. No speedup or new GPU
  qualification is inferred from this structural change.

## Evidence

The policy tests exercise resident replays, explicit release, independent audit
ordering, invalid publications, classification failures and cancellation for
both profiles. Native strict cold/resident/transformed/cleared endpoints retain
the pinned independent energy/force gates for H2, water and unrestricted HF,
including rotational force covariance and geometry-bound model identities.
Benchmark constructor tests require both arms to select the public policy owner without
manual convergence or arithmetic controls. Existing Fast native oracle and
prepared-fleet tests remain required.

Local qualification used a fresh CPU-only Release build from this checkout,
with verified sccache 0.16.0 and generated compiler commands invoking it. The
focused policy/native/benchmark/documentation suite passed 827 tests and 70
subtests; 64 CUDA tests were skipped. Sphinx HTML with warnings as errors,
Ruff checks/formatting and the native runtime ownership audit passed. Compiler
cache statistics were retained before and after the build. No real-GPU
qualification or performance result is claimed: local Slurm had no usable GPU.

## Rejected alternatives

Making Fast discard its workspace would unify the wrong behavior and lose
qualified reuse. Bypassing the facade for Standard would keep benchmark control
and timing paths different. Collapsing Standard and Fast tolerances or removing
Fast audits would change scientific contracts and is not part of unification.
