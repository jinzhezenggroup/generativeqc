# Decision: retain one screened force classification across exact-class consumers

Status: implemented, opt-in; CUDA qualification pending
Date: 2026-10-04

## Problem

The mixed bounded derivative kernel reaches high register/stack use. Splitting
consumers into homogeneous angular passes reduces reachable recurrence state but
repeats screening. PR #1836 measured the whole fixed-density source on n1 RTX 5090:
mixed/grouped medians were 27.292842/29.813762 s without the Schwarz index and
13.598428/14.065734 s with it. Numerical and 55-class work gates passed, yet both
schedules regressed. That PR was closed; smaller stack was not sufficient.

## Decision and ownership

The actual PBE0 independent J/K derivative owner is `GeneratedExchangePlan` in
`direct_coulomb.cpp`, not RHF's `prepare_generated_shell_tasks()`. Reuse the bounded
classifier, generated prefix/scatter and qualified force contraction at that
owner. Do not route through a second RHF provider or duplicate screening math.
The compiler owns page admission and canonical class/order dispatch. Native code
owns stream ordering, allocation, disjoint traversal and warp-local consumers.

The page retains each accepted raw pair and its class. Prefix/scatter consumes
those tags without rescreening, then fixed-class/order kernels consume contiguous
slices. Raw `max(pair), min(pair)` orientation preserves the qualified derivative
consumer's AO multiplicity; do not substitute generated canonical task ordering
without independently validating its permutation/scatter convention.

## Work and memory

Let `B` be the selected block-product domain and `Q = 1024 P` the page candidate
capacity. Screening visits each block-product interval once, followed by cheap
tag/scatter traffic over at most `1024 B` slots, and the retained recurrence work.
This replaces repeated whole-domain screening, not the exact integral's formal
growth. At most `C ceil(B/P)` class launches occur; empty slices still cost launch
overhead. Scratch is `25 Q + 884` bytes, capped at 104,858,484 bytes. Budget decline
and device OOM retain the mixed bounded route; other execution errors do not
silently fall back. The lower-memory policy adds disjoint pages, not rescans.

## Invariants and qualification

- Independent J/K sources, spin weights, AO screening, FP64 recurrence and
  spherical transforms are unchanged; no DF or mixed precision is introduced.
- Optional storage is charged to the prepared owner and retained until its stream
  is drained. Prepared/checkpoint identity includes the schedule control.
- Host capsules test actual indexing, composition and error paths, not GPU races
  or chemistry. The native CPU-ERI oracle remains mandatory for real consumers.
- Keep the candidate default-off until separate-source numerics, class work,
  sanitizer/resource checks and same-GPU complete endpoints qualify it. Preserve
  cold and changed-geometry regressions, not just the best warm sample.

## Rejected alternatives and coordination

Do not revive #1836's per-order whole-domain scans or replace bounded pages with
an unbounded molecular quartet queue. Existing RHF count/retry and exact-class
page paths are not universally single-screen; reuse their primitives, not their
work amplification. PR #1839 owns recurrence/Hermite-convolution changes; this
candidate is scheduling-only and does not duplicate that mathematical work.

The candidate was ported onto master `6cbb36343` after #1761 landed. Its retained
angular diagnostic is not enabled by this change: an admitted compact page takes
precedence for full-range work; declining the page preserves the prior owner
schedule. The LR schedule and opt-in reachable recurrence are unchanged.

## References

- PR #1836 rejection: https://github.com/jinzhezenggroup/generativeqc/pull/1836#issuecomment-5978288770
- Current contract: `docs/developer/direct_force_pages.md`
- Independent endpoint campaign (not this candidate): https://github.com/jinzhezenggroup/generativeqc/pull/1830#issuecomment-5978335157
