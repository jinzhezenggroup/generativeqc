# Decision: retain one screened force classification across exact-class consumers

Status: implemented, opt-in; CUDA gates passed, source performance regressed
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

## Qualification update: single-screen is necessary, not sufficient

Frozen production source `5de22fd9a377f43536a31866456f08c2187b9752`, based on
master `6cbb36343`, was compiled with ccache, Release O3 and sm_120 on n5 without
GPU execution there. Source identity is
`18f397771932744c7c438e2469241c024b09deb9d44a3a81167d294b6cd7c602`; the linked
library SHA256 is
`ec184b64996c856932fe1d9401cda4c38b1a42fcdd20e625aa41a8bbcc09e81c`.

n1 Slurm job 5740 used one allocated RTX 5090, unchanged 96-atom/768-AO spherical
def2-SVP density, FP64 and separate J/K sources. Both schedules retained the same
prepared owner and page allocation; only the borrowed page view selected mixed
versus compact execution. Three alternating paired repeats exclude the warmups
and separate admission-profile calls. Timings include the complete production
integral-source API (density transform, screening, queues, consumers and result
download), **not the complete molecular energy/force endpoint**.

| Schwarz domain | Mixed median (s) | Compact median (s) | Compact/mixed | Pages |
| --- | ---: | ---: | ---: | ---: |
| Unindexed | 26.200759399 | 49.375564404 | 1.88451 | 652 |
| Indexed | 13.076613403 | 16.524999858 | 1.26371 | 35 |

Every one of the 55 class ledger entries matches between mixed and compact, in
both domain modes. Totals are 92,233,228 admitted shell quartets, 1,263,186,780 AO
quartet capacity, 92,420,204 tile capacity and 5,944,643,268 primitive-quartet
capacity. These are admitted capacities, **not executed FLOPs**. Maximum separate
J/K disagreement across both modes is respectively 2.80e-12/1.31e-14, below the
1e-9 diagnostic gate.

The independent native CPU-ERI oracle passed mixed/compact crossed with both
domain modes. Additional four-distinct-center tests passed for spdf, pddd and
dddd in Cartesian and spherical representations, retaining the oracle's spin,
mask and separate-source checks. This is not nonzero-force coverage of every
through-f class. Compact indexed memcheck and synccheck both reported zero errors.
The frozen host suite passed 705 tests.

Linked restricted force resources were 255 registers and 90,568 bytes static
stack for the mixed kernel, versus 255/19,024 for compact dddp and 255/26,640 for
compact dddd. Stack footprint fell, register pressure did not. Static stack is
not a measurement of dynamic spill traffic or achieved occupancy; it did not
predict the observed source performance.

The selected block domains contain 2,669,205 and 139,884 products respectively.
At 4096 products/page and 21 present classes, the schedule launches 13,692 versus
735 class consumers, including empty slices. Removing repeated screening does
not remove these page/class boundaries or the retained recurrence work. Do not
attribute the regression to a particular stage without its measured breakdown.

Raw gates, all paired derivative vectors and class ledgers are retained locally
under `.artifacts/qualification-5740/`; the validated summary is
`.artifacts/qualification-5740-summary.json` (SHA256
`8765e40bae69af0b9b8b04f213f0fa49dc4d2648d301b51175ac490c37fda500`). Diagnostic
sources and binary receipts are retained alongside them and the evidence is
reported on PR #1841. No candidate complete E/F, cold or moved-geometry speedup
has been established. Keep the candidate default-off; do not promote a smaller
stack or single screening traversal as a performance result.

### Instrumented consumer breakdown

Separate n1 Slurm job 5742 profiled one mixed and one compact source call per
domain with Nsight Systems 2025.1.3, using the same frozen library and density.
These instrumented sums are not the clean paired medians and must not be added
to a different endpoint record. Required families were present, not imputed:

| Domain | All compact consumers (s) | Classifier (s) | Prefix (s) | Scatter (s) |
| --- | ---: | ---: | ---: | ---: |
| Unindexed | 49.317497 | 0.039241 | 0.001620 | 0.033234 |
| Indexed | 16.444862 | 0.036527 | 0.000087 | 0.034903 |

The indexed candidate's largest consumers were dppp (class 12, order 5,
2.670615 s), dpps (class 11, order 4, 2.109974 s), then dddp (class 19, order 7,
1.376264 s). Unindexed dddp alone took 8.501128 s. The mixed kernel is not split
by class in this trace, so these are candidate hotspots, not proof that a
particular class caused the regression versus mixed execution.

This rules out classifier/prefix/scatter kernel time as the principal observed
cost. Equal admitted work with 652 versus 35 page boundaries has very different
consumer time; one screening pass does not by itself establish a good consumer
schedule. Next scheduling experiments should isolate consumer granularity,
live-task accumulation and load balance under an explicit memory budget, not
repeat the classifier optimization or assume order 7/8 dominates after indexing.
Register/recurrence work still needs its own investigation and must coordinate
with #1839 rather than duplicate it. Nsight reports, CSVs, diagnostics and binary
receipts remain under `.artifacts/profile-5742/` and `.artifacts/build/`.
