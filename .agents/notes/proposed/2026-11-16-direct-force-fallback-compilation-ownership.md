# Proposal: compile shared force-class and bounded-fallback bodies once

Status: superseded by qualified private NVIDIA-RDC compilation ownership
Date: 2026-11-16

The force include-wrapper is adopted in
`../implemented/performance/2026-11-16-direct-force-fallback-compilation-ownership.md`.
Retain this investigation record, including the less useful arithmetic-only
module and narrower Fock-group alternatives. Its intermediate checkpoints are
historical, not current production acceptance claims.

## Remaining cost

After the composite-source reduction, the retained clean sm120 build still has
a 1,514.488-second bounded-fallback task and a 229.088-second force-class task.
Those overlapping historical observations are not isolated cold-wheel timings.
Both belong to the existing NVIDIA relocatable-device-code archive, with
identical compiler, optimizer, architecture and constant-bank options. The
independent angular-force owner has different launch-bound constraints and must
remain outside this experiment.

An executable resource/ELF census of the retained native objects finds **274
common weak function identities** between `direct_bounded_fallback.cu` and
`direct_force_class_domains.cu`. Their duplicate sm120 SASS function bodies sum
to **35,889,280 bytes**. This is unlinked function-text ownership, not a wheel
size or memory measurement: the existing device linker already coalesces them.
Of these, **46** generated weighted-arithmetic functions account for **9,549,440
bytes**; the other shared native consumers account for **26,339,840 bytes**.
The force-class object has 320 function resource records, so extracting only
the generated arithmetic leaves much of its duplicated native compilation.

## Distinct hypotheses

1. An isolated arithmetic-owner prototype replaces those 46 existing noinline
   definitions in both consumers with declarations and compiles their unchanged
   bodies once. This introduces no new mathematical expression or call boundary,
   but cross-translation-unit visibility could affect compilation or execution.
2. Prefer a simpler, broader candidate if it qualifies: compile the two existing
   source owners in one native CUDA translation unit. Their common header-safe
   bodies then instantiate once without a new external-device ABI, new call
   boundary or changes to their scientific headers. Keep the original source
   files and independently compiled non-RDC/backend fallback; do not merge the
   separate launch-bound angular-force owner.

These are not repetitions of the rejected density-preparation or per-angular
range-dispatch extraction. Those changed call boundaries inside a scientific
consumer and produced insufficient complete-owner wins. See
`../rejected/2026-11-16-direct-shared-density-preparation.md` and
`../rejected/2026-11-16-direct-range-order-sharing.md`.

## Initial arithmetic-owner checkpoint

The standalone arithmetic module completes with the unchanged sm120 Release,
RDC, FP64/FP32 and constant-bank options and verified sccache 0.16.0: zero hits,
four CUDA-stage misses, 60.785 observed request seconds. The class consumer
removes exactly the 46 arithmetic definitions, retaining every other function
identity/resource record; all shared arithmetic resources also match. Its
290.502-second observation overlaps other compilation and is not a controlled
regression/speedup result. The complete bounded consumer finishes in 1,438.878
observed seconds, with a 5,897,272-KiB reaped-worker maximum RSS. Across all three
new owners, object bytes fall from 105,523,120 to 100,902,840 (4.38%). All **778**
unique function identities and their resource records remain identical; only
46 of the 280 extra function copies disappear. The three observations sum to
1,790.165 seconds, versus 1,743.576 historical overlapping task seconds. These
are not controlled timings and do not establish a compiler speedup. The modest
object reduction and additional external-device ownership do not justify
adopting this arithmetic-only design by themselves.

A separately linked real-GPU task probe exercises this actual arithmetic object
against the retained original library and **120 complete native density-to-force
task cases**, with independent Libcint orbit assembly and identical semantic
counters. Diagnostic one-lane timing has a median candidate/baseline ratio of
**1.028**, maximum **1.054**. This is not a complete Direct/HF endpoint or
occupancy qualification and is insufficient to adopt the split arithmetic ABI.

## Broader archive census

A complete weak-symbol census of the 14 native archive owners confirms the
two-owner overlap above. It also finds 21,761,280 bytes of repeated sm120 function
text between bounded fallback and Direct-JK, and 10,985,472 between angular Fock
and Direct-JK. These are potential repeated compiler work, not removable final
ELF text. Combining the largest owners indiscriminately could recreate the
wheel's 12-GiB memory failure; do not select a single giant translation unit just
because the symbol-overlap count is larger.

A second isolated prototype groups the smaller weighted-ERI, cached-tensor,
packed-Fock, angular-Fock and Schwarz launch owners. Its **484** dependencies,
original objects and commands are frozen separately. The hypothesis is that
shared recurrence functions compile once while keeping the combined owner
substantially smaller than the force fallback. That five-owner grouping is not
adopted: it reduces object bytes by 13.27%, with identical function resources and
SASS, but does not improve the observed complete-owner time. The narrower
cached-tensor/packed-Fock pair qualifies separately; see
`../implemented/performance/2026-11-16-direct-tensor-fock-compilation-ownership.md`.

## Complete sm120 force-unity checkpoint

The unchanged force owners together have 1,058 resource-function copies and
105,523,120 object bytes. Their unified owner retains all **778** unique
identities and exactly identical resources, with **86,893,200 object bytes**
(17.65% lower). Every scientific function's SASS section bytes are identical.
The only differing section is compiler intrinsic `__internal_accurate_pow`:
the two original copies already differ, and the candidate matches the original
bounded-fallback copy exactly. Do not hide that pre-existing distinction as
universal binary equality.

The candidate records zero hits, four CUDA-stage misses, 1,675.828 observed
request seconds and a **6,637,040-KiB** reaped-worker maximum RSS. The historical
original tasks sum to 1,743.576 seconds, so these overlapping observations do
not establish a substantial cold-build win. The larger owner's memory cost
also needs explicit wheel-matrix qualification; it must not recreate an OOM.

At this checkpoint, unchanged full-wheel **80;90-real;120-real**, compute80 PTX,
complete original and candidate requests were still running. The completed
matrix is recorded below; the narrower tensor/Fock pair's qualification does
not qualify this separate force experiment.

## Completed full force matrix

All three requests finish successfully under independent owned foreground
sccache servers and finite deadlines. The complete original group versus the
single include-wrapper, with unchanged **80;90-real;120-real** and compute80 PTX:

- Object sum: **178,165,680 -> 143,452,816 bytes** (19.48% smaller).
- Identically extracted PTX dumps: **177,680,559 -> 152,347,547 bytes**
  (14.26% smaller).
- Function copies: **960 -> 680**, **957 -> 678**, **1,058 -> 778** for
  sm80, sm90 and sm120. No function identity is added or removed, and every
  retained register/stack/shared/local resource record matches.
- Every candidate SASS section matches an original copy. The two original sm120
  `__internal_accurate_pow` copies already differ; the wrapper matches the
  bounded-fallback copy. Do not claim that all original copies were identical.
- Observed requests: **2,635.695 + 547.543 -> 2,867.417 seconds**, about 9.92%
  lower summed observations. Each request has **three hits / five misses**;
  requests overlap. This is not a controlled cold build or wheel speedup.
- Reaped-worker maximum RSS: bounded **5,710,904 KiB**, class **1,751,424 KiB**,
  wrapper **6,068,284 KiB**. The wrapper increases the largest observed worker
  by about 6.3%; retain serial wheel compilation and the 12-GiB container cap.

Authoritative receipts are `force-wheel-comparison.json` and
`force-wheel-matrix/` under the evidence directory. At this checkpoint a private
NVIDIA-RDC-only CMake wrapper was staged for actual native build/link and
endpoint tests, before publication or clean-source qualification. See the
implemented record for subsequent acceptance. Non-RDC/CuMetal and the separate
angular-force WPO owner retain their original source ownership.

## Acceptance boundaries

- Compare the complete combined old owners with every new owner, not just one
  removed header or helper. Retain real compiler-cache, worker-resource, object,
  PTX and per-architecture function evidence. Do not claim a final-ELF reduction
  from removing definitions already coalesced by device link.
- Preserve every kernel, mathematical function, angular order, range/omega
  admission, screening predicate, queue/window bound, force/scatter mode, spin,
  strict retry, counter and resource/lifetime fallback. No production CPU oracle
  or unsupported-architecture bypass is acceptable.
- Inspect register, stack, shared and local resources for all retained kernels
  and common callees; a larger unity owner can hurt compiler peak memory or
  launch-bound propagation even when duplicate symbols disappear.
- Require real native device-link, independent force/source oracles, failure and
  lifetime controls, and complete public cold/warm/moved endpoints with work
  counts. Qualify actual clean final code separately from generated-header or
  include-wrapper prototypes.
- Keep the 12 GiB wheel isolation and finite deadlines. Reject if the larger
  owner materially worsens memory/endpoint behavior or fails to reduce actual
  complete build work. Full manylinux build/link/repair/installed acceptance
  remains its own gate.

## Evidence location

Ignored `/tmp/gqc-wheel-weighted-owners/` retains the full owner census, decoded
ELF symbol sizes, both original compile commands, **496 frozen dependencies**,
the 46 unchanged arithmetic-body identities, derived candidate headers/source,
owned compiler-server receipts, complete objects/resources and task probe
receipts. Both original checkout and published PR remain unchanged by the
prototype compilation inputs. Do not restart a live request on an observation
timeout or relabel a prototype as a clean native artifact.
