# Decision: retain physical-slot Gram state for ordinary CUDA HF/KS

Status: implemented
Date: 2026-10-10

## Problem

Ordinary HF/KS recomputed every old-old residual dot after each ring insertion,
including normalized retries that only retire the oldest dependent vector.
Its augmented solve overwrites the matrix, so that allocation is not a valid
persistent Gram owner. #1874 requests incremental vector work and a shared
scientific reduction owner, not merely a vendor substitution.

## Decision and invariants

Generate the ordered FP64 dot from the existing canonical `diis_gram_program`.
Use a common tensor runtime to refresh the current physical row/column against
the live ring exactly once per insertion. Preserve the scalar accumulation
order of the ordinary kernel; mirror finite symmetric dots without reducing
them again. Keep the unnormalized cache separate from the destructive solve.
HF and KS numeric layouts charge `8 * batch * history_capacity^2` payload bytes
plus their existing alignment rules before allocation. No numeric allocation,
provider creation or host residual/Gram copy is added during replay.

The caller inserts the new vector before refresh, and live older vectors are
immutable until their physical slots are replaced. Replacing a slot refreshes
every newly needed pair. Resetting count/head needs no cache clear: successive
insertions establish all pairs in the new window. The first insertion defers
its norm, because a one-iteration warm endpoint never consumes it. The second
insertion computes that prior diagonal once alongside its new row. Thus an
unchanged warm endpoint performs no additional vector-dot work; initialization
cost remains explicit in the actual device counters. Retirement narrows that valid
window without changing old-old values. Normalization is applied only when
copying cached values into the solve; a singular retry cannot corrupt the cache.

Keep the old entry point and compact DF cooperative partials unchanged. The
new route requires one complete warp and independent cache storage; disabled
or one-vector histories retain the old copy-only behavior and accept no cache.
Optional caller-owned device counters count actual refreshed dots/vector
elements, not solve/cache traffic, wall time or complete endpoint work.

## Rejected alternatives

Reusing the solve matrix would retain eliminated/normalized values. Moving
history vectors to keep chronology would add unnecessary vector traffic.
Switching to cooperative/BLAS reductions at the same time would conflate work
removal with a changed reduction-order qualification. A slot/count equality or
source counter alone cannot prove numerical cache validity.

## Qualification and remaining scope

The native test calls both real production kernel entries, compares actual
history, coefficients, head/count and Fock output bits, and independently checks
cached dots using long-double accumulation. It covers 36 spin/size/history/
normalization combinations, 32 insertions each, wraparound, dependent-vector
retirement, uncleared reset state and poisoned inactive systems. Disabled and
single-vector cases are separate. History capacity two intentionally does not
retire a dependent vector under the unchanged legacy singularity policy; the
first test attempt incorrectly demanded retirement there and is retained as
failed evidence rather than a production bug.

Source-matched build, Slurm, independent numerical, resource and complete
endpoint results are retained under `.artifacts/issue1874-incremental-gram/`.
The AOT-disabled qualification build lacks packaged stationary PBE CUDA force
artifacts. A failed full-force benchmark is preserved; subsequent DFT energy-only
measurements are explicitly named and are not full E+F qualification.

This removes repeated ordinary residual-vector Gram work, not all of #1874.
Public live-history/work diagnostics, shared Fock-history extrapolation,
broader admitted complete E+F timing/crossover evidence and optional provider
competition remain. Do not infer a whole-endpoint speedup or issue closure from
the cache's lower vector count. Revisit changed reduction order only with its
own independent numerical and full endpoint acceptance gates.

## Final source-matched qualification (2026-10-10)

The final lazy-initialization freeze is tree
`37c20f0053cd56165fc2532633a99b4fa1c5caae`, based on
`f87d51ab616cc9aa774bd344a15e268cef7eaf05`. Before this appendix, all 9,208
manifest entries match the actual local source bytes. This appendix postdates
that freeze and does not change its qualified implementation or tests. The
freeze includes preceding allocation, residency and MP2 admission work;
qualification of that combined tree is not automatically qualification of an
isolated PR patch or a later upstream integration.

Slurm job **6967** uses `main`, `node1`, one `gpu:5090:1`, a finite 20-minute
limit and the scheduler-assigned device visibility. Its retained controller
state is `COMPLETED`, `ExitCode=0:0`. The actual-kernel ring probe passes the
36 combinations and disabled-history cases described above, including
bitwise legacy-route state comparison and independent long-double dot gates.
Fourteen public GPU endpoint/resource regressions pass, and the final host
suite reports **156 passed**. Compiler structure reports 504 modules and zero
dependency errors; focused formatting and diff checks pass. The qualified
library SHA-256 is
`afbb6a20d755513cda3d5ae0cc4c387d175421e44a5893a19bc1a583afe3e96f`,
and the native probe SHA-256 is
`7486276bc7ac134dd1667b5aaaaf4da2a339ca754b649c5b15e3981b08ae612a`.

The retained endpoint comparison uses baseline job **6966** and candidate
job **6967**, with cold, warm and changed-geometry calls for six cases. HF
cases include energies and forces; PBE RKS/UKS cases include **energy only**.
Iteration trajectories agree. Maximum energy, force and residual differences
are respectively `8.526512829121202e-14`, `8.604120020627715e-15` and
`8.443862231577993e-16`. The baseline library SHA-256 is
`9767c63e12770f6704a498525ab024b6f1115ee48a9e10dd7af48d01884de1c2`.
Most timing ratios are near one; the tiny-system sequential timing is noisy.
The comparison explicitly sets `qualifies_default_crossover: false` and
`claimed_endpoint_speedup: false`. DFT energy measurements do not replace
the missing stationary PBE CUDA force qualification.

Raw manifests, logs, completed controller states, binaries and comparison
receipts remain in ignored `.artifacts/issue1874-incremental-gram/`. Failed
attempts remain distinct from passing qualification; no issue closure or
release publication follows from this evidence.

## Isolated PR integration status

Upstream subsequently merged the opt-in incremental HF/KS Gram implementation
in #2159 (`828b8c1b0`). Its kernel, arena and endpoint changes overlap this
ordered-cache design. The isolated patch is therefore retained on its original
baseline as a draft for reconciliation, not as a merge-ready default promotion
or a reversion of upstream work. Review reduction order, lazy initialization,
resource charging and entry-point compatibility before choosing an integration.

The isolated branch registers the exact shared tensor dependencies, native
CUDA ownership shard and generated-dot family. Its 156 host tests, SCF/compiler
structure, ownership inventory and focused formatting gates pass. These later
host/inventory checks do not assign historical combined-tree GPU measurements
to the isolated PR head or qualify the still-unimplemented reconciliation.
