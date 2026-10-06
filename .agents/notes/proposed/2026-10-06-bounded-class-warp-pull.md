# Proposal: bounded class-queue warp pulling for Direct force

Status: proposed; implementation candidate prepared, not CUDA-qualified or promoted
Date: 2026-10-06
Agent: ChatGPT
Model: GPT-5.6 Sol

## Problem

Issue #1892 asks for dynamic warp consumption of already classified work, not
another experiment in independently claiming raw domain pages. #1978 retains
negative warp-page results and a separate 128-thread candidate. Neither proves
that this proposal wins, nor that Barrier stalls alone caused any prior speedup.

The implementation was rebased onto master `21e682d1e24de606b85c76f174ad65e82cec66d5`; the inspected Direct-force source blob remains `cf8e14098fc230edc719604dd317e7ae1cdd5ef3`. Its generic
bounded force consumer already uses a warp-uniform shell task and has no CTA
barrier between individual warp tasks. It has CTA barriers between admission,
scalar consumption, generic consumption and batch/page reuse. Static striding
can still leave one warp with a longer bundle than its siblings.

## Candidate

Keep the existing persistent CTA, page cursor, scientific admission, fixed
256-descriptor queue and scalar consumers. Link admitted descriptor indices
into CTA-local exact-class lists during that same admission pass. After the
existing publication barrier, a finished warp leader uses pop-only CAS to claim
its next descriptor. The warp keeps its current class while work remains and
otherwise looks for another class; all lanes receive one uniform descriptor.
The existing force recurrence, AO screening and atomic scatter are unchanged.

This is **batch-local dynamic warp draining**, not an unbounded/global persistent
warp pipeline. Different warps in one CTA may still execute different classes.
The scalar-phase and final batch-reuse barriers remain. Intra-task AO divergence
and the generic recurrence's static footprint are not solved by this change.
A class list is LIFO, not FIFO: no stable or bitwise force accumulation claim is
made. This is restricted to the existing unordered atomic-force consumer.

The diagnostic environment setting
`GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_WARP_PULL=1` selects a separate template
instantiation only at the full-range force launch boundary and only for the
original 256x1x1 CTA. Unset/zero/other strings retain ordinary selection. Value,
SR/LR, RSH and diagnostic angular-pass entrypoints remain ordinary. This is not
a user-facing production opt-in recommendation. Do not promote it or add method
name dispatch. A future accepted selector must belong to prepared scheduling.

## Concurrency and resource invariants

Each admitted slot is published once and stays alive until all consumers finish.
There is no publication during draining. `next[]` is immutable in that phase;
heads only pop, so a removed node cannot reappear and CAS has no ABA problem.
A publication-phase CTA barrier, not atomic exchange or register shuffle,
provides descriptor/next visibility. Warp-wide synchronization is retained at
collective boundaries and around existing tile mutations. Batch reuse waits for
all readers; no producer/consumer spinning protocol or cross-CTA barrier is added.

The shared metadata uses (55+1+256)*4 = **1248 bytes per experimental CTA**, plus
ordinary compiler layout/alignment. There is no new global arena, retained
scientific cache, second topology or device allocation. The final defensive bin
is for an unexpected class index; it makes no new scientific capability claim.
The disabled storage type has no arrays, but binary/linked resource identity,
module load/compilation costs and default-route performance still need checking.

Low-order scalar-owned descriptors are linked too, then skipped by the retained
warp ownership checks. This avoids a second copy of the scientific dispatch
predicate, but adds overhead that must be measured. No queue capacity is relabeled
as an admitted/post-AO/primitive work count.

## Evidence and remaining work

The independent Python interleaving model covers every packet tail 0..256,
multiple worker counts, skewed/all/fallback classes, stale-CAS retries and unique
ownership. A concurrent C++ test uses the actual portable algorithm. A standalone
CUDA gate exercises the real wrapper, repeated batch reuse and full-warp
collectives; neither harness is an ERI/force oracle.

No GPU, CUDA compiler or ccache is available in the implementation environment.
The native test explicitly skips without ccache. CUDA build, sanitizers,
scientific gates, source-matched NCU and full endpoints remain **unverified**.
No performance claim, default change, P0-A composition or issue closure follows
from the Python model. See the PR's mandatory GPU-validation checklist.

## Rejected shortcuts and follow-up

Do not let warps claim/re-enumerate raw domain pages. Do not remove batch reuse
barriers. Do not assume shuffle supplies a shared-memory fence. Do not equate
local-memory requests to spill bytes or use profiler replay duration as timing.
Do not infer increased active lanes or lower registers from dynamic work pulling.

If this candidate wins after full qualification, consider prepared/compiler-owned
selection, measured claim batching and cost ordering. Global/preclassified P0-A
queues require their own bounded storage/lifetime and composition qualification.
Do not claim those follow-ups are implemented here.

## References

- #1892, including comment 6005704129 (dynamic homogeneous warp scheduling).
- #1978 (separate warp-page negative and 128-thread evidence).
- NVIDIA CUDA Programming Guide, C/C++ language extensions: warp sync intrinsics.
