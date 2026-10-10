# Decision: explicit bounded workspace for large-tile DF CCSD GEMMs

Status: implemented
Date: 2026-10-10

## Motivation and diagnostic boundary

Increasing Q32 to Q64 has no observed complete-endpoint win and adds about
5.42 GiB; see `../../rejected/2026-10-10-df-cc-q64-default.md`. The two dominant
ladder GEMMs account for about 92.15% of representative Q32 GEMM summands.
That is a static work count, not a profiler timing percentage.

Slurm 2830 probes the actual row-major recipes on node2/PRO6000 using ordinary
FP64 `CUBLAS_PEDANTIC_MATH`: first N/T, m9945/n7072/k221, one batch; second N/N,
32 batches of m221/n9945/k221. Two observations per 0/4/12-MiB selection each
contain eight timed repetitions, excluding input/preparation/warmup. Long-double
CPU samples check the recipes independently, not every output or a molecule.

Median first call is 17.715916 ms at zero workspace and 17.267810 ms at 4 MiB
(about 2.53% shorter); 12 MiB is not better. The second call is unchanged at
about 17.1425 ms. Sampled errors are at most 2.35e-15 and 1.31e-15, below the
1e-11 probe gate. This signal is small: roughly 0.26 seconds across 570 full
Q32 tiles, before tails or complete endpoint ownership. Do not promote a
default or claim an endpoint win from it.

The first attempt (Slurm 2829) stops before the probe because ccache needs the
existing dependency library path for libhiredis. The failed receipt is preserved;
2830 adds that path and reuses the existing cache. No cache is cleared.

## Decision and fallbacks

The scratch candidate first changes only `CudaContractionContext` and the DF
CUDA solver owner. Its qualified production integration does the same. The context may
own at most 4 MiB of explicitly requested workspace, within its already charged
96-MiB provider allowance. Ordinary callers still request zero. Workspace OOM
or an allowance overrun drops it before rejecting the provider; a numeric arena
OOM also retries without workspace before discarding paired/replay/tile state.

Cleanup drains execution and destroys the handle before freeing its workspace.
Explicit release drains, resets the provider to zero workspace and invalidates
the binding generation; stale tables cannot silently continue. Preparation
remains forbidden during capture. The prototype requests workspace only for
actual DF matrix tiles larger than eight; automatic forces and standalone Q8
defaults retain zero. Larger explicit force caps are not qualified by this
energy-only experiment and must not be advertised as response promotion.

The existing 96-MiB capacity ceiling can stay unchanged while actual retained
allocation rises 4 MiB. This is not memory-free and is not a complete physical
peak/RSS measurement. The pilot's temporary owner receipt records actual active
workspace after numeric fallback; the production integration removes that print.
Explicit allocation/free use the shared resource wrappers, preserving active
ledger limits and allocation-journal generations. Host registry OOM propagates;
only device/budget OOM admits zero workspace. Release restores the caller's device
and drains even a default-stream binding. Pedantic math remains unchanged.

## Qualification sequence

Slurm 2831 rebuilds the frozen source-matched candidate with ccache, checks all
fourteen generated artifacts, exercises ownership/generation/zero/capture
contracts and memcheck, then runs two fresh complete Q32 energy endpoints.
This is feasibility only. Preserve independent total/(T)/expanded-replay gates,
exact semantic work/capacity accounting and both full endpoint records. Record
actual energy-bit equality rather than conflating arithmetic counters with
floating-point audit maxima. Only a useful complete-endpoint signal can justify
further qualification/ABBA and a PR; do not rerun unrelated suites for master.

Recipes, source overlays and receipts are in ignored
`.artifacts/df-cc-blas-workspace-{pilot,candidate}-20261010/` and corresponding
node2 `/data/jzzeng/qc-cc-blas-workspace-{pilot,candidate}-20261010/` directories.
The measured baseline is the qualified frozen #2197 candidate, not latest-master
RHF recomposition. Master observation `4444d0376` includes merged #2197,
allocation/residency observations, correlated acceptance and unrelated DFT work.
The contraction context and CC solver have no drift from the matched baseline;
the production integration adopts current wrappers rather than bypassing them.
No unrelated scientific campaign runs solely because master advances.

## Completed pilot and follow-up

Slurm 2831 completes successfully. All fourteen original artifacts are
byte-identical; ownership/generation/zero/default/extent/capture tests pass,
with memcheck zero errors. The real owner reports an active 4,194,304-byte
workspace after numeric admission, rather than inferring it from a requested cap.

Fresh complete wall is 153.94620775221847 -> 138.82550228503533 seconds
(about -9.82%); complete CCSD 58.184321617 -> 42.711630578 (about -26.59%).
Iteration is 48.299805428101564 -> 33.311652502109375; original expanded replay
9.670487355 -> 9.185948225 seconds. RHF/source/(T) are not credited with a win.
All semantic work, solver counts and reported numeric capacity are identical.
Total energy bits match; (T) differs by -1.0408340855860843e-17. Every independent
gate passes, including expanded physical replay. Do not claim all energy bits
identical or conflate audit maxima with semantic work counters.

The endpoint signal greatly exceeds the two-shape micro estimate: the probe
does not cover every primal/core/replay binding, and summands are not time.
It is not evidence that either measured micro shape explains the entire win.
The active cuBLAS/runtime libraries resolve to CUDA 12.9.1, not a newer provider.

Slurm 2832 adds targeted injected optional-OOM and allowance-pressure fallback
checks plus an actual pedantic-mode check, without allocating the entire GPU.
These are ownership tests, not mocked benchmark measurements. They pass,
and their memcheck has zero errors. The job then runs four fresh ABBA endpoints
using the unmodified libraries from the pilot. Every independent gate passes.

| Median seconds | Zero workspace | Owned 4 MiB |
| --- | ---: | ---: |
| Complete fresh-process wall | 154.23709546145983 | 138.9373795496067 |
| Complete CCSD | 58.211560294 | 42.7339797625 |
| Iteration | 48.325370939742186 | 33.334911327222656 |
| Original expanded replay | 9.672175935 | 9.1862517155 |

Wall time decreases 9.92% (1.1101x); CCSD decreases 26.59% (1.3622x).
Both arms retain 25,792 GEMM calls, 26,308,647,544,608 GEMM summands,
8,945,140,974 bytes conservative numeric capacity, 38 evaluations, and every
other integer semantic work counter. All total-energy bits match; maximum
(T) difference is 1.0408340855860843e-17, not bitwise equality. Floating audit
maxima are independently gated rather than compared as integer work.
The actual extra 4 MiB remains inside the unchanged 96-MiB provider ceiling.
These are two observations per arm on one scoped energy-only problem, not a
formal statistical, latest-master RHF, global, or force/response promotion.

The production integration rebuilds once in Slurm 2833. Its first validation
attempt stops before GPU tests because `/usr/bin/python` lacks pytest. Slurm
2834 passes 38 related solver/typed-binding checks; two checks fail due to a
missing generated-header symlink and a journal test incorrectly using the
frozen pre-journal runtime closure. Slurm 2835 fixes those source/environment
boundaries and reruns only the two failed checks, not the 38 passing checks.
The standalone journal test uses current-master runtime/tensor headers plus
the production context; the molecular library keeps the matched frozen source.
Ledger limits/journal pairing, optional OOM, allowance pressure, stale bindings,
default-stream release and memcheck pass, with zero sanitizer errors.
One real-default final endpoint takes 138.42897410597652 seconds wall and
42.69463147 seconds CCSD. Original independent gates and exact work/capacity
counts pass; total energy bits match, (T) differs by -1.0408340855860843e-17.
No successful ABBA or unrelated force/compiler campaign is repeated.

The successor publication retains all ABBA/pilot records and the one final
integration endpoint in `benchmarks/results/df-cc-blas-workspace-20261010/`.
The six preceding #2197 files (19,127 bytes) are byte-verified against its existing
merged commit `9f67e7806e3151454530baf0ee66ae8808d826f0`, with exact recovery
identities in the Q32 sibling's `snapshot.manifest.json`. No evidence cap changes,
accepted sample loss, external archive, release, or merge authorization is used
to manufacture headroom.
The envelope embeds all seven records and recipe maps together, avoiding repeated
metadata. It references the prior immutable generated hashes and unchanged
source reconstruction rather than duplicating them. All 22 overlay files are
checked byte-exactly; five changed leaf identities and the complete inventory
digest are retained. The aggregate stays at 67,108,766 / 67,108,864 bytes.

## Revisit when

Another device, shape, provider version, capture/lifetime contract or measured
pressure profile justifies a different bounded workspace. Require the same
independent endpoint gates and scoped matched timing; neither the two-shape
microprobe nor Q64's fewer operations certify an endpoint improvement.
