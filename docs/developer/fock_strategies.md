# Fock provider dispatch and strategy selection

This page owns how a Fock build request is resolved into actual integral
sources, and how the resolved strategy is selected for CPU and CUDA. The
mathematical `FockBuildSpec` contract, public `FockPlan` API, output semantics
and density/coefficient conventions are owned by
[the shared Fock construction contract](fock_build.md); this page covers
dispatch, source selection and the exact-K scheduling policy.

## Resolution model

Presence is not a coefficient. A term that is present with coefficient zero
still requests its raw matrix, and an absent term performs no work. Coulomb and
exchange are resolved independently, so a plan may request, for example, a
fitted Coulomb term and an exact exchange term. `AUTO` never authorizes changing
an exact Hamiltonian into a fitted one: the only way to obtain fitted
mathematics is an explicit `density_fitted` term.

| Boundary | Current responsibility |
| --- | --- |
| `FockBuildSpec` | Mathematical request: spin, derivative order, and independent Coulomb/exchange presence, coefficient, operator, range parameter and approximation |
| `ResolvedFockBuild` | Validates the request and separates it from backend, schedule, screening and DF metric threshold |
| `FockPlan` / `CpuFockProviderView` | Binds immutable CPU integral owners; shares one provider call when both terms use the same source |
| `CudaFockProviderView` | Binds the same composition to CUDA direct and DF plan items, with typed output selection |
| `CpuFockPlanView` | Validates both sources before execution |
| `run_cpu_fock_strategy` / `run_fock_strategy` | Shared CPU iteration and finalization; accepts explicit mixed semantics |

CPU RHF/UHF direct and fitted entry points use the same iteration and
finalization code. Fock, energy and first derivatives use one binding; the DF
metric cutoff and complete spectrally truncated response are retained. The
source owner's DF buffers stay inside CPU resource observations.

Standard CUDA HF keeps its established fused solver. General CUDA requests that
need independent terms select the independent path: CUDA one-electron
generation, direct/DF J/K and matching two-electron response feed the shared
host SCF iteration and finalization. The direct provider reuses the contracted-ERI
value/Dual evaluator and retains metadata, AO matrices and compact gradient
storage without molecular ERI tensors. Bounds and computed outputs reject
nonfinite results transactionally, and absent or zero-weight responses do not
evaluate unused quadratic terms.

CUDA DF raw services accept a typed output selection for resident, streamed,
tiled, batch, item and device-pointer execution. Unselected device outputs may
be null and are never accessed; host outputs are empty. Plan scratch capacity
remains accounted and available for subsequent selections.

`PreparedFockPlan` owns the shared views and the underlying sources. Independent
CUDA single and fleet replays retain it per item with immutable-input and
backend-variant compatibility checks and transactional replacement. CPU SCF
keeps its transient ERI lifetime. Compatibility covers unchanged requests and
rejects changed geometry, coefficient, auxiliary basis, cutoff, device identity
and failed replacement.

The additive C API in `include/generativeqc/fock.h` and the Python `FockPlan`
expose independent J/K choices, fixed-density values and responses, complete
native SCF, explicit warm replay, and requested/resolved/actual-source
diagnostics. Mathematical identity stays separate from execution/source
identity.

## Fixed-density semilocal coupling

`FixedDensityMeanField` connects a native unit-Coulomb, absent-exchange request
to the executable semilocal XC integrator. LDA and PBE keep independent
fixed-density fixture and energy-variation checks. This is the fixed-density
mean-field consumer, not DFT SCF; see
[the shared Fock construction contract](fock_build.md#fixed-density-semilocal-consumer)
for the request construction and the MethodIR exact-exchange lowering.

## Exact-K task scheduling

The generated raw exact-K owner freezes
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=incumbent|fill|primitive|work` at
preparation. Unset and empty select `work` by default. Explicit `fill` restores
the previous bounded cross-chunk schedule; `incumbent` retains per-original-chunk
execution for rollback and comparisons. The prepared `DirectExchangeSelection`
freezes this axis together with the K lowering masks, so topology and launcher
selection cannot disagree after later environment changes.

With both controls unset, the qualified [Rys-task preference](direct_rys_tasks.md)
takes priority for its covered classes with its own Work companion; incumbent
work kernels handle the remaining eligible classes.

`fill` holds accepted identities and contribution bounds across chunks in a
bounded 2W arena, consumes full W-task batches, and flushes the final partial
batch. Experimental `primitive` additionally stably groups the admitted
lookahead window by cached ket primitive-pair count; the bra and angular class
are fixed within that queue.

The default `work` schedule retains eight bounded survivor queues per bra:
cached ket primitive-pair work has bins 1, 2–3, 4–7 and 8+, each split by
whether both ket shells are contracted. Bra work and angular class are fixed,
so this groups primitive-product work without reordering the screening domain.
The 8+ bin is deliberately saturated; it does not cap primitive traversal or
promise a work-ratio bound for arbitrarily long custom contractions.

Each queue has a 2W arena. New candidate chunks are scanned only while every
queue has fewer than W pending tasks. Full expensive bins drain first; partial
bins accumulate across original chunks and flush independently at the row tail.
Storage is bounded by 16W identities/bounds per independently queued group, not
by the quartet domain. Task-parallel Rys uses four independent warp groups with
W=32 and 24,736 bytes of shared queue storage per CTA. Separate
class-specialized work kernels keep this storage out of the incumbent, fill and
primitive kernels. Whole-CTA, native and unsupported classes retain their
existing workers, including high-angular generated classes beyond
`psss`/`psps`. Task-parallel Rys has a separate Work companion sharing its
quartet arithmetic; explicit component-Rys/block alternatives are unchanged.
Unknown values fail at preparation. Change the setting before creating a new
owner, not while replaying an existing one.

The control affects covered packed/subgroup generated raw-K streams only.
Coulomb, combined-HF preparation, other schedules and unsupported classes retain
their existing selection. Recurrence selection remains independent. Screening
scans the original Schwarz-ordered stream once; grouping never resorts unscreened
inputs. Matrix semantics, precision policy, thresholds and nonsymmetric/spin
contracts do not change, and no quartet-domain allocation is introduced.
Rys-task retains its value producer and selects its warp-private Work companion
for that schedule. Component-Rys and restricted-only block lowerings retain
their own workers; incumbent work dispatch applies where no alternative is
selected. UHF cannot use restricted block contraction and follows the frozen
work/incumbent fallback instead. `fill` alone does not disable the automatic
Rys-task preference; restoring the pre-promotion generated defaults requires
both `GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent` and
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=fill`. Set the latter to `incumbent` for
per-original-chunk execution instead. Neither rollback changes Coulomb or forces.

Rationale and development evidence for these defaults are retained in the
[queue decision note](../../.agents/notes/implemented/performance/2026-10-07-direct-k-cross-chunk-queue.md),
the [fill-default decision](../../.agents/notes/implemented/performance/2026-10-07-direct-k-fill-default.md),
the [work-bucket decision](../../.agents/notes/implemented/performance/2026-10-08-direct-k-work-buckets.md),
the [work-default decision](../../.agents/notes/implemented/performance/2026-10-09-direct-k-work-default.md)
and the [task Work decision](../../.agents/notes/implemented/performance/2026-10-09-rys-task-work-buckets.md).

## Scope and rollback

These selectors change scheduling only; they do not change the mathematical
request. Unsupported range-separated operators remain explicit and fail before
execution. New providers must extend this common boundary rather than introduce
another method-specific selector or cache.

## Validation

Dispatch changes are accepted with fixed-density raw matrices at matched
approximation, then complete CPU/CUDA energy-plus-force, replay,
changed-geometry, ragged-batch and dispatch-overhead endpoints under matched
final accuracy. Retention, source identity and publication rules follow
[evidence retention](../maintainer/evidence_retention.md); the archived
source-matched non-regression study is
[benchmarks/results/fock-strategies](../../benchmarks/results/fock-strategies/README.md)
with its rationale in the
[strategy-comparison note](../../.agents/notes/implemented/performance/2026-10-09-fock-strategy-comparison.md).

For a complete PBE0 cold trace,
`benchmarks/pbe0_k_work_profile.py --trace trace.sqlite --endpoint cold.json --output classes.json`
validates paired Coulomb-then-K class passes, includes native `dddd`, and reports
K time separately from Coulomb and force work. Trace sums are intrusive device
diagnostics, not clean endpoint timing or primitive-work counts.
