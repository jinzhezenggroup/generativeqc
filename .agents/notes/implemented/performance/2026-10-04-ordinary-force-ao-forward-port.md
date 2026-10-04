# Decision: forward-port the existing ordinary force AO consumer

Status: implemented opt-in; current-source GPU qualification pending
Date: 2026-10-04

## Why this is integration, not a new optimization

At master `e79efea00`, #1772's bounded resident cache and #1774/#1780's native
SCF local-AO support are present, but ordinary stationary forces still pass
`None` to `feature_task_device_points`. #1779 is closed without a merge; #1761's
current diff integrates the composite nonlocal consumer, not this ordinary
caller. Combining master with #1830 therefore does not automatically reproduce
the previously measured force-local-AO candidate.

Port only #1779's caller and focused tests from `a718695de`, not its unrelated
native fixes, historical storage changes or stale source hashes. Preserve the
current compiler tile planner and concurrent-owner reservations. The shared
discovery producer, cache, AO/density gather and force scatter are unchanged;
there is no new scientific kernel or independent cache implementation.

## Work and memory contract

For tile sizes B_t, full AO count M and selected counts m_t, dense density/AO
contraction work is proportional to sum_t B_t M^2; selection changes that term
to sum_t B_t m_t^2. First use or a changed geometry additionally pays the shared
discovery traversal. These expressions are work models, not measured FLOPs or
an unconditional asymptotic improvement. Partition response still visits the
full grid and atom-pair domain, including tiles with no active AO.

The cache is optional, default off, capped by the remaining total host budget
after the complete tile and prepared TensorIR owners are admitted. Full device
capacity remains charged; mean active AO count is not an allocation bound.
Maps cover the configured force derivative order and may be reused only for
the same checked geometry/grid/basis generations and resident owner. Density
is rebound independently on every complete call. Budget/capability misses keep
the dense fallback, not a relaxed cutoff or partial result.

The strict capacity audit now binds these helpers and prepared-request policy.
After the earlier tile-planner refactor, the optional map charge must follow
successful complete tile selection and precede artifact lookup. Preserve that
execution order rather than copying the old source-file ordering check.

The shared benchmark normalizer retains the caller's `resident_ao_selection`
policy and counters; otherwise README endpoint records would drop the actual
local-AO work again. Absent telemetry stays null, not a zero-work claim, and
point-weighted AO-square counts are not labeled FLOPs.

## Evidence and limitations

The historical #1779 complete96 warm result (77.159 to 62.792 seconds) belongs
to its frozen source/library, not this port. Its first-use discovery cost,
cold/moved regressions and original independent error gates remain relevant.
Do not add that percentage to #1767 or #1830's independently measured gains.

The initial current-source focused host suite passes 287 tests on n2: caller
admission/lifetimes, actual empty/noncontiguous AO loop execution, failed drain
handling, dense-default orchestration and exact capacity/mutation checks. This
does not replace current-source molecular numerical or performance validation.
No GPU job was launched for this port, no approximation was enabled by default,
and no n3 job was submitted.

The expanded host suite, including the shared map cache and telemetry schema,
passes 322 tests on the same current source. Compiler structure checks 422
modules with zero errors; repository-pinned Ruff 0.16.9 and diff checks pass.
This supersedes the focused 287 count rather than adding to it.

## Required next validation

Freeze one integrated source and matching ccache-built library containing this
consumer, existing exact force paging and optional #1830 phases. On one GPU,
compare default / paging+local-AO / paging+local-AO+phases with identical cutoff,
tile, memory and numerical settings. Retain cold, five warm, moved, five
moved-warm and zero-cache fallback. Verify work-map invalidation, actual phase
execution and timer normalization before allocating GPU time. Do not splice
prior job 5728's incomplete campaign with its separate job 5729 qualification.

After that baseline, expensive DPPP/DPPS and DDDP/DPDP classes are the next
integral scheduling candidates. Existing class-isolated timings suggest those
classes ahead of DDDD, but every isolation still traverses screening and is
not an additive endpoint decomposition. Primitive capacity is not executed
root count; #1787's uniform arithmetic calibration does not establish timing
for heterogeneous recurrence/spill kernels. Do not repeat the rejected scalar
DPPP weighted slab or promise a speedup from static stack size alone.
