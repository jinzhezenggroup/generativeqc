# Decision: default to the qualified five-class Rys-K task schedule

Status: implemented
Date: 2026-10-08

## Problem

The [independent task experiment](2026-10-08-rys-task-k.md) produced opt-in Cold
gains, but fresh unset-selector qualification of the original 32-thread,
five-class configuration regressed. Both valid negative cohorts are retained:
job 6665 medians 113.719/119.571 s, Focks [18,17,17]/[17,19,19]; job 6668
112.808/118.552 s, Focks [17,18,17]/[19,22,19]. These are not stale-build
exclusions. Their production generated mathematics are byte-identical; the
cause of the different SCF trajectories is not proven. Earlier opt-in gains
do not qualify an unchanged default promotion.

## Decision

Use a 128-thread CTA with four independently claiming and retiring 32-lane
warp queues. Each warp owns its bra cursor and two-batch survivor scratch,
sharing only the existing global bra head. No CTA barrier remains in the
streaming worker after a warp can retire. Per-quartet roots, primitive order
and axis recurrence remain unchanged. Ordinary/persistent entry points use
the actual task width and lane-private storage rather than growing shared
frames. This extends scheduling, not the scientific integral algebra.

For the compiled `sm_120` profile only, unset/empty K lowering selects the five
qualified classes `psps`, `ppps`, `dsss`, `dpss`, `dsps`. Compiler registry
metadata keeps measured preference separate from full capability. The native
prepared K owner freezes that preference intersected with both the class filter
and enabled incumbent coverage. No method/molecule/density policy enters the
compiler. Portable and other profiles keep a zero preference. Explicit
`incumbent` rolls back all five; explicit `rys-task`, old `rys`, and `block`
retain independent experimental semantics. J, analytic force lowering and SCF
tolerances are unchanged.

## Invariants

Preserve exact admission, primitive-pair reuse, FP64 arithmetic, independent
oracle gates and optional-storage bounded fallbacks. Never retry a failed
selected launch into a partially written output. Class-filter `all` must not
expand the default beyond five. Environment changes after preparation cannot
change an existing owner. New targets require separate qualification.

## Evidence

The original opt-in receipts remain under
`benchmarks/results/rys-task-cold-20261008/`; their measurements are not relabeled
as default measurements. The default-promotion receipts separately exercise an
unset lowering selector and class filter, with effective prepared masks,
independent integrated matrix checks, fresh-process complete 48/96-atom E+F,
actual Fock counts and mean/median acceptance. Benchmark workers actively
remove an inherited selector rather than silently benchmarking opt-in mode.
The dispatch identity refresh changes the registry. The multiwarp schedule
changes only task sections in sm_120 shards 0/1/2; the empty task section in
shard 3, portable artifacts and retained incumbent/component-Rys/block
mathematics stay byte-identical. Per-quartet Rys recurrence is not replaced.

Job 6672 passes both performance gates: 96-atom medians 113.514541 to
110.314194 s (2.82% less), but mean improvement is only 0.22%, retaining
[17,17,17]/[19,17,17] Fock counts. At 48 atoms the median improves 4.71%, with
19 Focks throughout. The small 96-atom mean margin motivates an independent
five-pair untouched-default confirmation, retained with all rejected trials in
the follow-up report, rather than a strong reliability claim from three pairs.

Job 6674 confirms both gates with five new pairs: medians 123.137497 to
114.822158 s (6.75% less), means 119.373880 to 113.906666 s (4.58% less).
Fock counts are [17,17,19,19,19]/[17,17,18,18,19]; the median improvement is
not a same-iteration kernel ratio. See the full retained
[follow-up receipts](../../../../benchmarks/results/rys-task-default-20261008/README.md).

The wide schedule passes 576 Libcint matrices including 129-task CTA tails.
Memcheck/racecheck/synccheck each pass 64 cases with zero findings. Native
integrated matrices check both spins, density restoration and environment
freezing. Fixed-density three-way diagnostics match admission vectors at all
scales: 81,907,624 quartets at 96 atoms and 23,480,495 at 48.

## Rejected alternatives

Do not promote all eight classes or other architectures based on the 5090
result. Do not treat fixed-density K timings as complete Cold wins, omit slower
converged trajectories, or infer active-lane/barrier counter contributions:
NCU performance counters were unavailable. Selecting only `psps/ppps`, or
only `dsss/dpss/dsps`, also fails both Cold performance gates (job 6673), despite
accurate energies and forces. Default promotion requires fresh
unset-selector complete-endpoint evidence, not just opt-in receipts.

## Revisit when

Requalify target/class preference when schedules, roots, contraction bounds,
admission, compiler options or endpoint semantics change. Expand preference
only with independent correctness and complete-endpoint evidence. Preserve
the explicit rollback and bounded resource fallback.
