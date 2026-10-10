# Independent task-parallel Direct Rys-K

Compiled non-portable CUDA profiles default to value-only `_rys_task` AOT variants
for `psps`, `ppps`, `dsss`, `dpss`, `dsps`, `ddss`, `dsds`, `dpps` and `dspp`
Direct exchange classes when the target supports their packed-task schedule.
Other classes and generic portable profiles retain the incumbent recurrence
with the independently selected queue schedule. Selection freezes at provider
preparation; it does not reselect the old component-lane `rys` experiment,
select Coulomb J, or select analytic derivatives. Set
`GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent` to disable task preference;
also set `GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=fill` to roll back both promotions.

## Compiler and execution ownership

`integral/production_rys_tasks.py` intersects the compiled streaming-Fock
inventory with `rys_task.py`'s bounded value capability. Eligibility requires
one through three Rys roots, s/p/d shells, at most a p shell on the fourth center,
and at most 64 Cartesian components. `preferred_rys_task_candidates` uses the
same nine-class algorithmic policy on any compiled non-portable target. The
compiler validates the actual target's schedule and the runtime intersects
enabled incumbent coverage. A target need not repeat the full scientific
numerical matrix just because its compute capability differs. Performance
results remain device-specific evidence, not an assumed cross-GPU speedup.

`lowering/fock_rys_task.py` changes execution ownership: each lane in a
128-thread packed CTA owns a complete admitted quartet. Its four 32-lane
warps claim independent bra rows and keep separate bounded survivor queues.
An exhausted warp retires without stranding another at a CTA barrier. Queue
storage remains bounded by two 32-task batches per warp per work bin. Primitive geometry, roots,
TRR/HRR values and Cartesian integral accumulators are lane-local. There are
no barriers inside the primitive/root loops. Streaming queue collectives are
warp-local; ordinary persistent entry points retain their CTA claim barriers.
Raw and streaming task/storage are lane-private. The existing primitive-pair cache, pair
orientation, cross-chunk filling, Schwarz and cross-density admission remain
authoritative; this path does not build a second task queue.

The value body reuses `RysState`, `build_rys_axis_program` and `_state_expression`.
The one-root rule uses the existing strict FP64 Boys owner with weight F0 and
squared node F1/F0. The two-root rule uses the existing high-accuracy tables,
inlined for lane ownership. Three-root tasks reuse the same high-accuracy
table owner, with bounded polynomial unrolling to limit live coefficient state.
Classes above 32 components use a 64-bit retained-component mask; smaller
classes retain their existing 32-bit generated source. AO normalization follows
the primitive sum.
No host integral oracle, CPU reference evaluation, or runtime source generation
is part of the production execution contract.

Restricted raw-K tasks contract into bounded local blocks before atomic
publication. Their maximum footprint is 80 doubles, without changing the
existing Hermite K-block experiment's 32-double default. UHF, combined J/K and
HF-weighted K use the established generic symmetry scatter. This is not an
atomic-free design. Classes exceeding that private contraction bound retain
generic scatter, not an unbounded local array. Static register counts do not
substitute for measured occupancy or complete-endpoint performance.

## Selection and fallback

`GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES` optionally restricts candidate
classes, using the existing comma-separated exact-class selection convention.
Unset, empty, or `all` permits the compiled candidate inventory; `none` permits
none. With the lowering selector unset or empty, this filter can only restrict
the nine-class algorithmic preference, not expand it. Explicit `rys-task` selects
the full permitted capability inventory (twelve classes on `sm_120`), which is
an experiment rather than a generally faster configuration. Both selections
intersect enabled incumbent Fock coverage and freeze in the prepared K owner.
Changing the environment after preparation does not change that owner. There
is no molecule-, method-, size-, or density-dependent promotion table in the
compiler or runtime.

Current qualification and rejected current-main alternatives are recorded in
the [three-root execution decision](../../.agents/notes/implemented/performance/2026-10-08-rys-task-three-root-k.md).
Historical frozen-base five-class receipts retain their original source identity.

An example explicit selection is:

```bash
export GENERATIVEQC_DIRECT_K_FOCK_LOWERING=rys-task
export GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES=psps,ppps,dsss,dpss,dsps
```

One prepared `DirectExchangeSelection` owns both the lowering masks and
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE`. Default dispatch is qualified Rys-task
first, then the separately compiled work-bucket worker for remaining eligible
classes, then the incumbent worker for unsupported work classes. Explicit
`rys` and `block` disable automatic Rys-task preference and keep their own
workers; block contraction is restricted-only, with the frozen schedule's
fallback for UHF. A selected Rys-task producer uses its own separately compiled
work companion for schedule `work`, reusing the exact same quartet arithmetic.
Each warp owns eight two-batch arenas, partitioned by the existing ket-pair
primitive-work classifier. The four-warp queue uses 24,736 bytes of bounded
shared storage and only warp-local collectives. `fill`, `primitive` and
`incumbent` retain the original task worker and its smaller shared footprint;
unsupported work classes retain their explicit incumbent fallback.
See the [task Work queue decision](../../.agents/notes/implemented/performance/2026-10-09-rys-task-work-buckets.md)
for ownership, bounded-storage invariants and qualification limits.

Unselected or unsupported alternatives retain the incumbent exact recurrence
with the frozen queue schedule. Optional
storage refusal retains the existing bounded provider fallback. A failed
selected launch propagates its error; the runtime never retries into a
partially accumulated K matrix. Range-separated operators keep their existing
owners. Lowering `incumbent` disables Rys-task preference but does not reset
the independent task schedule. For complete rollback to the pre-promotion
default, set both lowering `incumbent` and task schedule `fill`; task schedule
`incumbent` instead restores per-original-chunk execution. `rys` and `block`
remain independent experiments. See the
[combined selector decision](../../.agents/notes/implemented/architecture/2026-10-09-unified-direct-k-selector.md).

## Qualification

`tests/python/test_direct_rys_values_cuda.py` accepts
`GENERATIVEQC_RYS_VALUE_FAMILY=task` and an explicitly compiled production-shard
library through `GENERATIVEQC_RYS_VALUE_LIBRARY`. It checks both spins and
combined/J/raw-K/HF-weighted-K matrices against Libcint, including signed unequal
primitives, coincident/reversed pairs, one task, a 33-task warp tail and a
129-task CTA tail. These
checks require `GENERATIVEQC_RESOURCE_CUDA_TEST=1` and a finite Slurm GPU job.

`benchmarks/rys_task_cold.py` measures complete fresh-process PBE0 E+F for the
48- or 96-atom water case, spherical def2-SVP and a 48x16x32 grid. Construction,
preparation, the first strict FP64 SCF/analytic force call, host force return and
owner teardown are timed. Imports/input decoding are excluded; persistent
compiler/artifact caches are shared equally. No density, CUDA context, or
prepared owner is primed. Supply an independent reference matching its exact
protocol; acceptance is 1e-8 Ha energy and 1e-7 Ha/Bohr maximum force error.

Run each worker in a separate process under the same finite Slurm allocation,
alternating modes, and retain every result. Set `GENERATIVEQC_LIBRARY` and
`PYTHONPATH` to the qualified build/checkout first. For example, inside that job:

```bash
python -m benchmarks.rys_task_cold --mode incumbent --atoms 96 \
  --reference reference-96.json --output control0.json
python -m benchmarks.rys_task_cold --mode default --atoms 96 \
  --reference reference-96.json --output candidate0.json
```

The `default` worker unsets the lowering selector even if its parent exports
one. The task schedule is a separate, fixed comparison axis: unset it and the
class filter when qualifying the untouched combined default. New v2 records
retain the effective task schedule, and the summary rejects mismatched schedules;
historical v1 records remain explicitly unspecified rather than being relabeled
as work-scheduled qualification. An `incumbent` worker with the schedule unset
is a work-only baseline, not a rollback of both promotions.
Use `--samples` to summarize at least three processes per mode. The summary
rejects incomplete, inaccurate, primed, duplicate-process or unmatched
library/protocol/host/Slurm-device/class selections and requires both median
and mean improvement. Actual SCF/Fock counts
accompany complete timings: a diagnostic fixed-density K gain is not a Cold
endpoint gain. The retained compatibility fixture separately pins all
incumbent, old Rys-value and old K-block generated mathematics while allowing
the independent candidate to change bundle/registry identity.
