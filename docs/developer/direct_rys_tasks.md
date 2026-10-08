# Independent task-parallel Direct Rys-K

The qualified `sm_120` profile defaults to value-only `_rys_task` AOT variants
for `psps`, `ppps`, `dsss`, `dpss` and `dsps` Direct exchange classes. Other
classes and profiles retain the incumbent. This selection freezes at provider
preparation; it does not reselect the old component-lane `rys` experiment,
select Coulomb J, or select analytic derivatives. Set
`GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent` for the complete rollback.

## Compiler and execution ownership

`integral/production_rys_tasks.py` intersects the compiled streaming-Fock
inventory with `rys_task.py`'s bounded value capability. Eligibility requires
one or two Rys roots, s/p/d shells, at most a p shell on the fourth center,
and at most 27 Cartesian components. Capability and measured preference are
separate: `preferred_rys_task_candidates` records the five qualified classes
only for the `sm_120` profile. Portable profiles and other architectures retain
their incumbent until independently qualified.

`lowering/fock_rys_task.py` changes execution ownership: each lane in a
128-thread packed CTA owns a complete admitted quartet. Its four 32-lane
warps claim independent bra rows and keep separate bounded survivor queues.
An exhausted warp retires without stranding another at a CTA barrier. Queue
storage remains bounded by two 32-task batches per warp. Primitive geometry, roots,
TRR/HRR values and Cartesian integral accumulators are lane-local. There are
no barriers inside the primitive/root loops. Streaming queue collectives are
warp-local; ordinary persistent entry points retain their CTA claim barriers.
Raw and streaming task/storage are lane-private. The existing primitive-pair cache, pair
orientation, cross-chunk filling, Schwarz and cross-density admission remain
authoritative; this path does not build a second task queue.

The value body reuses `RysState`, `build_rys_axis_program` and `_state_expression`.
The one-root rule uses the existing strict FP64 Boys owner with weight F0 and
squared node F1/F0. The two-root rule uses the existing high-accuracy tables,
inlined for lane ownership. AO normalization follows the primitive sum.
No host integral oracle, CPU reference evaluation, or runtime source generation
is part of the production execution contract.

Restricted raw-K tasks contract into bounded local blocks before atomic
publication. Their maximum footprint is 64 doubles, without changing the
existing Hermite K-block experiment's 32-double default. UHF, combined J/K and
HF-weighted K use the established generic symmetry scatter. This is not an
atomic-free design and makes no claim of lower register pressure.

## Selection and fallback

`GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES` optionally restricts candidate
classes, using the existing comma-separated exact-class selection convention.
Unset, empty, or `all` permits the compiled candidate inventory; `none` permits
none. With the lowering selector unset or empty, this filter can only restrict
the qualified five-class preference, not expand it. Explicit `rys-task` selects
the full permitted capability inventory (eight classes on `sm_120`), which is
an experiment rather than a generally faster configuration. Both selections
intersect enabled incumbent Fock coverage and freeze in the prepared K owner.
Changing the environment after preparation does not change that owner. There
is no molecule-, method-, size-, or density-dependent promotion table in the
compiler or runtime.

An example explicit selection is:

```bash
export GENERATIVEQC_DIRECT_K_FOCK_LOWERING=rys-task
export GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES=psps,ppps,dsss,dpss,dsps
```

Unselected or unsupported classes retain the incumbent exact path. Optional
storage refusal retains the existing bounded provider fallback. A failed
selected launch propagates its error; the runtime never retries into a
partially accumulated K matrix. Range-separated operators keep their existing
owners. `incumbent` is the complete rollback; `rys` and `block` remain independent
experiments.

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
one. Unset the class filter too when qualifying the untouched default.
Use `--samples` to summarize at least three processes per mode. The summary
rejects incomplete, inaccurate, primed, duplicate-process or unmatched
library/protocol/host/Slurm-device/class selections and requires both median
and mean improvement. Actual SCF/Fock counts
accompany complete timings: a diagnostic fixed-density K gain is not a Cold
endpoint gain. The retained compatibility fixture separately pins all
incumbent, old Rys-value and old K-block generated mathematics while allowing
the independent candidate to change bundle/registry identity.
