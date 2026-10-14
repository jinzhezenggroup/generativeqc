# Bounded DFT scheduling experiments: #2475–#2479

Qualification is in progress for the complete direct-scheduling cohorts and
tile-eight candidate. No new scheduling default is promoted.

This campaign separates complete energy/analytic-force endpoints from intrusive
profiling. The starting master is
`36574095876e4d12a3945b39cef72f6c77895f00`; later master changes are not silently
included in its timings. #2464 remains a separate point-expression CSE change.
The four-jet experiments apply that prerequisite to both comparison arms and do
not claim its speedup again.

## Protocol and provenance

Every clean point requests cold, five warm, moved, and five moved-warm calls
through the unchanged `benchmarks/run_readme_dft_benchmarks.sh` driver. The whole
point keeps its original 1800-second limit. Each returned energy and force array
is checked against the retained independent geometry oracle at `1e-8 Eh` and
`1e-7 Eh/Bohr`; preparation and force export remain inside the complete endpoint.
The reducer rechecks the raw arrays, source inventory, Python consumers, native
library and reference hashes before publishing any accepted phase. Missing or
failed populations receive no complete-population median.

Scientific inputs retain FP64, screening and SCF tolerances, the moving
48×16×32 grid per atom, and actual iteration/Fock-build histories. r²SCAN-3c uses
canonical def2-mTZVPP, D4(BJ)-EEQ-ATM and native CPU gCP. The ordinary four
functionals use spherical def2-SVP. Direct and fitted Hamiltonians have separate
oracles and tables; their timings never constitute a direct-method speedup.

Endpoint measurements use Slurm RTX 5090 allocations on n1; Nsight Systems uses
finite n4 RTX 5090 allocations. Numerical/sanitizer controls also use n2 RTX PRO
6000. Slurm's device visibility is preserved. Builds use Release, sm_120,
the automatic AOT profile and a verified sccache 0.16.0 launcher. Cache statistics
are retained; shared-server resets/concurrent clients prevent attributing all
reported cache hits to this campaign. No cache was cleared for qualification.

Raw source trees, binaries, scheduler/build/test logs, endpoint arrays and Nsight
databases remain in ignored storage at
`/data/jzzeng/qc-perf-2475-2479-20261014` on the measured nodes, with local copies
under `.artifacts/`. Compact records preserve full-precision timing samples,
work, errors, provenance and reconstruction of dirty pilot sources. A dirty
pilot is identified by its base **plus retained patch/new native files**, not
just its Git revision. Untracked build directories alone also make the driver's
Git dirty flag true; the clean master baseline has no tracked source changes.

## Current-master direct attribution (#2475)

The unmodified source passes all 12 calls at both 48 and 96 atoms (Slurm 8368).
For 96 atoms, complete cold/warm/moved/moved-warm times are
463.117 / 95.380 / 314.350 / 92.741 seconds, where warm cells are medians of all
five calls. Cold/moved use 26/16 Fock builds; each warm call uses one. These
observed counts are not normalized away.

The separately instrumented n4 warm capture (Slurm 889) measures these actual
kernel intervals on the same master library:

| Force consumer | Kernel seconds |
| --- | ---: |
| Generated low + spatial order six | 26.187 |
| Cooperative order seven | 28.016 |
| Materialized order eight | 5.766 |

These are diagnostic device intervals, not clean endpoint samples. Exact kernel
names and launch resources are retained by `summarize_dft_profile.py`. Its
exclusive host ledger and independently partitioned device window each
reconcile, with zero unattributed kernel launches. Host/API/kernel durations
must never be added together. The older observer captured 92.527 seconds inside
the 97.093-second driver interval; the remaining 4.566 seconds include profiler
start/stop outside that observer and remain explicit. The current observer
separately records profiler controls, driver return/final synchronization and
preparation, reconciling the complete endpoint.

Nsight Compute was attempted under Slurm 890 and refused with
`ERR_NVGPUCTRPERM`. Achieved occupancy, warp/barrier stalls, hardware memory
traffic and atomic contention are **unavailable**. Registers, stack/local bytes,
shared memory and launch shape are resource bounds, not substitutes for those
measurements.

The initial diagnostic's warm census records 380,073,488 generic order-eight
candidates, 379,911,128 angular rejects, 96,168 admitted dddd shell quartets and
95,976 actually executed dddd quartets/primitive products. Order seven executes
811,824 shell quartets and 1,250,544 primitive products. These are actual consumer
counters; they are distinct from the unscreened combinatorial domain. Low-order
products count precontracted shell traversals; order-six products count admitted
AO-component traversals. A unique executed order-six shell count is unavailable.

This motivates two different hypotheses: avoid generic order-eight angular
filtering using the prepared dd-pair stream, and test whether shorter existing
dd×dp pages reduce order-seven CTA tail imbalance. Neither hypothesis changes
the generated recurrence or establishes a speedup without clean endpoints.

## AO discovery (#2476)

Both schedules call the same generated AO-jet producer and exact sampled
`abs(jet) > 1e-16` predicate. The device candidate performs stable compaction into
the existing reserved map, borrows dead scratch and downloads offsets at bounded
chunk boundaries. No additional device allocation or full-grid bitmap is added.
Geometry ownership, transactional publication and dense resource fallback stay
with the existing plan.

The preparation-only trace observed the following counts for all four
functionals, with identical active sums and AO work between schedules:

| Atoms | Tiles | Route | Synchronizations | D2H bytes | H2D bytes | Active AO sum |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 48 | 4,608 | host | 4,609 | 7,096,320 | 4,971,104 | 621,388 |
| 48 | 4,608 | device | 36 | 37,008 | 0 | 621,388 |
| 96 | 9,216 | host | 9,217 | 28,348,416 | 11,683,392 | 1,460,424 |
| 96 | 9,216 | device | 72 | 74,016 | 0 | 1,460,424 |

The 96-atom discovery evaluates 7,247,757,312 AO jet values in either arm. Its
roughly one-second cost is small relative to complete cold/moved calls. For
M06-2X/96, host versus device complete cold is 153.369 versus 153.482 seconds and
moved is 115.195 versus 114.796 seconds. Other moved trials change SCF work
(including B3LYP 12→15 and r2SCAN 13→12 builds); these changes remain in the
endpoint observations. Large D2H/fence reductions do not establish a repeatable
complete cold/moved improvement, so **host discovery remains the default**.

The matched AO binary also covers 3/12-atom host/device/dense controls. Independent
CPU E/V, sorted sampled maps, spherical/cartesian s/p/d/f expansion, empty/tail
maps, multiple ballot waves, tiny scratch, capacity refusal and failed-retry
tests pass on real GPUs. Compute Sanitizer reports zero errors. All five AO
production files in the initial endpoint image are byte-identical to this PR's
AO implementation; later changes extend force diagnostics and qualification.

## Four-jet contractions after CSE (#2478)

The matched 16/32 builds have the same post-CSE source, AO radial reuse setting,
scientific formulas, active maps and point/force protocol. The only CMake
difference is `GENERATIVEQC_CUDA_XC_MATRIX_TILE`. In the 96-atom direct warm
diagnostic, both M06-2X and r2SCAN execute 4,520 batched launches for each matrix
consumer:

| Matrix tile | Density products (s) | Symmetric Vxc (s) | Sum (s) |
| ---: | ---: | ---: | ---: |
| 16 | 0.924 | 1.342 | 2.267 |
| 32 | 1.100 | 3.705 | 4.805 |

The separate force geometry-point time is about 1.246 seconds for M06-2X and
0.675 seconds for r2SCAN. It is not another SCF contraction. The larger tile
increases CTA width from 256 to 1024 lanes and Vxc static shared memory from
8,720 to 33,808 bytes, while retaining 48 registers per thread. The measured
regression rules out promotion; resource metadata alone cannot identify the
unavailable occupancy/stall counters as its cause. The smaller tile-eight
experiment and complete endpoint results are retained separately.

## Reproduction

Build with verified CXX/CUDA compiler launchers and the same generated/AOT
configuration. Select one immutable installed image with
`GENERATIVEQC_LIBRARY`, retain the exact basis and reference files, then run the
ordinary README runner inside finite `srun`. Runtime ablations are:

```text
GENERATIVEQC_CUDA_KS_AO_DISCOVERY=host|device
GENERATIVEQC_EXPERIMENT_DDDD_FORCE_CLASS_MAJOR=0|1
GENERATIVEQC_EXPERIMENT_ORDER7_FORCE_PAGE=32|16|8
```

Keep diagnostics off for timing populations. For separate attribution, use
`python -m benchmarks.profile_dft_endpoint --capture warm ... --repeats 5`
under Nsight Systems and export its SQLite database. Reduce it with
`benchmarks/summarize_dft_profile.py`. `GENERATIVEQC_DIRECT_FORCE_CENSUS` and
`GENERATIVEQC_DF_TRACE` enable actual work/event records. The optional
`GENERATIVEQC_DIRECT_FORCE_CENSUS_SPLIT_LOW=1` splits low/order-six diagnostics
by repeating enumeration; its kernel times cannot replace the incumbent's
combined-kernel endpoint timing. `null` execution/recurrence fields mean
unavailable, not zero.

Recheck all raw endpoint populations and reconstruct their source bindings:

```bash
PYTHONPATH=.:python python benchmarks/results/bounded-dft-schedules-2475-2479/reduce.py \
  /data/jzzeng/qc-perf-2475-2479-20261014 \
  --output summary.json.xz --table endpoints.md
```
