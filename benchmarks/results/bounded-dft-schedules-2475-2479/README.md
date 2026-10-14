# Bounded DFT scheduling experiments: #2475–#2479

Qualification retains the existing defaults. Of 110 planned 12-call populations,
109 complete and 1 reaches the unchanged 1800-second point limit.
All 1,317 returned native calls pass the independent energy/force gates;
missing timeout calls remain missing. Tile 8 reduces four-jet matrix device time,
but none of these experiments establishes a robust universal default change.

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
library and reference hashes before publishing any accepted phase. A source
binding with missing, partial or mismatched native/consumer receipts is explicitly
unqualified and cannot publish accepted timing. Missing or
failed phases receive no complete-phase median. A complete earlier phase stays
visible even if the whole point later times out; it never qualifies the full
12-call population.

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
under `.artifacts/`. Compact records preserve full-precision endpoint timing samples, per-call
work, final residuals/errors, provenance and reconstruction of dirty pilot sources.
Full SCF component-energy trajectories, raw force vectors, clean-run component
clocks and intermediate progress events remain in the hash-bound ignored inputs.
Independent cold/moved oracle vectors and intrusive attribution ledgers are kept
in the publication. Native inventory digests replace duplicate per-file hash maps;
the revision plus patch/new files reconstruct the same inventory. A dirty
pilot is identified by its base **plus retained patch/new native files**, not
just its Git revision. Untracked build directories alone also make the driver's
Git dirty flag true; the clean master baseline has no tracked source changes.

## Frozen-master direct attribution (#2475)

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

The same capture's **exclusive host** partition is below. Each child is
subtracted from its parent; CUDA kernel intervals are an independent view and
are never added to this table.

| Exclusive warm owner | Seconds |
| --- | ---: |
| SCF exact J, including its transforms/screening | 7.648 |
| SCF XC/AO/grid | 5.707 |
| Remaining SCF/publication | 0.222 |
| Stationary J′ | 59.977 |
| Stationary one-electron sources | 2.274 |
| Remaining force/grid/Becke/publication | 9.810 |
| Final-state snapshot, decode, validation | 6.435 |
| D4 / native CPU gCP | 0.445 / 0.002 |
| Public wrapper and nested calculator construction | 0.006 |
| Outside the older observer (including profiler controls) | 4.566 |
| Complete driver warm time | **97.093** |

The broad force/grid row contains moving-grid response, transfers, fences and
host assembly that cannot be individually separated by this capture. It is an
explicit 9.810-second residual owner, not proof of zero transfer/host cost.
Exact unrounded intervals are archived. Warm preparation is zero; the diagnostic
cold record separately retains constructor/preparation time and nested scopes.

Both timing views are diagnostic observations, not clean endpoint samples. Exact kernel
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

The separate split-low diagnostic (Slurm 8408) measures 4.399 seconds generated
low classes and 21.673 seconds spatial order six, with 27.885 seconds order seven
and 5.742 seconds order eight. This **repeats low/order-six enumeration** and
changes specialization; it is not an exact split of the unchanged kernel.
Classes 14/17/18 execute 1,308,211,472 / 566,613,312 / 525,680,128 primitive-pair
products per admitted AO component. Order seven executes 1,250,544 Coulomb
preparations. These counters point to the remaining critical path:

- `src/scf/cuda/direct_force_quartet.cuh:158` still enters a primitive product
  for each surviving order-six AO component. Reusing a spatial ladder across
  components is a future generated-source hypothesis, with the potential to
  remove repeated preparation from this approximately 22-second consumer.
- `python/generativeqc_compiler/integral/direct_cooperative_gradient_cuda.py:268`
  traverses ket primitive pairs, prepares a shared Coulomb ladder and contracts
  independent centers with CTA barriers. Page width changes acquisition but
  do not remove any of the 1,250,544 preparations or those dependencies. A
  future liveness/barrier optimization needs a new independent proof and
  measurement; unavailable NCU stalls cannot establish its benefit.

## Direct scheduling (#2477 / #2479)

All four schedules use the same experiment source and binary. In the separate
96-atom warm event/census captures, retained work is identical: order seven
executes 811,824 shells and 1,250,544 primitive products; dddd executes 95,976
shells/products. The observed kernel intervals are:

| Schedule | Low + order six (s) | Order seven (s) | Order eight (s) |
| --- | ---: | ---: | ---: |
| Default (page 32, generic dddd acquisition) | 26.283 | 28.079 | 5.781 |
| dddd class-major | 25.934 | 27.799 | 5.458 |
| Page 16 | 26.277 | 27.839 | 5.781 |
| Page 8 | 26.271 | 27.804 | 5.792 |

Class-major dddd acquisition reduces route-two candidates from 380,073,488 to
2,164,240 and pages from 5,959,152 to 68,640, with no angular rejects. Required
contraction work is unchanged; the approximately 0.323-second isolated gain
cannot close a 90-second complete warm gap. Page 16/8 increase order-seven
acquisition pages from 665,600 to 1,331,200 / 2,662,400 without removing primitive
work. Their approximately 0.24–0.28-second observed kernel differences are small.
The census borrows 5,400 diagnostic bytes; clean runs allocate no census storage.
All scheduling candidates reuse existing resident topology/cursors and add no
quartet inventory or force panel allocation.

The clean direct-scheduling cohort below includes all five samples for every
reported warm median. `incomplete` is not a timing; the raw partial samples
remain in the compact archive. The whole-point outcome is separate from each
completed earlier phase.

| Atoms | Arm | Point status | Cold (s) | Warm (s) | Moved (s) | Moved-warm (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 48 | baseline | measured | 83.696 | 22.358 | 62.445 | 22.352 |
| 48 | force-default | measured | 85.372 | 23.034 | 63.973 | 23.054 |
| 48 | force-dddd | measured | 85.355 | 22.909 | 63.797 | 22.863 |
| 48 | force-page16 | measured | 85.473 | 22.831 | 63.764 | 22.833 |
| 48 | force-page8 | measured | 85.491 | 22.687 | 63.720 | 22.718 |
| 96 | baseline | measured | 463.117 | 95.380 | 314.350 | 92.741 |
| 96 | force-default | timeout | 615.526 | 93.001 | 468.809 | incomplete |
| 96 | force-dddd | measured | 502.805 | 92.084 | 271.889 | 92.157 |
| 96 | force-page16 | measured | 426.635 | 91.250 | 310.264 | 91.736 |
| 96 | force-page8 | measured | 436.333 | 93.026 | 316.259 | 92.832 |

The same-image default uses 37/27 cold/moved Fock builds and reaches
1800 seconds after only 2 moved-warm calls. It is not a qualified complete
comparison population. Every candidate's 96-atom warm gain is nevertheless below
2%, already short of the required >=5% warm **and** moved-warm gate. Cold/moved
iteration differences are retained, not interpreted as schedule savings. All
3/12-atom r2SCAN-3c and HF controls and 48/96-atom PBE0 controls pass.
**Keep generic dddd acquisition and page 32 as defaults.**

The diagnostics-off experiment default is approximately 3% slower than frozen
master at 48-atom warm, but faster at 96-atom warm; this is not proof of either
zero census-disabled overhead or a repeatable default speedup. These separate
source/device observations and the changed SCF work remain explicit.

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
tests pass on real GPUs. Compute Sanitizer reports zero errors. The initial
five-file AO implementation is retained in source reconstruction. The subsequent
master merge adds separate precision paths around discovery; post-merge FP64
validation is reported separately rather than relabeling the old image.

## Integration with current master

Master `eae30bcdb` includes #2452's separate PBE0 Fast precision owner. The merge
preserves its fields and retry contract alongside the scheduling/census fields.
The diagnostic marks mixed force execution as incomplete coverage: FP32 attempts
and strict retries require the precision owner's separate accounting. This
campaign's numerical and timing populations remain strict FP64. The merged
host suite passes 758 tests. Source-matched Slurm 8431 GPU qualification passes
all 36 independent Libcint schedule/spin/representation cases, AO discovery,
matrix/point/ragged controls and AO memcheck (zero errors). A further 105 targeted
host checks pass after CI fixes, including standalone CMake code generation
and executable queue/admission coverage replacing fragile source-text assertions.
Two all-coordinate
independent FCI finite-difference cases additionally check the class-major/page-8
route in both Cartesian and spherical representations (Slurm 8436). The first
attempt skipped for missing `threadpoolctl`; that receipt is retained, and the
qualified rerun uses a task-local pinned 3.6.0 installation. The earlier experiment
image also passes force memcheck and synccheck with zero errors.

The merged scientific source at `6e8cf0f23` has a strict 96-atom diagnostic
(Slurm 8434) retaining cold and one warm call at 459.134/91.373 seconds,
independent E/force gates, complete wall reconciliation and actual force census.
This is integration validation, not a replacement for the frozen clean cohorts.
Its exact library/source identity and scheduler record are in the diagnostic
archive. The subsequent CuMetal compatibility fix spells the same capture-refusal
status as its stable integer ABI (900); an NVIDIA-header static assertion and
fault-injected compatibility-header tests verify this host-only error path.
The CLI-only output guards and the standalone codegen option default likewise
leave scientific generation unchanged. The CMake closure test compares the
produced grid source with the canonical emitter.
These later host/CLI checks are separate from the frozen GPU image.

## Four-jet contractions after CSE (#2478)

The matched 8/16/32 builds have the same post-CSE source, AO radial reuse setting,
scientific formulas, active maps and point/force protocol. The only CMake
difference is `GENERATIVEQC_CUDA_XC_MATRIX_TILE`. In the 96-atom direct warm
diagnostic, both M06-2X and r2SCAN execute 4,520 batched launches for each matrix
consumer:

| Matrix tile | Density products (s) | Symmetric Vxc (s) | Sum (s) |
| ---: | ---: | ---: | ---: |
| 8 | 0.956 | 1.020 | 1.976 |
| 16 | 0.924 | 1.342 | 2.267 |
| 32 | 1.100 | 3.705 | 4.805 |

The tile-eight pilot reduces matrix time by approximately 12.8% with unchanged
AO/jet work and launch counts. Its Vxc CTA uses 64 lanes and 2,320 static shared
bytes, still 48 registers/thread and zero reported local bytes/thread. This is
launch metadata, not measured occupancy. The active maps have 9,216 tiles,
768 empty tiles, active counts 0–536, sum 1,460,424 and 373,868,544 point-AO
visits. `point_ao_square_sum` is 75,674,112,000 versus the dense logical
1,391,569,403,904; these semantic counts are shared by all matrix arms.
The meta-GGAs retain all four density/weak-potential jets; the PBE0 control
retains its single matrix jet. No optional prepared GEMM provider is admitted.

The separate force geometry-point time is about 1.246 seconds for M06-2X and
0.675 seconds for r2SCAN. It is not another SCF contraction. The larger tile
increases CTA width from 256 to 1024 lanes and Vxc static shared memory from
8,720 to 33,808 bytes, while retaining 48 registers per thread. The measured
regression rules out promotion; resource metadata alone cannot identify the
unavailable occupancy/stall counters as its cause. The smaller tile-eight
experiment and complete endpoint results are retained separately.

All 36 clean matrix populations (432 returned calls) pass the unchanged
independent energy/force gates. The 96-atom complete warm medians are:

| Method | Provider | Tile 16 warm / moved-warm (s) | Tile 8 warm / moved-warm (s) |
| --- | --- | ---: | ---: |
| m06-2x | Direct | 17.258 / 17.259 | 17.496 / 17.619 |
| m06-2x | DF | 29.426 / 29.650 | 28.278 / 28.296 |
| r2scan | Direct | 16.075 / 16.068 | 15.525 / 15.674 |
| r2scan | DF | 12.834 / 12.705 | 12.369 / 12.727 |
| pbe0 | Direct | 13.412 / 13.453 | 13.378 / 13.405 |
| pbe0 | DF | 22.934 / 19.337 | 22.204 / 19.012 |

Tile 8 improves the isolated matrix pipeline, r2SCAN direct, and M06-2X DF,
but M06-2X direct warm/moved-warm regress by approximately 1.4%/2.1%; r2SCAN DF
moved-warm is approximately flat. Its complete cold/moved M06-2X direct times
are 120.968/92.693 s versus 123.968/94.100 s at tile 16. Actual SCF work and all
five-sample ranges, including the 48-atom controls and every tile-32 regression,
are in [endpoints.md](endpoints.md). There is no consistent universal endpoint
win, and the source includes a separate unmerged CSE prerequisite. **Tile 16
remains the default; tile 8 remains an explicit build option.**

These are finite cohorts on separately allocated devices. Five warm repeats
describe within-population variation; they do not estimate independent
cold-run or between-device uncertainty. No sample is replaced, filtered or
rescaled to a preferred SCF iteration count.

## Retained evidence

- [summary.json.xz](summary.json.xz): all 110 expected populations, all returned
  full-precision timings/errors/work, independent geometry oracles, missing
  phases, raw hashes and six source/image reconstructions. Maximum native error
  across returned clean calls is 2.1964297e-10 Eh / 1.6855894e-10 Eh/Bohr.
- [endpoints.md](endpoints.md): every accepted phase, five-sample range, observed
  iteration/build pairs and whole-point status.
- [diagnostics.json.xz](diagnostics.json.xz): separately gated two-call profiles,
  exact kernel names/resources, exclusive ledgers, actual work censuses, AO
  transfer counters, cache/compiler provenance, test/sanitizer receipts, refused
  profiler attempts and additional dirty-source reconstructions. Collection
  scripts are retained under `receipts`; every original input has a hash.
- [manifest.json](manifest.json): checksum/size inventory and explicit default
  decisions. No external Release or archive is required to read these records.

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
