# Experiment: compiler-planned Becke primal lifetime across reverse phases

Status: proposed; opt-in shared-owner integration, **not default enablement**
Date: 2026-10-04
Base: master as observed at experiment start,
`d22c92d8fca233e3eb742033e923ed8dac004c42`

## Motivation and scope

PBE0 still has large exact-integral derivative and grid geometry costs. A retained
local-AO profile on a different snapshot reported 26.5024 s in full-range
derivative kernels and 15.4836 s in cooperative geometry kernels. These are
instrumented kernel sums, not current-source exclusive endpoint wall phases.
The complete-endpoint benefit of the experiment below is **unknown**. It does not
address the larger integral derivative producer, SCF J/K, or AO-density products.

The GPU4PySCF v1.8.1 comparison points to a different parallel decomposition:
`gpu4pyscf/lib/gdft/gen_grids.cu` assigns a thread to (point, derivative atom),
whereas our large-domain helper repeatedly streams triangular strips through a
point block. Both are O(G A^2), not algorithms with different asymptotic order.
Our actual stationary resource planner expands point workers under budget; the
32-lane *minimum* in initial admission is not the actual 96-atom launch size.
Do not repeat the mistaken diagnosis that the ordinary path launches only 32
point blocks.

Existing work was considered rather than duplicated: #1779 local-AO force
consumption, #1774/#1780 selected-AO SCF, #1767 indexed exact-force pages,
#1798 approximate AO-product gating, and the #1686/#1819 DF handoff. The paused
incremental-J/K experiment is not resumed. Earlier scalar DPPP slabs and exact
double-zero Becke pruning were negative directions, not prerequisites here.

## Compiler change

`python/generativeqc_compiler/xc/grid_phased.py` explicitly plans seven phases:
distance, pair primal, atom logs, point normalization, pair reverse, atom gather,
and point motion. This is a compiler-owned schedule/lifetime experiment, **not a
general automatic IR optimizer**. Existing grid_response Graphs still own the
scalar norm, ratio, logarithm, and Becke partials. The normalized-product adjoint
in grid_native is generalized to pointer-like views without changing its algebra.

The pair layout is `[word][canonical_pair][point]`. Four primal doubles survive
until normalization finishes, then reverse overwrites them with four pullbacks.
Same-stream phase completion makes that alias safe. Atom-owned incident gathers
retain the original neighbor order, without floating-point atomic accumulation.
Maximum-log scaling, clipping, exact-zero rules, and failure publication are not
replaced with GPU4PySCF's cutoffs or normalization policy.

The initial standalone prototype had no production caller. The subsequent
shared-owner integration below is qualification-only and default-off. Normal
production routing and its bounded fallback remain unchanged. The standalone
benchmark owns allocation, launches, sticky errors, and transactional host
publication.

## Work and memory model (not FLOPs)

Let A be atoms, G total points, B points per tile, and P=A(A-1)/2.

- Large-domain strip schedule: 2GP pair primal evaluations; phased: GP. Host
  counted executions verify this reduction. At A<=32 the current retained
  cooperative route already produces GP, so there is no such saving.
- Both still perform O(GA^2) pair consumption/log-product work, including two
  incident readers per unordered pair in each atom-gather pass. This is a
  constant-factor work and parallelism change, not a lower scaling exponent.
- Optional primal cache: 32BP bytes. At A=96, B=256 this is 35.625 MiB.
- Standalone phase scratch: `32BP + 8B(12A+1)` bytes, including 11 double atom
  fields, size_t zero counts, and one normalization maximum per point. At
  A=96/B=256 it is 39,716,864 bytes (37.876953125 MiB); B=1024 needs
  158,867,456 bytes (151.5078125 MiB).
- Inputs, output, center cache (48P bytes), descriptor map (8P bytes), other
  owners, and allocator/runtime overhead are **not** part of that scratch
  number. The planner's occupied_bytes argument must charge their concurrent
  reservations before admission. A returned None requires the old bounded path.
- Logical pair-panel accesses alone are about 176 bytes per (point,pair):
  primal write 32, incident factor reads 16, reverse read/write 64, and incident
  pullback reads 64. This is a layout-level access count, not measured DRAM
  traffic; cache reuse and generated load instructions differ.

These admission formulas are neither observed peak memory nor a proof that the
complete 512-MiB owner budget fits. No missing endpoint counters are filled with
zero, and no task/pair count is represented as a FLOP count.

## Same-GPU isolated measurements

Node n5, Slurm main/gpu:5090:1 job 1436, 10-minute limit; RTX 5090 UUID
`GPU-c17495e9-988d-0626-8bdd-95d0d1b47f36`, driver 570.86.16, CUDA 12.9.1.
No cross-node or cross-device calibration is used. Compilation uses verified
ccache 4.5.1, `-std=c++20 -O3 --expt-relaxed-constexpr --fmad=false -arch=sm_120`.

Each row selects 32 evenly spaced *whole tiles* from the unchanged README water
grid, and uses signed normal seeds (seed 20261004), **not SCF XC seeds**. Both
schedules consume identical sampled inputs within that row. After warming each
schedule, CUDA events cover ABBAABBA; the table reports the four-sample median.
Intervals include all sampled Becke tiles and per-point publication. They exclude
allocation, input copies, center preparation, output readback, AO/XC work, and
all other endpoint phases. Rows with different B have different sample sizes and
selected points; their raw times cannot be compared as equal-work tile trials.

| A | B | sampled G | old ms | phased ms | old/phased |
|---:|---:|---:|---:|---:|---:|
| 24 | 256 | 8,192 | 2.689888 | 3.013488 | 0.893 |
| 24 | 1024 | 32,768 | 5.661584 | 4.314960 | 1.312 |
| 48 | 256 | 8,192 | 12.817920 | 5.588768 | 2.294 |
| 48 | 1024 | 32,768 | 19.934560 | 10.978128 | 1.816 |
| 96 | 256 | 8,192 | 45.661615 | 14.407856 | 3.169 |
| 96 | 1024 | 32,768 | 97.540623 | 35.023888 | 2.785 |

All six rows have zero maximum absolute per-point difference between the two GPU
schedules. The 24-atom/256-point case **regresses 12.03%** and must remain a
negative result, not be hidden by the larger cases. Do not multiply the speedup
by an older geometry wall phase and claim saved endpoint seconds.

Raw samples, sample hashes and work models:
`benchmarks/results/pbe0-becke-phases-20261004/isolated-abba.json`.
Emitted source SHA256:
`730764b935f56680e499e677a5ce0d0f73e77c53db68a246cfad58528ca8de1c`.
Probe library SHA256:
`182f26dcbc80a630ee34e05f7da906d0616d7e120745b30a047ab1efce246fd5`.

## Validation and retained failures

- 368 host tests pass: new phased tests, existing retained/tiled cooperative
  tests, and stationary geometry resource tests. Includes iterations 1/3/5,
  A through 128, counted pair production, changed geometry, tails, exact-zero
  and nonfinite semantics, Decimal finite differences, translation/permutation,
  cache admission boundaries, and guarded scratch panels.
- Five additional stationary geometry host-emulation tests pass. GPU tests skip
  outside their explicit Slurm/probe configuration; those skips are not passes.
- CUDA job 1437: 21 pass, 28 explicitly skip because the probe is compiled only
  for iteration 3. Tests cover 257 points at B=17/256, A=1..128, moved-and-restored
  centers, independent host and Decimal gates, and no publication after failure.
- CUDA job 1438: memcheck, racecheck, initcheck, and synccheck all clean for the
  96-atom moved/tail case and independent Decimal case. These are isolated probe
  sanitizer results, not a production owner or full endpoint qualification.
- Compiler/SCF structure checks, codegen test ownership, Ruff, and diff checks
  pass. Existing helper mathematics remains independently exercised.
- The first GPU build omitted `--expt-relaxed-constexpr`, emitted host-constexpr
  warnings, and failed center preparation. Its job ID was not captured and Slurm
  accounting is disabled; it yielded no timing data.
  The qualified build uses the requisite flag; do not treat that failed run as
  a numerical pass or zero duration. An unnecessary atomic read-modify-write
  status read was also replaced with a relaxed atomic load before qualification.
- The first broader host run failed because an experiment-only ccache wrapper
  named nonexistent `/usr/bin/ccache`. It was corrected to the verified
  `/home/jzzeng/.local/bin/ccache`; the successful rerun above is retained separately.

Ignored logs, emitted sources, binaries and ccache statistics are retained in
`.artifacts/` in `/data/jzzeng/qc-pbe0-grid-phases-20261004`; remote logs are copied
under `.artifacts/remote/`. The original dirty checkout was not edited.

## Shared-owner integration checkpoint

Draft PR #1830 now tracks an opt-in integration rebased on master `ebc0c9ea9`,
which includes #1767 and #1686. The public default remains the bounded schedule.
In particular, merged #1767 remains opt-in through
`GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE`; these default-control runs do not enable
it. Its historical improvement must not be subtracted from this baseline.
The initial isolated measurements above are still historical measurements of
the earlier base, not timings of this integrated revision.

`method/stationary_becke_phased.py` emits the iteration domains and source-panel
consumer. The existing cooperative AO pass supplies its actual XC/optional
external seed; seven same-stream phases then write the existing weight-response
panel before the normal all-source reduction. The native owner allocates and
launches only. It checks target capabilities, one point per lane, retained center
geometry, and the complete owner's spare reservation. Optional allocation
failure retains the bounded route. The separately named v1 configuration and
metrics interfaces leave old AOT artifacts on their old route. No work counters
are synthesized for a missing interface.

The integration charges `scratch_bytes + 8B + 8P`: seeds and the canonical
descriptor map are additional to the phase scratch. At A=96/B=256 this is
39,755,392 bytes. Existing lane panels remain allocated; no unproved storage
borrowing is credited. Resource and actual-allocation accounting distinguish
retained storage from bounded fallback. Per-execution phase counts are deltas;
reserved bytes remain capacities, never execution counts.

Integrated correctness evidence (not timing calibration): n2 Slurm job 2202,
one allocated RTX PRO 6000 Blackwell, 12 tests passed. The actual shared owner
compares bounded and phased source panels at A=48/96, 257 points with a one-point
tail, explicit/implicit point ownership, full/subset/empty AO maps, and moved
then restored centers. Pair-production counts and owned allocations are checked.
AO/features in this gate are synthetic, so it does not replace a molecular
independent oracle or the complete E/F gate. These PRO 6000 results are not mixed
with the earlier RTX 5090 performance data.

The combined integration host suite reports 446 passes, including the original
373 gates, resource admission, lowering, source accounting and metric deltas.
Earlier 210-pass and 44-pass runs overlap and must not be added to that total.
The existing host resource stubs were extended for the optional owned buffer.

n2 Slurm job 2205 additionally passed memcheck, racecheck, initcheck and synccheck
for the actual 96-atom shared owner with full AO, explicit owners and a tail.
Each tool ran one test (11 deselected); all reported zero errors. These remain
synthetic shared-owner gates, not complete molecular-oracle qualification.
The shared-wrapper source and library SHA256 values were respectively
`ba8e2844c7e56e89a8abe70218d34722cfe1e56e2a25a11c050c929752d5f62a`
and `f73a25c085bb570f5085fc5976523a061f22cd5fdc64ce7c9b3f85e4878cf5be`.

The expanded shared-owner gate in n2 job 2208 passes 24 tests, adding resident
external seeds with nonzero offsets, failure-before-host-publication, and reset
recovery for both schedules. Job 2209 passes all four sanitizers on the 96-atom
full-AO explicit-owner external-seed/recovery case. Job 2206 first failed in the
test's NumPy-to-CuPy assignment (before the recovery gate); explicit stream-bound
array upload fixes the test harness, not the scientific kernels.

An additional 44-pass host boundary suite checks retained allocation, admitted
but unallocated fallback, known old ABI, inconsistent allocation, and missing
metrics on a supported interface. These mock allocation outcomes and are not
forced device-allocation failures. The 26-pass benchmark schema suite preserves
actual phase batches, pair-production counters and capacity bytes without
inventing counters for old artifacts. Both suites overlap prior evidence where
applicable and are not added to a cumulative unique-test total.

The attempted n4 run did not reach scientific execution: NVML reports a
driver/library version mismatch (job 622), and CUDA device access reports
`cudaErrorCompatNotSupportedOnDevice` (job 623). No driver or visibility override
was used. n2's first attempt lacked transferred artifacts because direct n5-to-n2
SSH host-key verification failed; transfer through the already trusted local
connections resolved that setup failure before job 2202.

The ordinary README protocol can be run with the qualification-only schedule
switch via `python -m benchmarks.readme_pbe0_phased native ...`. Verify actual
`phased_becke_batches` before attributing timing to this schedule: admission may
still take a resource/capability fallback. Normal `readme_pbe0` remains the
unchanged schedule control.

## Required next gate; not yet completed

The first full-endpoint attempt on n1, job 5726, failed in the control's cold
force setup: the configured wrapper was named `nvcc-n1`, while the compiler
adapter resolved its sibling `nvcc`. No full native E/F result was produced;
the elapsed failed endpoint is not a timing sample. The rerun uses a correctly
named ccache wrapper and retains the failed JSON/logs. The initially attempted
source/library transfer also exhausted n1's root filesystem; only this
experiment's directories were moved to its available `/data` filesystem.
No jobs are submitted on n3 at the user's request.

Job 5727 then stopped before native force execution because the wrapper directory
lacked the sibling `ptxas` required by compiler identity collection. The complete
wrapper bundle now passes host-only identity, cached compilation, linkage and
symbol-execution preflight. Both failed campaigns remain retained. The clean
sequence restarts at job 5728, with reference/control/candidate in one allocation;
do not combine reference timings across these jobs.

The 48-atom control in job 5728 completes all 12 energy/force gates. Its cold
endpoint is 356.278 s, including 227.206 s in the reported preparation/compiler/
owner-setup group. The five warm endpoints are 17.713, 17.680, 17.674, 17.664 and
17.666 s. These are control-only observations, not phased-schedule speedups.
The roughly 6.07 s geometry wall group is not a Becke-only kernel measurement.

That run exposed a reporting gap: the benchmark normalizer retained only the
old AO-task derivative timer, dropping the existing native prepared and shell
derivative wall phases. The normalizer now sums those exclusive phases and
retains the original exclusive timeline; 55 adjacent host tests pass. Frozen
job-5728 benchmark code is not changed mid-run. Its missing normalized derivative
times remain missing, rather than being reconstructed from unattributed time.

The cold group motivates auditing compiler root demand, separately from the
warm Becke experiment. The ordinary enlarged-domain caller requires a complete
native integral producer and forbids the AO-task fallback, yet prepares the full
primitive derivative provider. The composite owner already requests only the
nuclear primitive with `integral_derivatives=False`. The full SPD compiler
inventory has 313 ERI, 16 overlap, 16 kinetic, 16 nuclear-attraction and one
nuclear request. They are emitted source requests, not executed integrals or
FLOPs. Removing unreachable compilation is a candidate, not an implemented
optimization or evidence that all cold preparation time is avoidable.

The larger cases justify trying integration, **not automatic enablement or a
ready-to-merge performance claim**. A draft PR can track the integration and
missing gates. There are no new complete E/F cold, warm, moved-geometry,
or GPU4PySCF endpoint timings for this schedule. Those costs remain unknown.

Continue complete-molecular qualification of the shared integration. Actual
device allocation failure remains an unforced optional fallback, not measured
peak-memory evidence. Actual resource capability and small-domain
negative evidence must continue to select the bounded fallback. Actual pair
production remains separate from existing pair-visit budget counters.

Then compare frozen same-source/same-GPU complete E/F runs against the unchanged
schedule and independent GPU4PySCF, including cold startup and moving geometry.
Do not infer a PBE0-vs-GPU4PySCF gap reduction from this isolated result. If the
complete endpoint fails to improve materially, reject the extra storage/launch
complexity rather than polishing this probe. Do not pursue blanket small-system
enablement, a universal 1024-point tile, or claims of sub-cubic scaling here.

## Complete-endpoint evidence: job 5728

This later evidence supersedes the pending-endpoint statements above; it does
not make the entire campaign complete. One finite 75-minute n1 RTX 5090 Slurm
allocation ran reference, control and candidate sequentially at 48 and 96
atoms. The control is `ebc0c9ea9`; the candidate and frozen benchmark are
`5a3e9b236`. The later normalizer repair was not inserted mid-run. Source and
binary identities, rather than absent remote Git metadata, bind the records:

| Identity | Control | Candidate |
| --- | --- | --- |
| Source | `679172dbdfc3599de3a3afbf9f31894b936f6c5f7e8257cc9f25c768057c66ff` | `8ee029b7c6ab2a619b9bc4be00c20b72ed19b992e2e70947cfecbff70c686f11` |
| Library | `56c985bf50d5c9053c676f497f5f1efcde4c1d7abd930cd675e1167f2a1f4320` | `a94ce0b054ce5b2b2e2a22bde6d0b501a9763b7df0a9a097fcb6fdae4b626ce5` |

Full-grid spherical def2-SVP PBE0 complete energy/analytic-force seconds:

| Atoms / phase | Control | Candidate | GPU4PySCF 1.8.1 |
| --- | ---: | ---: | ---: |
| 48 cold, including preparation | 356.278 | 222.126 | 50.230 |
| 48 warm, median of five | 17.674 | 16.724 | 6.066 |
| 48 moved, including preparation | 83.744 | 69.494 | 46.943 |
| 48 moved-warm, median of five | 17.735 | 16.760 | 4.004 |
| 96 cold, including preparation | 527.618 | 519.366 | 77.565 |
| 96 warm, median of five | 75.904 | 67.248 | 15.196 |
| 96 moved, including preparation | 253.203 | 245.098 | 86.388 |
| 96 moved-warm, median of five | 76.051 | incomplete | 9.972 |

All native warm calls use one SCF iteration. Reference 96 warm uses 3/5/4/5/3;
48 moved uses 15/12/43 iterations for control/candidate/reference, versus
12/12/44 at 96. The measured warm reductions are 5.38% and 11.40%, with remaining
candidate/reference ratios 2.76 and 4.43. These are ordered, not interleaved
causal estimates. Cold differences also include cache/order effects and must
not be attributed solely to Becke. Reference XC reports CUDA LibXC on_gpu=true.
The #1767 paging and #1798 approximate product gate were not enabled, and force
local-AO integration was not present. This is not the requested combined-policy
baseline or a latest-master performance result.

The scheduler killed the last candidate 96 moved-warm at the finite time limit.
Its raw JSON remains `running`, with 11 completed rows and only four completed
moved-warm repeats. No five-repeat median is reported for that phase. All 12
candidate 48 and 11 completed candidate 96 energy/force gates pass unchanged
1e-8 Eh / 1e-7 Eh/Bohr thresholds. Maximum candidate 96 errors are 1.042e-10 Eh
and 3.121e-11 Eh/Bohr. The timeout, failed extension request and incomplete
record are retained; no samples are discarded or synthesized.

Actual candidate phase batches are 4608/9216, pair primal productions are
1,330,642,944 / 10,758,389,760, and extra retained bytes are
10,433,344 / 39,755,392 for 48/96 atoms. The old logical pair-visit budget is a
different counter, including nuclear pair work; none of these counts are FLOPs.
The native derivative duration omitted by the frozen normalizer remains
unobserved in this campaign, not zero or reconstructed from residual wall time.

Raw arrays/logs and scheduler failure are retained locally in
`.artifacts/endpoint-n1-5728/`, not a published replay bundle. Candidate 96 JSON
SHA256 is `9984323d4141a92a6afacd55fc9b228835d0b4ecd5f6237f27a3b877cd2fa4f3`.
A separate candidate-only job 5729 uses the retained reference as a numerical
oracle. Its attempt to substitute the corrected normalizer did not take effect:
the benchmark imports that function locally from `dft_force_components`, not
from the module attribute that the wrapper replaced. Completed rows still have
missing native derivative durations. This observer failure is not a numerical
failure; the run may qualify complete E/F but cannot pass its timing-observer
gate, fill job 5728's missing sample or supply a fresh paired comparison. Use
the committed normalizer directly for subsequent integrated-source work rather
than repeat this wrapper. Do not infer a missing duration from wall residuals.

## CI integration repairs

The optional cooperative phase-seed parameter changes its host-stub ABI from
18 to 19 arguments; the noncooperative ABI is unchanged. Test those calls
separately and compare declarations after removing default initializers. The
probe now uses the existing raw-output retention policy. Rebind the eight
reviewed DFT-MP source spans affected by optional phase storage and ownership,
without weakening any budget or numerical gate. A mutation test rejects a
change to the planner's default-off policy. The focused host suite passes 156
tests on n2; this is not a claim that all exact-head CI jobs have passed.

## Separate complete96 qualification: job 5729

All 12 candidate calls finish and pass the unchanged numerical gates, including
cold, five warm, moved and five moved-warm, with 9216 actual phased batches on
every call. Maximum errors are 1.056e-10 Eh and 3.159e-11 Eh/Bohr. The source
and library match candidate 5728; this is a distinct n1 RTX 5090 allocation,
with the old reference used only as a numerical oracle. Complete seconds are
526.035 cold, 68.422 median warm, 249.353 moved, and 68.684 median moved-warm.
Native iterations are 26/1/12/1. These observations are not a fresh paired
speedup and do not replace the missing last row in job 5728.

The job exits 1 after producing its `measured` 12-row JSON: the final observer
assertion rejects missing derivative duration (`None > 0`). Independent replay
of all 12 rows confirms numerical/phase-count qualification, not an observer
pass or a successful overall diagnostic job. No replacement run is launched
solely for this missing timer. Full rows, identity/ccache receipts and failure
log remain in `.artifacts/diagnostic-n1-5729/`; candidate JSON SHA256 is
`578e56382e667d1be84117e1fa47f56e67254a6d5ead842ab91f74f6efbf7fd6`.

## Reproduction

Host-only emission and cached compilation (CUDA compiler on PATH):

```sh
mkdir -p .artifacts
PYTHONPATH=python:. python -m benchmarks.becke_phased_probe --emit .artifacts/probe.cu
ccache --version
ccache nvcc -std=c++20 -O3 --expt-relaxed-constexpr --fmad=false -arch=sm_120 \
  -Xcompiler=-fPIC --shared .artifacts/probe.cu -o .artifacts/probe.so
```

Run the probe only inside a finite Slurm `main --gres=gpu:5090:1` allocation,
preserving its CUDA_VISIBLE_DEVICES:

```sh
PYTHONPATH=python:. python -m benchmarks.becke_phased_probe \
  --library .artifacts/probe.so --output .artifacts/isolated-abba.json
GENERATIVEQC_BECKE_PHASED_PROBE=$PWD/.artifacts/probe.so \
  PYTHONPATH=python:tests/python python -m pytest tests/python/test_becke_phased_cuda.py -q
```
