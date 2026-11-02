# Raw metric scheduling and README seven-method comparison

Metric construction now selects the existing raw-value schedule, which has no
transformed auxiliary reduction. Automatic metric outputs use one thread each;
explicit auxiliary/component/primitive controls remain available. The
transformed consumer, equations, FP64 policy and resource capacities are unchanged.
There is no GPU-name selector in this change.

Measured native source:
`cbd56578885c044e5ccbc636f4d0151e37a039834102a8cbb33da546d1652e25`.
Native library SHA256:
`48c02b5b9a87d61bc21d25d9d85cb1155c367d241e13cea9c96db6147de5b22b`.
The predecessor is source
`6527e5625b096f10a8c431ce2829b67d44a98abb656889f486f2b13786dc419c`, library
`327a417cb66c115b7f5d35ca9fde6a89ec217da632e2d0cb0c589bead84b614f`.
Frozen dirty-source receipts, original base commits and reconstruction patches
are retained; later publication commits do not relabel measured images.
The receipts retain an inherited `source_revision` value of
`542f514ce099a4475eabe6b25fb46c4cd112d65a`; the primary benchmark driver is
byte-identical at that revision. That field does not identify the modified
native tree: use the source inventory, archive/base/patch and library hashes
above, which the reader checks for every native population.

## Complete endpoint protocol

All seven methods use the README water-96 task, spherical orbital bases,
method-specific retained auxiliary bases, strict FP64 and stock semilocal/VV10
grids. Native SCF/force tiles are 256/1024. Every force call includes moving-grid
response and, for r2SCAN-3c, the complete D4/gCP corrections. Eight Slurm CPU
slots are allocated and math libraries use one thread. Engines retain their
own SCF convergence rules and are independently checked against common gates.

For each method and device, parent, candidate and GPU4PySCF each have one
conditioning process (cold plus one warm). Measurements then run
parent/candidate/GPU4/GPU4/candidate/parent, each a fresh owner with cold plus
three frozen engine-local warm calls. Cold includes calculator construction,
preparation, SCF, forces and host output. Compilation caches are conditioned;
this is not a first-install compilation measurement. All two cold and six warm
samples per measured arm are retained, with ranges, MAD, first-warm and complete
cold-plus-three-warm times. No fastest-sample selection is used.

All 126 conditioning/measured populations of the qualified comparison pass the independent
README reference gates. One additional complete parent population also passes
scientific gates but is excluded from the tables for the allocation issue below.
The source-matched reference-reuse manifests retain their original
producer identities and exact package maps. Paired native energy/force and SCF
iteration/Fock-work gates apply to every sample. This is a default-grid
comparison for the sampled static workloads, supplemented by independent small
moved-geometry tests; it does not establish a global fastest configuration.
Positive changes below mean slower candidate latency.

## RTX 5090

| Method | Native cold (s) | GPU4 cold (s) | Cold change | Native warm (s) | GPU4 warm (s) | Warm change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbe | 10.788044 | 12.098472 | -10.83% | 3.681179 | 4.012001 | -8.25% |
| pbe0 | 20.927796 | 22.930387 | -8.73% | 5.898681 | 10.160929 | -41.95% |
| b3lyp | 21.098876 | 24.156703 | -12.66% | 5.964517 | 10.159241 | -41.29% |
| r2scan | 12.670371 | 15.260540 | -16.97% | 4.173039 | 5.940855 | -29.76% |
| r2scan-3c | 22.523749 | 23.294689 | -3.31% | 6.839443 | 9.890556 | -30.85% |
| m06-2x | 24.198434 | 27.776393 | -12.88% | 6.274419 | 12.072267 | -48.03% |
| wb97m-v | 235.536117 | 377.606862 | -37.62% | 29.977908 | 126.083098 | -76.22% |

## RTX PRO 6000 Blackwell

| Method | Native cold (s) | GPU4 cold (s) | Cold change | Native warm (s) | GPU4 warm (s) | Warm change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbe | 10.099993 | 12.564430 | -19.61% | 3.426346 | 3.763323 | -8.95% |
| pbe0 | 19.520249 | 22.714339 | -14.06% | 5.521251 | 9.459772 | -41.63% |
| b3lyp | 19.661339 | 23.869870 | -17.63% | 5.587996 | 9.460141 | -40.93% |
| r2scan | 11.813780 | 15.388221 | -23.23% | 3.897841 | 5.534697 | -29.57% |
| r2scan-3c | 21.136324 | 22.774373 | -7.19% | 6.419302 | 9.154639 | -29.88% |
| m06-2x | 22.527180 | 27.124161 | -16.95% | 5.871178 | 11.226156 | -47.70% |
| wb97m-v | 208.101391 | 350.235023 | -40.58% | 14.328729 | 116.610931 | -87.71% |

## Change versus the frozen native predecessor

| GPU | Method | Cold change | All warm change | Cold + three warm change |
| --- | --- | ---: | ---: | ---: |
| 5090 | pbe | -3.063% | -0.024% | -1.559% |
| 5090 | pbe0 | -1.207% | -0.054% | -0.678% |
| 5090 | b3lyp | -1.009% | -0.057% | -0.560% |
| 5090 | r2scan | -1.137% | +0.062% | -0.546% |
| 5090 | r2scan-3c | -3.585% | +0.047% | -1.884% |
| 5090 | m06-2x | -0.737% | -0.047% | -0.436% |
| 5090 | wb97m-v | +0.276% | +0.040% | +0.203% |
| pro6000 | pbe | -1.773% | -0.108% | -0.921% |
| pro6000 | pbe0 | -0.918% | -0.027% | -0.506% |
| pro6000 | b3lyp | -0.935% | +0.003% | -0.512% |
| pro6000 | r2scan | -1.498% | -0.088% | -0.801% |
| pro6000 | r2scan-3c | -3.230% | +0.050% | -1.706% |
| pro6000 | m06-2x | -0.765% | +0.019% | -0.426% |
| pro6000 | wb97m-v | -0.075% | -0.004% | -0.061% |

Every slower sample and positive median change remains visible. Small warm
changes should be interpreted alongside the complete sample ranges in the
replayed report, rather than as resolved gains or regressions.

## Allocation interruption and retained excluded owner

The original n1 job 15148 reached its 90-minute limit at 20:42:46 CST. Its last
parent WB97M-V receipt was written at 20:44:43.914906, after the allocation ended.
The receipt has no per-phase wall timestamps, so no phase can be certified inside
the allocation. All its complete energies, forces and timings are retained and
independently qualified, and its timings are reported separately as resource
unqualified. None enters the latency medians above.

The first conditional recovery job 15171 exited before numerical work because
this cluster disables accounting storage. Job 15182 then recorded controller
TIMEOUT evidence but found all nine receipt files complete and did no numerical
work. The subsequent resource audit identified the late final owner. Only this
owner is measured again in job 15183, on the same physical GPU with the same
8-CPU/math-one limits and immutable parent source/library. The reader binds its
actual allocation and requires the new receipt to predate its successful end.
The recovery uses a 20-minute Slurm limit and a 15-minute foreground timeout
with a 30-second kill grace. This keeps the benchmark in the foreground
process group and bounds shutdown before the Slurm deadline. Earlier attempts and the
excluded population remain authenticated in the bundle; the recovery is due to
resource qualification, not its measured latency.

## Numerical, lifetime and work qualification

The raw/metric/J mapping suite has sixteen passing cases, including automatic
and all explicit mappings, Cartesian/spherical representations, auxiliary g,
rank-deficient metrics, signed density, ragged panels and differing batch
layouts. Four mixed-representation derivative cases and forty-eight independent
libcint/NumPy raw/metric/J/K cases pass, including long contractions. Six public
HF/PBE0 cold/warm/moved tests protect independent forces and constrained-memory
response fallbacks. Six direct native Compute Sanitizer processes pass with
zero errors and zero bytes leaked, with all output tensors independently gated.
Host schedule/policy validation has 81 passing tests.

The 108-atom lifecycle retains six complete calls per source: cold, retained,
forced eviction, second replay, moved and moved-warm. All paired numerical and
physical-work gates pass. This is larger paired lifecycle evidence, not an
independent large-geometry production ranking. Cold improves 30.748 to 29.827 s;
moved geometry improves 22.430 to 21.503 s. All six timings remain in the report.

A separate complete Nsight endpoint reduces metric-kernel time from 0.875058 to
0.100226 s. The measured launch shrinks from 11,614,464 to 362,952 blocks of 128
threads. Both publish 46,457,856 elements for 6,816 auxiliaries. The symmetric
23,232,336-pair domain is derived from these dimensions, not an instrumented
arithmetic counter. All 69 recorded downstream work/capacity receipts are equal.
Profiled timings are excluded from clean comparisons.

The preceding global-mapping pilot retains all twelve independently qualified
populations. It changes both raw and transformed controls and is diagnostic;
only the focused metric fix supplies the production comparison above. Original
raw tensors, full traces and build logs stay in ignored local capsules with
hashes and sizes in the retained inventory. No raw sample or failed attempt is
dropped to fit the unchanged 80-MiB tracked-evidence ceiling.

## Offline replay and remaining scope

```bash
PYTHONPATH=python:. python benchmarks/results/metric-value-schedule-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/metric-value-schedule-20261101/report.py
```

The reader restores exact FP64 words and original text hashes before checking
source/binary/allocation identities, every independent numerical gate, the full
seven-method/two-device census, semantic work and complete warm populations.
Normal and optimized Python must produce identical reports. Build provenance
records the verified sccache launcher and SM120 real-image library used here.

The isolated aTrust VPN portal authenticates, but SSH to the A100 entry is reset
before key exchange; direct connections from the host time out. There is no A100 execution measurement for
this source; prior jobs 224223/224225/224229 remain unretrieved and were not
resubmitted. Older A100 values cannot be assigned to this source. Two Blackwell
models do not establish a universal GPU performance guarantee or completion of
the full A100/5090 objective.
