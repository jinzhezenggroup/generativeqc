# Default cold Direct r²SCAN-3c96 and six-endpoint regression

The final cold candidate uses the unchanged **#2786 default-grid framework**:
complete construction, preparation, SCF, analytic forces and host export,
followed by three calls from each engine's own frozen cold density. Three
fresh-process native/GPU4PySCF rounds alternate first-arm order on one
Slurm-allocated RTX 5090 per scientific pair. No research selector is enabled.
The campaign retains its existing spatial order `2` and validation GEMM `1`.

Direct r²SCAN-3c96 cold median is **31.3617 s versus 33.1508 s**; ranges are
[31.3293, 31.5113] and [32.5021, 33.2414] s. Warm medians are **13.8667 s versus
14.8755 s**. Native cold performs ten FP64 Fock builds,
including three full anchors, six deltas and one post-SCF closure; all work
remains counted. Frozen warm performs one full build. Preliminary Hückel work
is reported separately in the original receipts. GPU4PySCF records iterations;
its complete operator count is not inferred from them.

All six targeted cold medians remain below GPU4PySCF in the final frozen
implementation. The 36 populations / 144 calls include every measured round.

| Endpoint | Native cold median [range], s | GPU4PySCF cold median [range], s | Native / GPU4 warm median, s |
| --- | ---: | ---: | ---: |
| direct wb97m-v 48 | 82.7153 [82.6185, 83.0483] | 98.2344 [96.8318, 98.2702] | 6.6039 / 32.4958 |
| df wb97m-v 48 | 68.0524 [67.9785, 68.1449] | 90.9800 [90.9766, 91.1969] | 5.7061 / 32.3247 |
| df hf 96 | 19.0266 [19.0119, 19.0332] | 19.9482 [19.9408, 20.3837] | 4.1206 / 10.0700 |
| df pbe0 96 | 24.5324 [24.3508, 24.5472] | 26.3632 [26.2054, 28.1195] | 8.1765 / 13.4279 |
| direct hf 96 | 11.3160 [11.2924, 11.4322] | 11.6199 [11.5716, 11.6615] | 2.6641 / 3.3181 |
| direct r2scan-3c 96 | 31.3617 [31.3293, 31.5113] | 33.1508 [32.5021, 33.2414] | 13.8667 / 14.8755 |


## Scientific qualification

All main r²SCAN and Direct HF populations pass the unchanged independent
common gates: energy ≤1e-5 Eh/atom, maximum force ≤1e-4 Eh/Bohr and RMS force
≤3e-5 Eh/Bohr. Original oracle bytes and producer identities are reused from
[`readme-refresh-20261025`](../readme-refresh-20261025/README.md), with exact
scientific inputs checked by the shared qualifier. The inherited DF HF96
fitting-energy failure remains visible for both engines. The separate stronger
1e-7 default-HF force diagnostic still fails as documented in
[`cold-hf-defaults-20261026`](../cold-hf-defaults-20261026/README.md); passing the
README gates does not supersede that result.

Eight additional Direct r²SCAN-3c calls cover 96 and 120 atoms, cold, warm,
moving an existing owner and constructing a fresh owner at moved geometry.
Independent GPU4PySCF default-grid references retain their original samples;
exact geometry is checked before reuse. Every energy and force gate passes;
maximum force discrepancy is 2.97e-6 Eh/Bohr. These moved comparisons qualify
implementation agreement at the sampled grids, not grid convergence for every
possible displaced geometry. Main static populations use the denser independent
README oracle qualification separately.

The final GPU suite passes native precision policy, 25 one-electron cases,
18 force scheduling cases, nine incremental convergence/lifetime cases,
44 primitive MD-J cases and two larger automatic-policy cases. Public Schwarz
s/p/d/f, Cartesian/spherical, batch, capacity and NaN checks pass under memcheck.
The four signed proposal/restore/graph cases also pass memcheck; both sanitizer
runs report zero errors. The initial suite stopped because the isolated test
environment lacked `typing_extensions`; version 4.16.0 was installed and the
unchanged full suite was rerun successfully. Both logs are retained.

Host validation includes 1,362 MD allowance/controller checks and the compiler,
policy and one-electron followups. A proposal uses an invocation-local MD view;
its cap cannot become a prepared-owner or captured-graph property. The original
cumulative omission proof remains directed-rounded and covers signed density,
zero cutoffs, and tighter requested targets.

## Mechanism and rejected experiments

Eight cooperating lanes partition nuclear attraction while sharing the existing
one-electron scalar IR. Canonical public Schwarz ownership removes duplicate
symmetric evaluation; packed through-d bounds use 32 threads per block.
Materialized dddd forces use the existing class-major path under its unchanged
cache/topology/domain guards, retaining the generic bounded fallback.

The finite restricted primitive pure-J proposal epoch may spend its requested
cumulative omission allowance up to **1e-9**. The immutable physical owner,
full-density closure and ordinary graph/warm replay retain **1e-12 or the
user's tighter target**. Hybrid, unrestricted and nonprimitive caps are
unchanged. Neither convergence thresholds nor the maximum provisional epoch
are widened. The existing coarse-screening off control restores the tighter
proposal route.

The preceding combined v1 default remained slower: cold 32.5703 s versus
32.4400 s. Two research rounds at the bounded 1e-9 proposal cap produced
31.3836/31.4514 s versus GPU4PySCF 32.5536/32.5931 s, with force changes from
the tighter cap below 3e-11 Eh/Bohr. The clean v2 default results above use no
research override. MD workers 512/1024/2048, SCF tile 512 and XC matrix tile 8
did not yield stable complete-endpoint gains. Split linking regressed; a full
no-RDC build timed out after 60 minutes and has no valid endpoint measurement.
All selected experimental receipts are retained under `experiments/`; none
replaces a clean default population.

The separate intrusive census preserves all ten cold Fock builds and the
three warm builds. It measures the following actual primitive work; its elapsed
time is excluded from every performance table.

| Phase | Counter | v1 | v2 | Reduction |
| --- | --- | ---: | ---: | ---: |
| Cold, all ten builds | uniform_roots | 14,739,657,202 | 12,125,063,162 | 17.74% |
| Cold, all ten builds | uniform_summands | 777,250,604,704 | 640,895,744,240 | 17.54% |
| Six provisional delta builds | uniform_roots | 8,509,727,772 | 5,895,133,732 | 30.72% |
| Six provisional delta builds | uniform_summands | 478,580,102,192 | 342,225,241,728 | 28.49% |
| Three warm builds | uniform_roots | 4,853,733,750 | 4,853,733,750 | 0.00% |
| Three warm builds | uniform_summands | 267,503,446,848 | 267,503,446,848 | 0.00% |

Only calls 2–7 receive the provisional 1e-9 cap. Calls 1 and 8–13 retain
1e-12; all full-density and warm work counters match v1 exactly. Diagnostic
versus clean final force differences stay below 1e-11 Eh/Bohr, with identical
iteration counts. The v2 diagnostic energy agrees exactly with its clean
receipt. Counter logs and original observable receipts are retained.


## Provenance and replay

The measured `r2-default-v2` image is based on
`42b266917f6676ecf0559fb8965a4e19f60d0fe3` plus the exact retained
`source.patch.xz`. Its frozen archive SHA-256 is
`e4af728d14fc6279faa146748b65fcc38d87b98dbe5e0ee0233d3df5658b6b71`.
The parent commit alone does not represent the measured dirty source. All
13,178 archived files match the frozen tree; all 1,858 compiler-input files
match the actual build tree. The core library hash is
`9ad12bbe8e66f8dfb002e68a2b9d90e2e314f475bfc141dd3701258082407462`.
Core/AOT hashes, exact commands, build allocation, cache version and shared
cache statistics remain in `evidence.json.xz`. CUDA 12.9.1/GCC 13.3,
Release/sm120, sccache 0.16.0 and one math thread were used. Scheduler device
visibility is preserved. Publication adds documentation and retained evidence;
production sources remain identical to the measured archive.

Full build output, source archives and binaries remain in ignored artifacts
at `n1:/data/jzzeng/cold-opt-20261025/`. No external release asset is required.
The tracked compressed original receipts support host-only replay:

```bash
PYTHONPATH=python:. python benchmarks/results/cold-r2-defaults-20261026/recheck.py
```

This recomputes all 144 main calls' common qualifications and cold medians,
including the retained failures, plus the eight larger/moved comparisons.
The [decision note](../../../.agents/notes/implemented/performance/2026-10-26-direct-r2-cold-defaults.md)
records scope, fallbacks and conditions for revisiting these defaults.
