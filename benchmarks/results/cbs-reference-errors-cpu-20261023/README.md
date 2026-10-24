# Native CPU CBS reference errors

Refs #2681. This publication records the full bounded **CPU conventional exact**
reference slice, not the unmeasured parent DF/CUDA scope or a current-master SDK.

Measured source: `bbd41a42764e8d22d15c1348f70a633961d4dcf5`. Native source identity:
`61887ec62ded5f806c1f1efcf0b83dfeff70a24ce2b49472d81b80d49e305f3f`. Library SHA256: `67ffc728d989014fa383faa8a784b09ab607a30e284fd404d4c556dbb37176b4`.
The frozen Git LF snapshot, inputs, basis identities, model, source-matched
build, compiler commands/cache statistics, original modes, actual PIDs and full
raw endpoint/trace/log files are retained in the raw containers.

## Results and unchanged gates

All **13 applicable native CPU tests**, **65 CBS unit tests** and **6 budget
validator tests** passed. The native test union is checked against the retained
full collection. Each physical spherical H2 cc-pVDZ/cc-pVTZ and synthetic
water-algebra fixture ran once with its E+F, five constituents, composite,
energy-only parity and two reconverged directional steps. H2 explicitly used
`GENERATIVEQC_CBS_LARGE_TEST=1`. No reference data were regenerated.

`errors.csv` lists energy errors and maximum/RMS force errors. `summary.json`
also retains algebra terms, independent PySCF directional comparisons, native
reconverged derivatives, and per-constituent displaced energies/projections.
Energy tolerance is 1e-9 Eh; force and composite directional tolerance is
2e-6 Eh/Bohr. Component FD-to-PySCF-force projections are labeled distinctly
from the independent stored composite finite-difference records.

The synthetic water basis is not physical water DZ/TZ qualification.
Correlation-only DF and full RI-JK analytic CBS require CUDA; both are
UNAVAILABLE for this CPU slice. CUDA was not enabled, no Slurm identity was
fabricated, and the original guard remains unchanged. PySCF appears only in
the independent committed oracle, never as the native production consumer.

## Reproduction and recovery

From a checkout with the frozen Git source available, run:

```text
python benchmarks/results/cbs-reference-errors-cpu-20261023/replay.py --verify-publication
python benchmarks/results/cbs-reference-errors-cpu-20261023/unpack.py --output /new/empty/directory
```

Replay requires only Python's standard library and Git. It verifies publication
bytes, the committed LF native source closure, reference inputs, full raw
identities and all numerical gates without running another calculation.
Recovery rejects existing destinations and restores exact raw bytes; original
Linux modes are applied on POSIX. Historical Windows preparation modes retain
their separate context. The raw input runners document original execution;
their old absolute roots are provenance, not commands to execute blindly.

## Capacity and interpretation

The first 64 MiB capacity preflight failed before scientific execution; the
complete water package also exceeded later headroom. Those failures and all
planning snapshots remain verbatim in `raw-build.json.xz`/`raw-audits.json.xz`.
The user explicitly authorized increasing capacity. The bounded **64→80 MiB**
proposal belongs to the separate capacity PR #2704. This CPU evidence change
does not modify policy; publication must fit the actual integrated policy and
tree. The 1 MiB per-file limit, 2 MiB change-review budget and every scientific
and protected-delivery gate remain unchanged. No other owner's historical
records were changed.

The integrated policy remains **64 MiB**. Normal upstream lossless compaction
restored sufficient capacity for this complete CPU publication; PR #2704 is
therefore not a prerequisite for this slice. Earlier capacity failures retain
their original source/tree context and are not reclassified as passing.

The same own 55-CPU/300-GiB Notebook retained source and SDK. Short supporting
tests overlapped H2; both configured 32 threads, so their theoretical sum was
64 above the 55-CPU allocation. Active-thread/peak-memory contention was not
measured. The overlap, allocation, actual commands and outcomes are retained;
no clean timing, speedup, allocator peak or performance qualification is claimed.
Public backend label `cpu_reference` denotes the executed native CPU ABI path.
Numerical PASS, storage acceptance, CI/review and actual merged delivery are
separate states; this file does not itself assert PR merge or close the parent.
