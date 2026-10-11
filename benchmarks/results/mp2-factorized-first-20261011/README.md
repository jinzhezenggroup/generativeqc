# Compact factorized MP2 first-stage evidence

Bounded CPU qualification for one #1626 slice. `cpu-summary.json.gz` lists the
completed pairs and the exact collection time; unfinished matrix cases remain
unqualified. This evidence does not qualify a CUDA device, full UMP2 forces,
dense RCCSD(T), broader endpoint profitability, or issue closure.

The paired actual-source builds use base `637466c624c4c4c4dd4eb7c511a2c88675744325`
and the common C++ source SHA recorded in `provenance.json`. The production
common/header/native probe did not change after qualification. Python test type
annotations and canonical LF disposition metadata are publication-only changes.
`njzjz-bot` subsequently repaired the formatted tracker publication
hash and added artifact-hash regression checks. Neither changes the qualified
production source or historical measurements.

`common-receipts.json` contains executed counters and scoped allocation receipts
from the actual instrumented common C++ translation unit. For the Cartesian d/s
fixture, second accumulation iterations fall from 16807 to 3724 and shell-buffer
peak payload from 53040 to 40320 bytes. Every compact/staged callback weight and
gradient matched exactly; both layouts passed the independent direct oracle.
32 synthetic plus 16 independent PySCF 2.14.0 ERI-center tests passed.

`h2-records.json.gz` retains the initial four raw H2 endpoint/timing records.
`cpu-endpoints.json.gz` adds completed Notebook and fresh HPC records, build
provenance, and the exact executed tracker/experiment texts. Loaded
library source identity must equal the recomputed canonical inventory identity.
The full cold, warm and displaced E+F results match exactly between builds.
Independent PySCF, three-step finite differences, covariance/invariants, changed
geometry, batch and response gates pass. The original overall/resource status is
FAIL because production reports its documented unmeasured peak sentinel. The
CPU driver's vacuous CPU/CUDA parity flag is not CUDA execution evidence.

H2's measured ordinary C++ new/new[] peak is unchanged at 18911 bytes; total
requested payload decreases by 48 bytes per E+F. This is negative evidence for
an endpoint-peak benefit. Payload interposition excludes C malloc, aligned-new,
allocator overhead and borrowed Python data. RSS is a cumulative process high
water mark and includes the Python process/oracle. Timings from those runs are
interposed; separate timing records omit the tracker and contain seven repeated
warm E+F measurements. These small, sequential samples do not establish an
endpoint speedup; `cpu-summary.json.gz` retains their medians and ranges separately.

## Additional completed CPU evidence

Notebook H2 and LiH baseline/candidate cold, warm and displaced E+F values match
exactly. All nonvacuous scientific gates pass; original overall/resource FAIL
remains visible. LiH's maximum independent force error is 3.971e-8 Hartree/Bohr
and its finest finite-difference error is 2.321e-8 Hartree/Bohr. The raw records
retain all three finite-difference steps, physical invariants and response data.

| Complete E+F payload observation | H2 baseline / candidate | LiH baseline / candidate |
| --- | --- | --- |
| Ordinary C++ peak bytes | 18911 / 18911 | 289735 / 289735 |
| Total requested bytes per warm call | 37571231 / 37571183 | 151608649823 / 151608642143 |
| Ordinary new/new[] calls | 579370 / 579370 | 301806200 / 301806200 |
| Energy tiles | 1 / 1 | 4 / 4 |
| Response iterations | 0 / 0 | 4 / 4 |

The unchanged peaks and allocation-call counts are negative endpoint evidence.
The large total requested payload is cumulative allocation traffic, not
simultaneous storage. Other endpoint consumers still execute; the common
probe's reduced accumulation counts do not imply that
total integral or response work was removed. The scientific phase includes
independent PySCF and all additional target calls; its time is retained separately
from ordinary cold/warm/displaced endpoint timing.

| Untracked complete E+F seconds, baseline / candidate | Cold construction + E+F | Median of 7 warm E+F | Displaced E+F |
| --- | --- | --- | --- |
| Notebook H2 | 0.045590 / 0.043633 | 0.035045 / 0.030462 | 0.035390 / 0.031636 |
| Notebook LiH | 43.7174 / 41.5407 | 42.7012 / 40.8345 | 42.8734 / 40.8308 |
| HPC H2 | 0.044421 / 0.043746 | 0.032925 / 0.032678 | 0.033284 / 0.032703 |

These use single-thread OMP/OpenBLAS/MKL settings on separately admitted 4-CPU
instances. Baseline and candidate run sequentially, without CPU pinning or a
crossed run order; host drift and small samples prevent a general speedup claim.
Imports are outside cold timing. Individual repeats and their ranges remain in
the raw bundle and summary, along with the much longer scientific/oracle phase.

The HPC GCC 11.4.0 / Python 3.12.15 environment freshly built both CPU Release
libraries through verified sccache 0.16.0. Baseline 34 and candidate 50 tests
passed, including all 16 independent molecular cases for the candidate. Verbose
logs contain 137 compiler-launcher invocations per build; hashes, generated build
identity and probe output are retained. Its H2 pair has exact value parity and
passes the same scientific gates. Later-collected cache counters are zero after
the earlier server processes were no longer live; they do not establish build cache hits or
misses. No rebuild or cache clearing was performed to manufacture statistics.

The published `payload.cpp` received a formatting-only pre-commit change after
measurement. `artifacts.payload.cpp` binds published bytes;
`tracker.source_sha256` and the two loaded tracker hashes bind executed inputs.
The bundle preserves the exact measured source, so reproducing it does not
depend on retaining an unmerged commit after squash.

## Upstream integration and storage

The summary is losslessly gzip-compressed; decode it with Python
`gzip.decompress` before parsing JSON. Its decoded SHA and byte count are in
`provenance.json`; every original record and value is retained.

Integration commit `1ac537e67` preserves the qualified MP2 common/header/native
probe byte-for-byte, while merging newer upstream production
inputs. The historical source/library identities above qualify the original
measured snapshots, not the integrated build. Latest-head build and endpoint
qualification require separate evidence; CI success alone is not that scientific
acceptance.

## Reproduction

Use immutable LF baseline/candidate source snapshots and a verified compiler
cache. Build CPU Release libraries with CUDA and packaged CPU force AOT disabled,
then select each actual library explicitly with `GENERATIVEQC_LIBRARY`. Retain
compiler-cache statistics and verbose launcher commands. The bounded common
harness is `tests/python/test_mp2_derivative_pullback_probe.py`. Default CI runs
unmodified C++ with independent oracles. Set `GENERATIVEQC_MP2_WORK_PROBE=1`
only for the opt-in, source-bound accumulation/work audit; numerical, staged
fallback and molecular ERI-center tests remain enabled by default.

Compile `payload.cpp` to a PIC object through that launcher and link a shared
library. Run `endpoint.py --source SOURCE --case h2 --scientific --payload
--output NEW_DIRECTORY` with that library in `LD_PRELOAD`. For ordinary timing,
unset `LD_PRELOAD` and run `endpoint.py --source SOURCE --case h2 --repeats 7
--output ANOTHER_NEW_DIRECTORY`. Set OMP/OpenBLAS/MKL thread counts to one.
Python module imports occur outside the timed Calculator construction/E+F scope.
Keep failed original qualification gates visible and report all target and
independent-oracle/FD costs separately. No installed-build telemetry is replaced
by resource-plan estimates.
