# Compact factorized MP2 first-stage evidence

Partial qualification for one #1626 slice. Broader complete CPU endpoints and
the second HPC compiler environment are still running; this evidence does not
qualify a CUDA device, full UMP2 forces, dense RCCSD(T), or issue closure.

The paired actual-source builds use base `637466c624c4c4c4dd4eb7c511a2c88675744325`
and the common C++ source SHA recorded in `provenance.json`. The production
common/header/native probe did not change after qualification. Python test type
annotations and canonical LF disposition metadata are publication-only changes.

`common-receipts.json` contains executed counters and scoped allocation receipts
from the actual instrumented common C++ translation unit. For the Cartesian d/s
fixture, second accumulation iterations fall from 16807 to 3724 and shell-buffer
peak payload from 53040 to 40320 bytes. Every compact/staged callback weight and
gradient matched exactly; both layouts passed the independent direct oracle.
32 synthetic plus 16 independent PySCF 2.14.0 ERI-center tests passed.

`h2-records.json.gz` retains all four raw H2 endpoint/timing records. Loaded
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
warm E+F measurements. These small samples do not establish an endpoint speedup.

## Reproduction

Use immutable LF baseline/candidate source snapshots and a verified compiler
cache. Build CPU Release libraries with CUDA and packaged CPU force AOT disabled,
then select each actual library explicitly with `GENERATIVEQC_LIBRARY`. Retain
compiler-cache statistics and verbose launcher commands. The bounded common
harness is `tests/python/test_mp2_derivative_pullback_probe.py`.

Compile `payload.cpp` to a PIC object through that launcher and link a shared
library. Run `endpoint.py --source SOURCE --case h2 --scientific --payload
--output NEW_DIRECTORY` with that library in `LD_PRELOAD`. For ordinary timing,
unset `LD_PRELOAD` and run `endpoint.py --source SOURCE --case h2 --repeats 7
--output ANOTHER_NEW_DIRECTORY`. Set OMP/OpenBLAS/MKL thread counts to one.
Python module imports occur outside the timed Calculator construction/E+F scope.
Keep failed original qualification gates visible and report all target and
independent-oracle/FD costs separately. No installed-build telemetry is replaced
by resource-plan estimates.
