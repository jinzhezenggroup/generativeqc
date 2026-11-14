# Decision: bound Clang CI with room for observed build variation

Status: implemented
Date: 2026-11-12

## Prior decision and new evidence

The [earlier GCC budget decision](2026-11-12-cpu-gcc-ci-build-headroom.md)
retained Clang at 20 minutes because a complete Clang build and 141 CTests
finished in 18m09s. That observation remains valid; a later run demonstrated
that it did not leave enough room for variation on the same production sources.

The successful reference is [Clang job 130920737181](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43138869481/job/130920737181):
PR head `6667d3722e5f0aa39a5ad8b26bad50f03afbbe20`, synthetic merge
`cc62754d44a5406e910e3a2bd45c7b7645b3b596`, master parent
`a7241b4fdaf586fef25da59913e4d267aca67461`. Its build took 15m40s,
all 141 CTests passed in 101.10 seconds, and its cache recorded 256 hits and
287 misses.

In [run 43146888679](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43146888679),
PR head `0c4f537d5c044830688ee63e8d1f3e9fdaa46524` checked out synthetic merge
`a35e75c9e54407626bb247c4aa6f0fc6a7c43ebf` with the same master parent.
The intervening changes were CI workflow/tests/notes only, not production
source or native test changes. Both Clang jobs restored the same master-cache
snapshot ending `14591f76fc40df14cd1dd6d082a1c726054e78e5cbc1e523238d8cf73188b4a0`.

- [GCC job 130946715585](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43146888679/job/130946715585)
  completed successfully in 15m48s, including all 141 CTests. Retain its
  25-minute ceiling.
- [Clang job 130946715650](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43146888679/job/130946715650)
  started at 03:14:56 UTC, completed setup/configure in about 48 seconds,
  then built from 03:15:44 to cancellation at 03:35:09. The last completed
  action, 603/729, appeared at 03:35:07, two seconds before cancellation.
  It completed 130/133 generated Gaussian-ECP objects. Cache statistics were
  525 calls, 252 hits, 273 misses, and no uncacheable calls. There was no
  failed compiler command, out-of-memory kill, or disk-full error. CTest
  never started; this run provides no new Clang test result.

## Matched tail and estimated margin

The last completed object in the truncated run was
`generativeqc_ks_device_resource_tests.dir/tests/native/test_ks_device_resources.cpp.o`.
It completed 20m11s into the later job versus 15m50s into the successful
reference, about 4m21s slower. In that reference, another 34.5 seconds finished
the build and about 102 seconds finished CTest. Transferring this observed
tail projects approximately 22m27s total for the later job, before further
tail/test variation and cleanup.

Action counts alone are insufficient: the three remaining large ECP objects
were `ecp_gaussian_5_3_projector0_1.cpp.o`,
`ecp_gaussian_5_3_projector1_1.cpp.o`, and
`ecp_gaussian_5_3_projector2_1.cpp.o`. Their completion order differed between
runs; the successful reference had already finished the first before the
matched native-test object. A planning range of roughly 22.5-24 minutes
therefore allows additional tail/test variation, but is still an estimate,
not a measured completed run.

A 25-minute ceiling would leave only about 1-2.5 minutes against that range,
less than the observed 4m21s checkpoint variation. Select a finite 30-minute
Clang ceiling, giving an estimated 6-7.5 minutes of headroom. This is a
conservative resource allowance, not a claim that the current Clang run passed
or a guarantee of future completion.

## Decision and preserved boundaries

Change only the Clang CPU job's outer timeout from 20 to 30 minutes. Retain
GCC at 25 minutes for every eligible event. Preserve all Python job budgets,
compiler/coverage flags, compiler-cache identities and publication policy,
build and CTest commands, test selection, individual test timeouts, and
scientific acceptance gates. The behavioral budget tests continue evaluating
each compiler/event combination and accepting equivalent expressions.

Do not reduce scientific work, alter CPU optimization, skip CTest, weaken
cache correctness, or infer a numerical/runtime-performance regression from
this compile-only timeout. An identical rerun is not a substitute for a
budget that accommodates observed work and variation.

## Revisit when

Require a complete hosted Clang result after this repair. If 30 minutes still
truncates progressing work, inspect actual object/test timings and resource
failures before another adjustment. If compilation or CTest reports a real
failure, investigate it directly rather than treating it as missing time.
