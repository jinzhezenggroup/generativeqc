# Decision: reuse the existing GCC CI ceiling for ordinary builds

Status: implemented
Date: 2026-11-12

## Problem and measured evidence

The ordinary GCC CPU job's 15-minute total budget stopped progressing builds
before CTest after the addition of 133 generated Gaussian-ECP translation units.
This was not a CTest timeout or an observed compiler failure.

In [run 43052160517, attempt 2](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/attempts/2),
PR head `c51adb1f5aa76d3ef52d1e82808631b8b107ab66` checked out synthetic merge
`8aa5af5790256137e15802ce8ec1908c0acade5a`:

- GCC job `130875677414` spent about 57 seconds in setup/configure and 14m15s
  building before cancellation at action 497/729. It completed 122/133
  Gaussian-ECP objects, with 193 compiler-cache hits and 226 misses.
- Clang job `130875676159` spent about 51 seconds in setup/configure and 19m22s
  building before cancellation at action 555/729, with 223 hits and 254 misses.
- Neither job reached CTest. Successful Python-native builds in jobs
  `130875676256` and `130875676276` took 17m30s and 17m42s for configure/build,
  including all 133 Gaussian-ECP objects. Those builds omit native test targets
  and use wheel configuration, so their success does not qualify CPU CTest.

In [run 43138869481](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43138869481),
PR head `6667d3722e5f0aa39a5ad8b26bad50f03afbbe20` checked out synthetic merge
`cc62754d44a5406e910e3a2bd45c7b7645b3b596`:

- [GCC job 130920737235](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43138869481/job/130920737235)
  spent 62 seconds in setup/configure and 14m11s building. Cancellation at
  02:20:08 UTC stopped action 499/729 after 124/133 Gaussian-ECP objects;
  the last object completed five seconds earlier. Cache statistics were
  193 hits and 228 misses. CTest did not run.
- [Clang job 130920737181](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43138869481/job/130920737181)
  completed in 18m09s: about 44 seconds setup/configure, 15m40s building and
  101.10 seconds in CTest. All 141 CTests passed, none skipped. All 133
  Gaussian-ECP objects completed; the cache recorded 256 hits and 287 misses.

Both latest jobs restored the same newer master snapshot. Their logs contained
no failed compiler command, out-of-memory kill, or disk-full error. A restored
snapshot is not a promise of hits for newly introduced source. CPU PR consumers
intentionally do not publish their newly compiled objects, so a rerun does not
inherently warm these PR-specific objects.

## Decision and estimated margin

Reuse the existing 25-minute GCC scheduled/manual ceiling for ordinary GCC
jobs, including pull requests and merge groups. Keep Clang at 20 minutes and
scheduled/manual GCC at 25 minutes. Change no build or CTest command, compiler
flag, cache identity, test selection, individual test timeout, or scientific
acceptance gate.

The new margin is an estimate, not a measured GCC completion time. GCC's last
completed object, `ecp_gaussian_5_2_projector2_0.cpp.o`, finished about 15m08s
into the job. Clang finished that same object about 12m53s into its job, then
needed another 5m14s to finish building and testing. Transferring that tail
suggests approximately 20m22s for GCC; allowing additional compiler/test
variation suggests roughly 20-22 minutes, with an estimated 3-5 minutes of
headroom under the reused ceiling. Different scheduling and compiler behavior
make this a planning estimate only. These measurements do not include a complete
GCC CTest run at that PR head. The repaired budget still needs a complete hosted
GCC job.

## Rejected alternatives and preserved boundaries

- Do not increase Clang solely because its earlier attempt timed out: its latest
  complete build and CTest fit the existing ceiling.
- Do not change source optimization, scientific work, test coverage, cache
  correctness, or reference tolerances to make this CI budget fit.
- Do not treat the GCC cancellation as a numerical failure or a successful test.
- Do not expect an identical rerun to manufacture a trusted warm PR cache.

The budget regression tests evaluate every compiler/event combination and
accept equivalent expressions. They continue protecting required build/CTest
work and independent trusted master-cache publication.

## Revisit when

Revisit this ceiling using complete GCC build/test timings if 25 minutes still
truncates healthy work, if an actual compiler or test failure appears, or if
later source growth materially changes the compile workload. No runtime
performance or broader numerical qualification is inferred from these CI logs.

## Subsequent evidence

The [later Clang budget decision](2026-11-12-cpu-clang-ci-build-headroom.md)
supersedes this note's choice to retain Clang at 20 minutes after another
same-production-source build exceeded it. The observations and GCC estimate
above remain the historical evidence used for this earlier decision.
