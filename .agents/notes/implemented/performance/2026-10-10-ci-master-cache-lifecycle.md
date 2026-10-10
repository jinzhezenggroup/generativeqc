# Decision: preserve master compiler-cache coverage and interrupted probe work

Status: implemented
Date: 2026-10-10

## Problem

The workflow inventory at master `985caaf01688bd6df3388b2c7985af3176efbf9d`
showed three distinct cache-lifecycle gaps. They must not be confused with an
absence of all default-branch execution:

- Master pushes seeded plain GCC and Clang objects, while ordinary GCC PR tests
  used the separate `ccache-cpu-v3-gcc-coverage-` namespace. Nightly coverage seeds
  did exist and succeeded on October 8 and 9. Both started cold, and later PRs
  again reported 295/295 misses. The evidence does not establish why older entries
  disappeared; merely adding another restore prefix cannot recover absent data.
- Python native-probe compiler work was saved only after successful tests. The
  October 9 scheduled core-b job restored no probe snapshot, compiled 160 misses,
  reached the 90-minute job limit, and skipped its save. Its post-test statistics
  still ran, demonstrating that valid compiler work existed at interruption.
- CodSpeed's compiler cache used a success-only post-job save after benchmarks.
  Independently, non-PR events never initialized the setup selector. All setup
  steps were skipped on master, but the benchmark action ran and failed because
  `.venv/bin/python` did not exist. A failed complete CI run also makes its CUDA
  snapshot ineligible under the existing successful-master-run trust policy.

## Decision

- Add only the missing GCC coverage configuration to the existing master CPU
  seed matrix. Match the PR compiler, flags, source-key inputs, build directory,
  and cache namespace. Keep plain GCC/Clang and avoid duplicate CTest/lcov work.
- Give Python tests explicit 30-minute routine and 90-minute full ceilings. Use
  finite 40/100-minute job limits to leave setup and cleanup headroom, with a
  two-minute cache-save limit. No tests, deselections, or scientific gates change.
- Save successfully compiled probe objects after failed or timed-out tests.
  Failed/interrupted runs use `-partial-<run>-<attempt>` snapshots so an incomplete
  immutable snapshot cannot occupy the complete primary key. Existing prefix
  restoration can recover that progress; a subsequent success can publish the
  canonical content-addressed key. Each shard retains its 256 MiB compiler limit.
- Default the CodSpeed setup selector to string `1`; the existing PR selector
  still explicitly overrides it for known non-performance changes. Split its
  compiler restore/save and save immediately after a successful library build.

## Invariants and rejected alternatives

Cache publication is acceleration, not test qualification. A failed test step
continues to fail the job. Merge-group probe caches remain restore-only. Compiler
validation, pinned inputs, source identities, permissions, benchmark selection,
CUDA snapshot provenance, and all required correctness gates remain unchanged.
Do not broaden trusted CUDA artifacts to failed workflow runs to hide CodSpeed's
setup failure; repair the existing producer instead.

The eight-workflow audit intentionally did not enable every workflow on push:
CuMetal already had successful daily master runs, and the CUDA resource job saved
all four caches during scheduled CI. CuMetal's empty QC-JIT path was attributable
to the existing upstream quarantine, not a missing trigger. Documentation and
pre-commit have dependency/tool caches but no comparable native compile seed.
PR overlap and advisory work audit require PR comparison context; merge-queue
cleanup has its own event-specific purpose and no compiler cache. The independent
wheel master-trigger repair is outside this change.

## Evidence and limits

- [October 8 scheduled GCC seed](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37723730751/job/113137046318):
  successful coverage save, 292 cold compiler calls.
- [October 9 scheduled GCC seed](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37880318519/job/113658303597):
  successful coverage save, 294 cold compiler calls.
- [October 9 scheduled core-b interruption](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37880318519/job/113658303754):
  167 cacheable calls, 160 misses, cancellation at the 90-minute job limit.
- [October 9 CuMetal schedule](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37975564895/job/113972898815):
  restored toolchain, saved object cache after 425 cold calls; QC quarantine left
  no JIT cache path to save. Successful workflow status does not certify the
  quarantined QC endpoints.
- [Master CodSpeed failure](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38014877582/job/114102822037):
  skipped setup, missing Python executable, exit 127.
- Source-level policy tests exercise cache-mode/flag parity, test/save budgets,
  interrupted-save guards, partial keys, early save ordering, and the actual PR
  selector shell. The new tests reject the pre-repair workflow.

Local policy tests do not simulate an actual GitHub timeout or prove future cache
hits. Hosted validation must confirm post-timeout save behavior and successful
master seed/baseline production. A hard runner loss can still prevent cleanup;
ten minutes of finite setup/cleanup headroom is not an unlimited guarantee.
