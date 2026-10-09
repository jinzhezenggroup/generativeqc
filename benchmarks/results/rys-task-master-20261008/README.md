# Quartet-parallel K qualification after main-branch AO/XC promotions

Qualification base: `99a2a5cac926edc69c28453977d1bc9b70d8bd06`.
Implementation parent: `c8117cc45`, rebased without conflicts onto that base.
The final changed-source capsule extends that implementation to three roots;
its native identity and binary checksum, not the parent alone, pin this cohort.
This independently labeled cohort includes the intervening native AO radial/axis
reuse and compact XC defaults. The older `4385f727` observations remain unchanged
in the adjacent [frozen-default report](../rys-task-default-20261008/README.md).

## Complete Cold E+F

RTX 5090, Slurm n1/`node1`, finite **exclusive-node** allocation, job
6688. Five fresh-process pairs at 96 atoms, three at 48.
Predeclared 96 order: `default0,incumbent0,incumbent1,default1,default2,incumbent2,
incumbent3,default3,default4,incumbent4`; 48 uses the first six entries.
The candidate actively unsets the K selector and class filters: this is the actual
default, not opt-in twelve-class Rys-task. The same binary executes both modes.

Strict FP64 PBE0 RKS water clusters, offline spherical def2-SVP (96 atoms/768 AOs),
48x16x32 grid, screening 1e-12, E/D tolerances 1e-12/1e-10, maximum 100 iterations.
Timed: calculator construction, CUDA context/preparation, first SCF and analytic
forces, host return and owner teardown. No supplied density, priming or warm owner;
imports/input decoding are excluded and persistent compilation/artifact caches are
equally reused. This is execution-Cold, not cache-empty JIT compilation timing.

| Atoms | Incumbent median | Default median | Less median time | Less mean time |
| --- | ---: | ---: | ---: | ---: |
| 96 | 116.382170 s | 101.296905 s | 12.96% | 13.03% |
| 48 | 49.426070 s | 45.583701 s | 7.77% | 6.79% |

Actual semantic work:
- 96 atoms, incumbent/default: `[19, 22, 19, 18, 19]` / `[17, 17, 17, 17, 19]`.
- 48 atoms, incumbent/default: `[19, 19, 19]` / `[19, 19, 20]`.

All converged observations remain in the cohort, including slower samples.
Both mean and median gates pass. Do not interpret an endpoint ratio with different
Fock counts as a same-iteration kernel ratio. All independent cached GPU4PySCF
energy/force gates pass (1e-8 Ha/1e-7 Ha/Bohr); exact protocols and reference
provenance remain in [the original report](../rys-task-cold-20261008/README.md).

## Numerical and mechanism checks

Separate exclusive-node job 6689 validates 176 public/prepared
Libcint K cases: all eleven selector/filter configurations, both spins,
symmetric/asymmetric signed densities, zero/tiny/restored density and frozen owners.
Maximum matrix error is 6.370e-12.
Full selector preflight verifies capability 36831, preference 36820, coverage
intersection, `all` not broadening preference and `none` selecting no candidate.

Six separate fixed-density processes compare incumbent, old component-Rys and
actual default. Full admission vectors match across all repeats and four density
scales, including zero tails. At full density, six device-K observations per mode
(two processes, three repeats each) give these diagnostic medians:

| Atoms | Incumbent K | Old component-Rys K | Default task-Rys K | Admitted shell quartets |
| --- | ---: | ---: | ---: | ---: |
| 96 | 1.681313 s | 4.294012 s | 1.339690 s | 81,907,624 |
| 48 | 0.951970 s | 1.582291 s | 0.749990 s | 23,480,495 |

These medians are not added to Cold timings and do not prove equal primitive/root
work. Complete per-class vectors and matrix errors remain in the raw records.
The final implementation passes 864 standalone Libcint
matrices and 96 cases per memcheck/racecheck/synccheck, with zero findings.
NCU hardware counters remain unavailable.

Current-base CPU suite: 248 passed, 18 skipped; 502 compiler and 223 shared-SCF
modules, zero dependency errors. Default-promotion audit: 40 entries, 84 controls;
tests prohibit unqualified class/profile expansion. Ruff and diff checks pass.

## Scheduling incident retained, not cherry-picked

The accurate five-class 128-thread current-base cohort, job 6680, also remains
as a valid negative: incumbent/default medians 106.313781/107.403082 s, means
106.354335/106.083368 s, Focks `[17,17,17]`/`[17,18,18]`. It fails the median
gate and does not run 48 atoms. It is not pooled with the final nine-class
source or excluded for timing. A separate two-root limited-unrolling prototype
reduces static registers but does not improve fixed-density K; it is rejected
before Cold and does not change the retained two-root producer. Both prototype
and negative observations remain in the archives.

Initial latest-base jobs 6678 and 6679 unexpectedly ran concurrently on node1
with the same requested RTX 5090 resource. Before acceptance, the entire
interrupted cohort was invalidated and retained under `rebased-overlap-invalidated`
in the observations archive, with job descriptions and cancellation timestamps.
It contributes no qualified performance observations, even those collected before
the overlap. Accurate slow endpoints are not an exclusion criterion. Qualification
restarts the full predeclared cohort with `srun --exclusive`; later mechanism work
starts only after endpoint timing completes. This infrastructure exclusion is
distinct from the valid negative 32-thread and narrowed-class cohorts retained
in the frozen report.

## Identity and reproduction

Main binary SHA256: `02bc03610e6fbabcd4d47b8584974565a4371a7dbbb78e954a84c089c6794001`.
Native scientific source identity: `31f6418b2944bac013426a98a016ea8ab4e85a3641b135a97141e6d1822cef7c`.
Archive SHA256 values are recorded in `summary.json`. `observations.tar.gz` retains
complete endpoints/forces, integrated/fixed-K records and build/cache provenance;
`validation.tar.gz` retains qualification drivers, checks and changed-source capsule.

Reconstruct from the pinned base plus `qualified-rebased-source-changes.tar.gz`,
build with the verified shared ccache and sm_120 profile, and keep both stationary
PBE0 force companions/manifests beside the main library. Use the archived driver
under a finite exclusive-node Slurm allocation; preserve assigned CUDA visibility.
The archived `build-final-n1.sh` is parameterized: set `RYS_BUILD_SOURCE` to the
reconstructed `promotion-source` checkout and `RYS_BUILD_PREFIX=promotion-nine`.
Then use `run-promotion-nine-cold-n1.sh` and
`run-promotion-nine-validation-n1.sh`, sequentially, under separate
`srun --partition=main --gres=gpu:5090:1 --exclusive --nodes=1 --ntasks=1`
allocations with explicit finite `--time` (01:00:00 for Cold, 00:30:00 for
validation). These archived drivers document the qualification host paths;
adapt tool/environment paths without changing the scientific inputs or gates.
Fresh workers and summary gates are implemented by `benchmarks.rys_task_cold`.
Do not pool the different source identities or Slurm cohorts.

Default is only `psps,ppps,dsss,dpss,dsps,ddss,dsds,dpps,dspp` on the `sm_120` profile, with 128-thread
CTAs/four independent warp queues. Other profiles/classes retain incumbent;
`GENERATIVEQC_DIRECT_K_FOCK_LOWERING=incumbent` fully rolls back. Explicit
`rys-task` still exposes twelve classes for separately qualified experiments.
J, analytic force recurrence, precision and SCF convergence tolerances are unchanged.
