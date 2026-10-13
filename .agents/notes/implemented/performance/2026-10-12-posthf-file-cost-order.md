# Decision: start expensive complete post-HF files first

Status: implemented
Date: 2026-10-12

## Problem and measured evidence

The routine post-HF shard uses four `loadfile` workers. xdist 3.8.0 sorts
complete files by selected test count, which is a poor proxy for these scientific
workloads. The ten-case unrestricted physical-response file starts after several
cheap, high-count modules and becomes the last worker's long tail.

Two completed hosted runs on master `922bbfc8108d345b99af4a7cbba20f43dc75d74b`
provide the retained ordering evidence:

- PR #2386, run 38306920093, job 115003130176: checkout
  `5ba9cb16f` merges `e6cdea9d8c57de2755e9eab2fe9c40b93bed51ff` into that
  master. 361 passed, 57 skipped, no failures/errors; pytest wall time 1295.199 s
- PR #2387, run 38308786441, job 115009202834: checkout
  `71fc4b2ca0eca9d0c9b0002d074ad75cb2a6c509` merges
  `d8403fa59676f408004bbaf1412f90f2bf4075e3` into the same master. Identical
  418-case inventory/statuses; pytest wall time 1756.648 s

JUnit sums include each case's setup, call and teardown. They are worker-seconds,
not shard wall time:

| Complete file | Cases | #2386 seconds | #2387 seconds |
| --- | ---: | ---: | ---: |
| test_uccsd_lambda.py | 26 | 759.044 | 1019.014 |
| test_uccsd_t_owner.py | 30 | 633.333 | 862.381 |
| test_uccsd_physical_response.py | 10 | 610.745 | 783.560 |
| test_dft_complete_cpu.py | 44 | 384.535 | 525.518 |
| test_hessian_directional.py | 26 | 281.132 | 399.319 |
| test_cc_complete_gradient.py | 65 | 247.678 | 351.167 |
| test_ecp_heavy.py | 49 | 227.756 | 310.277 |
| test_ccsd_t_complete_gradient.py | 3 | 165.214 | 237.917 |
| test_ecp_stationary_cpu.py | 12 | 139.722 | 201.563 |
| test_cc_triples_response.py | 15 | 131.027 | 184.032 |
| test_ecp_f.py | 13 | 127.826 | 171.189 |
| test_hessian_reference.py | 27 | 84.026 | 117.469 |
| test_hessian_analytic.py | 21 | 62.869 | 84.492 |
| test_response_operator.py | 10 | 45.661 | 65.866 |
| test_hessian_hvp.py | 6 | 21.236 | 29.158 |
| test_ecp_f_projector.py | 20 | 12.015 | 16.398 |
| test_ccsd_t_gradient_validation.py | 7 | 9.528 | 14.031 |
| test_mp2_gradient.py | 34 | 6.826 | 9.449 |

The three UCCSD files account for 2003.122/2664.955 worker-seconds. In #2387,
last per-worker test reports occur at 21:28:19, 21:32:42, 21:35:52 and 21:40:45
UTC. The physical-response file owns the final worker. This demonstrates a
scheduling opportunity; those idle intervals are not a measured speedup.

The largest individual cases remain complete numerical checks:
`test_complete_ccsdt_gradient_matches_pinned_pyscf[h2o]` is 165.212/237.914 s;
the asymmetric-H6 UCCSD(T) energy/channel case is 120.369/163.993 s; and
`test_f_complete_hf_energy_and_force[0-cartesian-cpu]` is 119.654/160.162 s.
The analytic-Hessian H2 first-order fixture's setup is 53.21/70.69 s.
These aggregate durations do not isolate oracle, source-generation, compilation
or numerical-execution cost, and do not justify dropping any assertion.

Both runs have Python 3.11.14, pytest 9.1.1, xdist 3.8.0, pytest-cov 7.1.0,
four workers and coverage enabled. Regions differ (eastus/northcentralus).
Across 87 unrelated cases taking at least ten seconds in both runs, the median
duration ratio is 1.377. The slower second run is not an isolated ECP-oracle
regression or evidence of savings from freezing that oracle. Hardware/host-load
effects have not been isolated. Test-step startup through scheduling is roughly
12/16 seconds, rather than a multi-minute collection bottleneck.

## Decision and invariants

After the existing selector finishes, apply a stable permutation of already
selected routine post-HF file targets, using the descending whole-file ordering
above. Pass `--no-loadscope-reorder` so xdist does not replace it with test-count
order. The existing collection adapter only compacts core-a/b/c and passes these
explicit post-HF targets through unchanged.

- Keep four workers and `loadfile`; every file's fixtures and oracle caches stay
  local to one worker
- Preserve every selected target, multiplicity, deselection, coverage argument,
  skip, numerical/failure assertion and timeout
- Never add an absent priority file; retain missing selected paths so pytest
  fails closed, and preserve unmeasured/new targets in their original order
- Leave all other shards and scheduled/manual full-matrix scheduling unchanged;
  their costs and Hessian ownership differ from the routine measurements
- Do not disable coverage, freeze more references, increase runners, split live
  fixtures or claim that a theoretical balance estimate is hosted performance

The project requirement is `pytest-xdist>=3.8`, not an exact pin. Version 3.8.0
introduces the reorder option. Although its changelog describes `loadscope`,
the official `LoadFileScheduling` implementation inherits that scheduling code.
The regression exercises the actual installed scheduler to protect this boundary.

## Cache follow-through

Both hosted post-test overlays are cold: 485 cacheable misses, zero hits, and
457 uncacheable calls. The separate native-library phase has 227/232 hits.
Counts alone do not establish compilation time saved or oracle cost.

PR #2382 merged at 20:36:50 UTC. The latest CI schedule, run 38201206725 at
03:48:25, preceded it. Scheduled/manual master runs already publish default-ref
Python overlays; PR snapshots remain PR-scoped and merge groups restore only.
The master CMake producer does not seed these JIT/test objects. No new per-push
full-Python producer is necessary to make the default-ref overlay path exist.

## Validation

- Actual xdist 3.8.0 control: unchanged 52 synthetic nodes (44 passed, 8 skipped),
  four workers, one worker per file and one setup per module fixture. Default
  scheduling starts four high-count files; reordered scheduling starts the four
  measured-priority files
- Full shard/collector/scheduler control module passes, including unknown,
  missing, duplicate and empty target inventories and unchanged core discovery
- One-off comparison against the base selector covers all 40 event/shard
  combinations: identical target multisets, deselections, distribution modes and
  coverage arguments; only routine post-HF target order and reorder flag change
- The two existing hosted JUnit inventories each retain all 418 node identities
  and statuses when passed through the before/after file permutations. This is
  an inventory proof, not a new scientific run
- No new numerical, CPU benchmark, GPU, or full CI campaign was run locally

Revisit priorities using normal hosted CI/JUnit when file costs materially
change. A future xdist update must continue to satisfy the real scheduler test.

## Sources

- [#2386 post-HF job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38306920093/job/115003130176)
- [#2386 JUnit artifact](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38306920093/artifacts/11773942499)
- [#2387 post-HF job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38308786441/job/115009202834)
- [#2387 JUnit artifact](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38308786441/artifacts/11775381668)
- [xdist 3.8.0 changelog](https://pytest-xdist.readthedocs.io/en/stable/changelog.html#pytest-xdist-3-8-0-2025-06-30)
- [xdist 3.8.0 file grouping](https://github.com/pytest-dev/pytest-xdist/blob/v3.8.0/src/xdist/scheduler/loadfile.py)
- [xdist 3.8.0 scheduling](https://github.com/pytest-dev/pytest-xdist/blob/v3.8.0/src/xdist/scheduler/loadscope.py)
