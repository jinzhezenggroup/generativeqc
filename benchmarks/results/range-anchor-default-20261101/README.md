# Guarded default range warm anchor

An unset `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY` now selects the existing
force-headroom-guarded deferral. Explicit `0` remains the eager control. The
candidate changes neither quadrature nor precision, convergence thresholds,
public tiles, source admission budgets or GPU compatibility. It avoids an
incumbent-provider energy action only when an audited retained source can serve
warm replay and the real additional force envelope fits current device memory.

Native source identity:
`6be420fb4ee14175922194d61b41e3a20f13665968bc012b474b21cada96f4cc`.
The retained dirty patch is relative to `2c64727c65aebf76f9219bdecba76e77954c59f9`;
that base alone does not contain the candidate. The source archive, measured
libraries, allocation identities, test hashes and driver controls are retained
in the authenticated receipts. A later test-fixture cleanup has its own patch
and test hashes; it does not change native source identity.

## Complete endpoint controls

Water96/WB97M-V, spherical def2-SVP, fitted full-range J/K with exact range
correction, stock XC/VV10 grids, SCF tile256 and force tile1024. Each finite Slurm
allocation reserves eight CPU slots and single-threaded math libraries. The
same binary runs a separate conditioning owner, then eager/default/default/eager
owners, each with a cold energy/full-force endpoint and three frozen warm calls.
Cold includes setup. All ten populations / 36 endpoints independently pass the
unchanged README energy, full-force and reference-grid-stability gates.

Negative changes mean faster than eager GenerativeQC on that device; these are
not new GPU4PySCF rankings.

| Device / allocation | Cold | First warm | Pooled warm | Cold + three warm |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 / n4 1427 | +0.016% | -0.015% | -0.003% | +0.008% |
| RTX PRO 6000 / n2 3857 | -5.999% | +0.010% | +0.001% | -3.793% |

PRO6000 cold medians are 221.521990 s eager and 208.231956 s default; physical
iterations stay at 28 while Focks fall from 29 to 28. On 5090 the force envelope
retains eager work in both arms (28 iterations / 29 Focks), with cold medians
232.358061 s and 232.394120 s: effectively unchanged. Every warm call on both
devices uses one iteration and one Fock. Maximum paired errors are 1.82e-12 Eh
and 1.87e-12 Eh/Bohr. The reader prints all samples, medians, ranges and median
absolute deviations; it does not select the fastest observation.

A separate 108-atom / 32 GiB 5090 control (n1 15068) completes cold, retained,
evicted-fallback, second, moved and moved-warm endpoints in both arms. It retains
identical work (cold 31/32, moved 20/21, warm 1/1), with maximum paired errors
1.37e-12 Eh and 6.76e-13 Eh/Bohr. Complete lifecycles are 925.786899 s eager and
925.575312 s default. This single larger pair is a differential/lifetime control,
not a speedup ranking, and does not replace the earlier incomplete 120-atom runs.

## Validation and negative evidence

85 host cases pass. Both n4 1427 and n2 3857 pass all 66 GPU cases, covering
eager/explicit/default owners, frozen seeds, policy/capacity/factor invalidation,
eviction, failed begin/retry, snapshot exports/imports, residency tiers and moved
geometries. Small systems retain independent GPU4PySCF energy/full-force gates.

Native-only Compute Sanitizer memcheck (n5 1958) passes five eviction, retry and
snapshot cases, with zero errors and zero leaked bytes. It uses fresh independent
GPU4PySCF observables from n5 1957 through an authenticated fixture adapter; these
are only comparison targets, never native seed inputs or production work.

Combined reference/native sanitizer execution remains unsuccessful. Jobs1953
and1955 pass two cases then fail three snapshot cases after allocation failure;
explicit GC/pool cleanup did not fix this. Isolated eager snapshot job1954 passes
its scientific test but still reports leaks. The reference-only instrumented
job1959, which constructs no native owner, reproduces about 14.9 GB retained
allocations and 6,757 sanitizer errors. Two ordinary reference calls in job1957
leave the CuPy pool empty after cleanup and do not accumulate device memory.
This isolates an instrumentation/reference-engine interaction, but does not
establish its underlying cause or a clean combined sanitizer run. All failed
outcomes and log digests remain retained; full transient logs stay in ignored
local artifacts.

Initial jobs1426/3856 lacked reference assets, job1951 had an incorrect BLAS
provider path, and job1952 lacked the reference basis. These are failed deployment
attempts, not timings. Job1956 failed in diagnostic type introspection before its
retry. Successful jobs use the corrected inputs and verify actual binaries.

A100 SSH still resets before authentication. This exact candidate has **no A100
measurements**; earlier opt-in measurements keep their original identities.
Previously submitted A100 jobs224223/224225/224229 are unretrieved and were not
resubmitted. All-functional parity and performance on other unmeasured devices
remain open.

## Replay

```bash
PYTHONPATH=python:. python benchmarks/results/range-anchor-default-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/range-anchor-default-20261101/report.py
```

Both commands authenticate the exact FP64 v3 streams and original receipt texts,
replay independent numerical gates, and bind source, binary, scheduler and work
counts before computing complete timings. The receipts retain launch scripts,
controls, patches and the larger lifecycle driver. Real GPU reruns require finite
scheduled allocations and preserved scheduler device visibility.

See the [decision note](../../../.agents/notes/implemented/performance/2026-11-01-default-range-warm-anchor.md)
and [earlier rejected unconditional control](../deferred-range-anchor-20261101/README.md).
