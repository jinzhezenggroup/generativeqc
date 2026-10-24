# Bounded automatic DF charge partitions

Build source: `ab2b592050ecc5214508f98ac6a279efdcbe8a30`.
Endpoint/test source: `f01d512f9165d68e2aceb2c86ea2d8d4c3a99987`.
Core SHA-256:
`399a8486b6cc5d930e1d60dc51c0074a9db3eeb7690cec22fcdefdf306046df4`.
The two sources differ only in the long-source test fixtures; 12,429 files
were authenticated and an incremental rebuild reported no work. The baseline
is `1ccc53419acd4c22d9496178a65fffd83f26d6b1` from the bounded-XC experiment.
Archive and PR commits are not relabeled as measured source.

With the charge-split override unset, SM 12.0 source classes containing at
least 8,192 orbital primitives request 32 partitions, capped by the existing
dead orbital-potential allocation. Smaller work, other devices and insufficient
capacity retain one partition. Explicit 1..64 overrides retain their semantics;
1 is the rollback. There is no new allocation, integral omission, screening,
metric, grid, basis, precision or convergence change. The kernel and ordered
partial reducer are unchanged; different summation grouping is independently
qualified below.

## Complete paired endpoints

Five alternating process pairs measure constructor/preparation, SCF, analytic
forces and host export in cold, followed by three own-density warm calls.
Both arms enable the separate optional physical-validation GEMM feature from
PR #2718; all other controls retain their defaults. These results do not measure
GEMM-disabled defaults. The Slurm-owned RTX 5090, thread controls, scientific
settings and iteration/Fock work match within pairs. Warm statistics use one
median per process.

| Standard water96 DF | Baseline cold (s) | Automatic cold (s) | Gain | Baseline warm (s) | Automatic warm (s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE | 17.890817 | 18.052194 | -0.902% | 6.087588 | 6.094628 | -0.116% |
| PBE0 | 29.755531 | 29.763044 | -0.025% | 9.513947 | 9.514470 | -0.006% |
| r2SCAN-3c | 48.213015 | 39.140704 | 18.817% | 13.893717 | 11.189122 | 19.466% |

Only r2SCAN-3c passes the gain gate
`max(0.02, 2*(baseline_MAD/baseline_median + candidate_MAD/candidate_median))`,
which is 2% for all six comparisons. No PBE/PBE0 speedup is claimed.

All **242 measured complete calls in 83 records** pass the existing independent
energy and full-force gates: 24 screening calls, 120 paired calls, 56 seven-method
Standard/Fast Direct/DF cold/warm calls, 32 PBE/r2SCAN-3c original/moved calls,
six larger-size calls and four profiler-instrumented calls. Every method is
retained, including slow omegaB97M-V and all force components. Direct matrix
coverage is numerical compatibility evidence, not a Direct speedup claim.

The larger repository fixture has 108 atoms. r2SCAN-3c DF cold/warm takes
65.472674/20.175591 seconds at baseline and 54.687337/17.784416 seconds with
automatic partitions. Stock GPU4PySCF takes 31.589337/15.078895 seconds.
These are single observations, not repeated performance verdicts. A separate
independent direct oracle checks grid levels 7/8 for stability; its two solves
are additional reference work, never production seeds. The retained reuse
manifest binds the new reference and unchanged smaller CPU/GPU spot controls.
The candidate has not reached all-method GPU4 parity.

## Scientific, ownership and execution checks

The full 188-case MD arithmetic suite passes. Ten long-source explicit/automatic
partition cases pass independent libcint charge/J and finite-difference
derivative gates. Graph replay, zero/changed density and constrained owner
capacity remain covered. Resource and native endpoint checks pass, including
94 resource tests, 89 XC tests, prepared/unprepared independent energy/forces,
and native sanitizer checks. The two automatic cases pass memcheck and racecheck
with zero errors/hazards; the complete native suite also passes synccheck.

Original failed attempts remain in provenance: the first fixture exceeded the
probe's old 16-primitive input limit, then a racecheck run exceeded the test's
90-second child timeout. A direct retry had zero hazards but a diagnostic
byte-equality comparison rejected a maximum difference of 1.29e-14. The final
harness changes only the child timeout to 600 seconds and passes both original
scientific tests, including every input and derivative tolerance. No production
kernel or scientific acceptance gate was changed to satisfy these retries.

The Nsight cold-plus-warm totals identify remaining density products, diagonal
potentials, geometry and DF derivative work; Direct also retains expensive
order-5/order-6 force work. Kernel totals are diagnostic and are not exclusively
cold time. All four profiled endpoints independently qualify.

Real-device work used finite Slurm allocations on node1 and preserved assigned
device visibility. Verified sccache 0.16.0, before/after cache statistics, build
recipes, package hashes, source verification, allocations, logs and failures
are retained. This cluster disables Slurm accounting: the failed `sacct`
attempt and its error are explicit, alongside original `scontrol` and `srun`
evidence.

## Offline verification and lossless storage

From the repository root:

```sh
PYTHONPATH=python:. python benchmarks/results/df-charge-partitions-20261024/reproduce.py
```

The verifier authenticates every original endpoint JSON text and support file,
reconstructs byte-identical source manifests from the retained bounded-XC base
plus explicit deltas, rechecks all 242 complete calls against independent
references, and recomputes paired statistics, settings and semantic work gates.
Profiler exports and original run scripts are retained in provenance; large
Nsight binary traces remain in ignored local/remote artifacts.

Six older root-level RTX 5090 JSON payloads are compressed with deterministic
gzip, recovering 320,789 bytes under the unchanged evidence caps. Each expanded
file equals its original Git blob, pinned by `history-storage.json`. To restore
their original filenames to a new ignored directory:

```sh
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/df-charge-partitions-20261024/history-storage.json \
  --output .artifacts/df-charge-history
```
