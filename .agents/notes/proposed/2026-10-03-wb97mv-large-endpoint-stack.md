# Proposal: qualify the WB97M-V force optimizations together on current master

Status: experimental integration; complete endpoint advantage not established
Date: 2026-10-03

## Scope and provenance

The untouched baseline is master
`a20b8801f7f43b83cd7da8001db180716d8df302`, independently built before composition.
Its library SHA-256 is
`ac9fe852d6c88e3b2d04fb56ccbca523b4fdb4bb3eece13f04997e18291b1e3f`, and the
embedded source identity independently matches
`f46f35ea7d273ffe1b534e3e93082c28c6ef6508a0a89e6702575f9c75fb4046`.
All 442 compiler commands use verified ccache launchers. Source archives and
before/after cache receipts remain in `.artifacts/wb97m-large-stack/`.

This review integration combines four independently tracked changes:

- #1752 (`5bb9b86c0`): lifetime-based reuse of private VV10 force buffers,
  lowering the full-grid I/O arena from 22N to 15N doubles.
- #1755 (`63e762b46`, note head `0adfd34b7`): requested-source force-product
  screening and K-only selection for the bounded range-exchange consumer.
- #1756 (`ac8e14728`, note head `7549684d2`): compiler-weighted LR force roots
  for angular orders 0--3.
- #1757 (`d5ee2a857`): compiler all-center LR moment reuse for orders 4--6.

The parent experiments retain their original master/source/binary identities.
Their individual speedups cannot be added or attributed to this integration.
In particular, screening changes which tasks reach the new low/high-order
consumers, and the private storage reduction may change the geometry tile
selected by the existing resource planner. Both interactions require complete
endpoint qualification at a larger size.

## Scientific and resource boundaries

No new numerical threshold, density/weight mask, precision mode, basis
normalization, independent source meaning or production oracle is introduced.
The existing SR/higher-order and resource fallbacks remain. The source-screening
selection is not a strict subset of the previous task set because its raw-K
linear bound has a different source weight. Actual executed bounded quartet
counts remain unexported; logical dense capacity is not an executed-work count.

Parent numerical gates include independent displaced CPU ERIs, RKS/UKS complete
forces, changed geometry and failure/stale-state isolation. The high-order
parent additionally passes 19,440 host gradient coordinates, native four-center
order-4/5/6 gates, zero memcheck errors and seven complete tests (n1 Slurm 5513).
Those results do not qualify the merged binary. Its own source fingerprint,
numerical/sanitizer gates and endpoint records must be retained separately.

## Acceptance plan

A new same-allocation full-grid 96-atom comparison starts with the untouched
master binary and a fresh full-Fock GPU4PySCF control. It records cold, priming,
three engine-local warm calls and every energy/force observation. The candidate
phase requires an exact-library-hash receipt from completed numerical gates;
missing or mismatched evidence terminates the job while retaining baseline data.

All observations must pass 1e-8 Eh / 1e-7 Eh/Bohr. Reference controls suppress
both dm_last and vhf_last, preserving the independent converged density, strict
SCF tolerances and all-repeat internal consistency gates. Report actual SCF
iterations, available semantic work/allocation counters, cold and warm timings
separately. Changed-geometry and smaller controlled endpoints complement the
larger run. Keep the integration experimental if the complete advantage is not
measured, even when an isolated kernel or component improves.

## Qualification of the composed binary

The frozen production composition is
`cac0727f3c26650161bd511484d7c402394981c1`. Its library SHA-256 is
`08b276c3dbfd7b37ac5efa2f7e8a0c4518933673822b9428f872d48102a0d5d7`, with
independently recomputed source identity
`8b42be023a9db34c66c2968a5ba7ace55ce9521ebe8831bd1c3c05d0d98ae106`.
All 442 compiler commands use ccache. Later note-only commits do not relabel
this source or binary.

n1 Slurm job 5526 completed successfully on RTX 5090:

- Native s/p/d/f SR/LR derivatives match independent displaced CPU ERIs;
  the same native gates pass compute-sanitizer with zero errors.
- All 41 storage-lifetime, capacity and device-facts tests pass, without skips.
- All 22 storage-lifetime tests also pass compute-sanitizer with zero errors.
- All seven selected complete independent-engine force and rebuild/failure
  isolation tests pass, without skips, in 197.35 seconds.

The retained qualification receipt binds the library and native-test executable
to hashes of all five logs. These hashes and the acceptance assertions were
independently checked after copying the evidence from n1. This qualification
releases the candidate phase of the separately running 96-atom job 5521.
Same-allocation full-grid 24-atom job 5528 and changed-geometry job 5527 are
additional controls; their completed results are recorded below. The large
96-atom comparison remains pending.

Master subsequently advanced to `5d68bcc07` through #1742, changing xTBloom
cold-timer accounting and related documentation/tests. It does not change the
WB97M-V production path or the comparator used here. The retained baseline
remains exactly `a20b8801f`; it is not relabeled as the newer master.

## Completed full24 and changed-geometry controls

Job 5528 compares the untouched baseline and frozen composition sequentially in
one RTX 5090 allocation, with 24 atoms / 192 spherical def2-SVP AOs and the full
48 x 16 x 32 grid. Each variant has its own fresh full-Fock reference control.

| Complete endpoint, seconds | Baseline a20 | Composition cac |
| --- | ---: | ---: |
| Native cold | 266.481280 | 256.094901 |
| Native priming | 39.316285 | 27.608468 |
| Native three-repeat warm median | 39.755570 | 27.628118 |
| Reference three-repeat warm median | 27.965236 | 27.953379 |

The composition improves the native warm median by 30.505%. Native cold calls
both take 18 SCF iterations; every native warm call takes one. Reference cold
calls take 14 iterations / 15 J/K builds and every reference warm call takes
one iteration / two J/K builds. All five pairs per variant and all reference
replay consistency gates pass independently: maximum energy error is
2.85e-12 Eh and maximum force error is 4.96e-10 Eh/Bohr, below the unchanged
1e-8 / 1e-7 gates.

Candidate warm samples are 28.678636, 27.628118 and 27.592642 seconds; reference
samples are 27.967601, 27.875132 and 27.953379 seconds. Although the candidate
median is 0.325 seconds lower, one candidate repeat is slower and cold remains
substantially slower. Treat this as small-case parity, not a robust reference
advantage or completion of the large-system objective. No sample is discarded.

The first warm integral-derivative phase falls from 14.987791 to 3.274979
seconds. The combined grid/pair drain remains around 11 seconds. Both variants
visit 589824 AO points and 1179648 geometry points, using 576 tiles of 1024
points and 651165696 partition-pair visits. Warm preparation is reused. The
additional device/host bounds fall by 33030144 bytes each through storage reuse;
executed screened integral and compacted nonlocal pair counts remain unavailable.

Job 5527 separately checks water12/def2-TZVP on the diagnostic 24 x 8 x 16
grid against retained independent CPU energy/force arrays. Every cold, priming,
warm, changed-geometry and fixed-state observation passes for both builds.
Single warm calls are 22.801920 -> 19.034839 seconds and moved-geometry calls
91.506313 -> 88.213000 seconds, with matching 22/1/1/11 SCF iteration counts.
Maximum energy/force errors are 1.48e-12 Eh / 8.97e-10 Eh/Bohr. This is a
changed-geometry accuracy gate, not a full-grid timing claim.

Raw reports, all-repeat verification and Slurm completion records are retained
under `.artifacts/wb97m-large-stack/`. Job 5534 additionally compares the exact
qualified composition against its fresh full-Fock control at 48 atoms; it does
not supply a same-allocation master delta. Jobs 5521 (matched 96) and 5531
(instrumented exact-candidate 96 endpoint) remain active. Profile timings are
diagnostic only and cannot replace clean endpoint observations.
