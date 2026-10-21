# Decision: admit Huckel preparation from simultaneous buffer lifetimes

Status: implemented
Date: 2026-10-21

## Problem

The 1088-AO r2SCAN-3c/96 preparation estimate is 321,801,344 bytes, exceeding
the unchanged 256 MiB default additional-phase cap. Native initialization
therefore reports `budget_skipped` and starts from the core Hamiltonian.
Increasing this cap diagnostically reduces Standard/Fast target iterations
from 22/14 to 16/10, but does not justify silently raising a resource limit.

## Bound and ownership

Replace the conservative 32-target-matrix term by sixteen FP64 matrices in
both native preflight and the Python planner. The default cap, atomic work
allowances, primitive/topology terms and unsupported/budget/core fallbacks
remain unchanged. No molecular seed is borrowed from an oracle.

Huckel assembly and final same-AO admission execute sequentially. The occupied
atomic subspace dimension is at most the target dimension; assembly's largest
eigen step needs at most seven subspace-square matrices. The larger admission
peak consists of the incoming seed, three snapshot/proposal copies, four
metric/spin matrices, two product temporaries that survive the nested eigen
call, and four eigen/validation matrices. Sixteen matrices cover this
fourteen-matrix peak with two matrices of margin. Retained target integrals
belong to the target inventory; the incoming/returned seed is included here.

H-Ar atomic coefficients have at most nine occupied columns per atom. Cached
frames, sparse orbital supports, systems and primitives retain their existing
linear allowances; one serialized atomic integral/DIIS owner retains its
quadratic/quartic allowances. The whole-endpoint planner still sums retained
fleet seeds and charges the largest remaining serialized workspace. This is
a lifetime correction, not a workload-specific AO threshold or an uncharged
memory allowance.

## Allocation and numerical evidence

The native probe tracks simultaneous C++ allocation payloads, including aligned
and array allocation, after target preparation. Blocks retain their observation
epoch through destruction. It excludes the retained target and allocator
headers, matching the documented preliminary-cap scope; it is not process RSS
or a measurement of external BLAS allocator storage.

All 24 retained independent H-Ar/molecular oracle inputs run through both the
legacy validator and validated native CPU eigen provider. Their densities,
occupations and traces remain covered by independent numerical comparisons.
Two additional 1088-AO allocation stress executions use the real r2SCAN-3c
atomic bases with an explicitly synthetic identity target metric; they are
resource tests, not molecular accuracy references.

All 50 allocation observations fit the proposed bound. The large case peaks
at 123,135,496 bytes with the legacy validator and 132,596,744 bytes with the
native provider, below the new 170,282,112-byte estimate. Its default cap stays
268,435,456 bytes. The revised native/Python oracle, preparation, resource and
typed-failure suite passes 167 tests. Pre-commit checks pass.

## Complete endpoints

The candidate and counterfactual use identical source/runtime inputs except
the two native/Python capacity implementations. Both use native Standard/Fast,
default XC/VV10 settings, full moving-grid force response and the unchanged
convergence/precision controls. Each population has one cold construction-to-
force-export call and three warm calls from its own frozen cold density.

| r2SCAN-3c/96 Direct | Baseline | Candidate |
| --- | ---: | ---: |
| Standard target iterations | 22 | 16 |
| Standard cold seconds | 78.617 | 67.100 |
| Standard median warm seconds | 18.894 | 18.900 |
| Fast target iterations | 14 | 10 |
| Fast cold seconds | 52.410 | 47.689 |
| Fast median warm seconds | 18.815 | 18.556 |

Both baseline populations and all eight candidate populations qualify against
the unchanged README independent energy/force gates (1e-5 Eh/atom, maximum
1e-4 and RMS 3e-5 Eh/Bohr), quarter-gate reference-grid stability margins and
CPU/GPU spot checks. The controls include Standard/Fast r2SCAN-3c/48,
PBE0/96 and r2SCAN-3c DF/96. All 40 complete energy/force calls qualify.

DF/96 remains slow: Standard/Fast cold 1261.321/1250.211 seconds, median warm
177.774/176.308 seconds, both sixteen target iterations. The Fast receipt
explicitly reports `strict_standard_fallback` with
`fp32_gradient_not_qualified`: this default-grid/basis case has no admitted
Fast gradient route and uses Standard FP64 controls. Admitting a better
seed does not repair repeated raw three-center generation. These results must
remain visible; this change does not establish universal performance leadership.

Compute Sanitizer memcheck reports zero errors for both newly admitted paths:

- Actual default r2SCAN-3c/96 Fast cold and frozen-warm energy, including the
  1088-AO CUDA seed eigensolver under the unchanged 256 MiB cap.
- r2SCAN-3c/12 Fast cold and frozen-warm energy/forces with an explicit 20 MiB
  cap between the old and new estimates. The new estimate is 20,496,000 bytes;
  the old formula exceeds the cap. Independent maximum force error is
  4.34715e-5 Eh/Bohr for both calls.

## Source and retained evidence

The source base is `66ccb5587bfefeab09ea75c8a113b10bc95a4ce8`, master
`e2f57d609c17ecefd28d01936aca2ed59a7deef6` plus the provider repair and strict-
tail repair. #2664's external merge contains only the provider repair; #2665
subsequently recovered the strict-tail delta on master. The resulting master
`ca48bf5b508dc0a4fc2dc3bbdfd1c5e08bf1b4d7` has exactly the same tree as this
measured base. #2667 independently submitted the same strict-tail delta before
that recovery became visible. Rebasing this capacity-only change onto that
master does not change the measured implementation; the original source and
binary provenance above remain authoritative.

The candidate patch SHA-256 is
`5569032c7d842cc5a5a2c545649adb82cbb32e6629a30f9b8d146e65a4be8f38`.
Candidate core SHA-256:
`5e647a73b8945be0a86881170ec1929b6ce845a4d02f119dbfb7bebe66f05a6c`.
Counterfactual core SHA-256:
`2a50a84ee4d3992d3ef6205a485b186329e372414e6eb6862098443b3b998820`.
Both manifests verify 12,143 input files against frozen build/runtime copies.
Release compilation uses verified sccache 0.16.0 with retained before/after
statistics. All real-GPU work runs in finite Slurm allocations on n1/node1.

Ignored local `.artifacts/huckel-capacity-audit-v1/`,
`.artifacts/huckel-capacity-v1/`, and `.artifacts/huckel-baseline-v1/` retain
probe sources/inputs, source manifests, binaries, launchers, receipts, failed
attempts, numerical qualification and test logs. The matching remote paths
are under `n1:/data/jzzeng/scf-performance-20261021/`. The allocation audit
uses the original strict-tail binary and proposed bound independently of the
new production preflight.

## Rejected shortcuts and remaining work

Do not raise the default memory cap to hide a lifetime accounting error. Keep
all failed/incomplete diagnostic receipts and explicit bounded fallbacks.

A separate unpromoted residual experiment under
`.artifacts/stationarity-alignment-v1/` and `-v2/` is not part of this change.
Relaxing ordinary strict RKS from 1e-9 to 1e-8 lets PBE and r2SCAN fail the
unchanged final canonical-density check, despite four other methods qualifying.
Their maximum commutators pass at 7.28350e-9 and 8.07519e-9, but maximum density
reconstruction errors are 1.15914e-8 and 1.02152e-8, exceeding 1e-8. A future
stopping/closure improvement must handle that actual acceptance condition;
it cannot simply weaken the residual ceiling or final validator.

Revisit the capacity estimate if molecular assembly/admission overlap changes,
new retained eigen workspaces appear, or atomic resource ownership changes.
This correction leaves convergence and precision policies unchanged.
