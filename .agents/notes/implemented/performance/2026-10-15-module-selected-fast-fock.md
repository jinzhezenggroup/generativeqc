# Decision: select Fast Fock arithmetic by shared operator capability

Status: implemented
Date: 2026-10-15

The upper AO/atom-count admission limits below are superseded by
[the size-cap removal decision](2026-10-19-fast-fock-size-cap-removal.md).
The module-selected arithmetic and other scientific exclusions remain unchanged.

## Problem

The initial budget-Fock implementation unnecessarily required canonical PBE0 in
both Python and native policy, even though the executable Direct recurrence,
resident values, contraction and final-state algebra are independent of XC.
Requiring mixed K evidence also excluded pure semilocal graphs, and grouping
full-range/range K in one census row could mislabel strict range work as mixed.

## Decision

Use the shared `scf::cuda_fock_precision_capabilities` contract. It inspects
backend, provider approximation, operator range, derivative order and spin,
never a functional name, XC registration, canonical fraction or MethodIR hash.
The existing native owner continues to validate the complete requested model.
Its basis/resource domain remains 24--768 AOs, 2--96 atoms, through d shells,
restricted and all-electron. Unsupported providers remain strict.

All executable native XC compositions inherit the same J/full-range-K policy.
Both fused and independent CUDA schedules consume the shared Direct modules;
the resolver selects independent scheduling for semilocal graphs and general
hybrid coefficients. A fused-only check would incorrectly reject even PBE0.
No particular grid is required by recurrence precision; the selected quadrature
and XC program are unchanged. XC, tau, separate range exchange and nonlocal
correlation retain strict arithmetic. Their presence does not reject lowering
an independently supported Direct operator. Corrections keep their existing
complete-model/derivative contract rather than selecting J/K precision.

Python requires the new v2 private capability before selecting relaxed controls;
v1-only libraries remain strict instead of being assumed to implement the new
domain. Explicitly tighter caller tolerances and observable targets retain their
meaning. Fresh physical FP64 Focks, canonical export and the existing trace,
idempotency/stationarity validation remain mandatory. Unsupported arithmetic
and numerical failure retain bounded fallback, not a hidden CPU oracle.

Descriptor admission no longer rejects AUTO solely because XC comes from a
generated split registration. Direct consumers share AUTO; fitted consumers
retain their independent strict contract. Complete-model composition checks
remain enforced. All force compositions now obtain their contract through
`StationaryDerivativeContract.for_native_force`: the composite range/nonlocal
consumer must honor the same immutable, authenticated native allowance as the
ordinary consumer. Numerical array checks, live native proof and the maximum
`1e-4` stationarity allowance are unchanged; response contracts stay strict.

Full-range and separate range K receive independent precision census records.
Pure semilocal graphs require positive mixed J/recurrence evidence but no K.
Hybrid evidence follows the existing native full-range projection, including
the identity converting SR/LR terms into a full-range term plus range correction;
the strict correction cannot prove that mixed K ran. This reuses the scientific
projection rather than implementing another exchange decomposition in policy.

## Rejected alternatives

- Copying the PBE0 allowance checks into a list of functional-specific branches:
  new functionals sharing the same modules should inherit the policy.
- Forcing every operator to FP32 merely because Fast is selected: range/VV10,
  XC and other unsupported consumers must report the arithmetic they execute.
- Removing physical closure or lowering determinant/canonicality checks to make
  a new composition pass: operator capabilities do not certify arbitrary final
  densities or per-geometry energy/force errors.

## Evidence

Host tests exercise shared capability composition, arbitrary exchange weights,
pure-J models, independent strict range accounting, tighter caller controls and
legacy-library fallback. Scheduled complete-endpoint tests compare LDA, GGA,
global hybrids, tau meta-GGA, generated split hybrids and range/nonlocal graphs
against independent moving-grid GPU4PySCF references using unchanged observable
gates. Source-matched raw results are retained separately from earlier PBE0
measurements; no old PBE0 speedup is generalized to every functional.

The old force-predicate text-mutation test is replaced by executable capability
boundary cases across both curated and generated compositions, including
renaming. No production-body fingerprint is refreshed to certify this change;
the historical capacity audit is not new numerical qualification. Initial GPU
runs that exposed the fused-only check, descriptor refusal and composite
allowance mismatch are retained as failures, not accepted speed measurements.

### Source-matched qualification

Slurm job 1598 on node5, `main`, one RTX 5090, UUID
`GPU-c17495e9-988d-0626-8bdd-95d0d1b47f36`, driver 570.86.16: **12 passed**.
Eight complete Fast water/def2-SVP spherical endpoints use independent
GPU4PySCF moving-grid energies and forces with the same `GridSpec(48, 16, 32)`.
Every endpoint has one solve, positive mixed J/recurrence, actual mixed K when
present, two final strict Focks and zero execution retries. Range K remains
strict. The four additional generated-hybrid ordinary AUTO tests cover both
RKS and UKS, live geometry reuse and two reconverged finite-difference steps.

| XC composition | Mixed + strict Focks | Energy error (Eh) | Max force error (Eh/bohr) | Force RMS (Eh/bohr) |
| --- | --- | ---: | ---: | ---: |
| LDA | 10 + 2 | 6.636469e-12 | 9.264072e-7 | 4.900786e-7 |
| PBE | 11 + 2 | 6.536993e-13 | 1.833849e-7 | 9.162970e-8 |
| PBE0 | 9 + 2 | 4.806111e-11 | 2.047088e-6 | 1.115835e-6 |
| B3LYP | 10 + 2 | 3.979039e-13 | 1.946185e-7 | 1.138641e-7 |
| r2SCAN | 9 + 2 | 8.341772e-12 | 9.807836e-7 | 5.339751e-7 |
| M06-2X | 9 + 2 | 5.721290e-11 | 2.655934e-6 | 1.348588e-6 |
| MN15 | 10 + 2 | 8.668621e-13 | 3.292768e-8 | 1.798557e-8 |
| WB97M-V | 9 + 2 | 1.400480e-10 | 5.035715e-6 | 2.344880e-6 |

Unchanged independent gates are `1e-5` Eh, `1e-4` Eh/bohr maximum force and
`3e-5` Eh/bohr RMS. These are observed qualification errors, not universal
per-molecule error bounds. Timings include cold JIT for unpackaged profiles and
are not a steady-state speedup comparison. The earlier 96-atom PBE0 campaign
remains historical; no new 96-atom or all-functional speedup is claimed.

Finite Slurm job 1599, same node/library, runs complete Fast PBE and B3LYP
energy/force endpoints under Compute Sanitizer memcheck: **2 passed, 0 errors**.
The sanitizer reruns retain the same independent oracle gates and real mixed
operator evidence; they are not added to the 12-test qualification count.

The current-library host selection passes **580 tests**, with 59 resource skips;
the separate strict-response/final-closure selection passes **67 tests**.
Analytic final-state and KS-final-state host executables pass. The actual
descriptor probe, native API syntax, Ruff, changed-line clang-format, ownership
inventory (362 files), API/runtime boundaries and compiler structure pass.
Full PR CI is not claimed green; historical source-contract failures are not
hidden by replacing production-body hashes.

Qualified main library SHA256:
`249ae396f2df8d13910a424a29a5b6d8037cffd85d7ec7f3646736de91e4fc59`.
Native source/build identity:
`20f96c726f48f31ab93fedb820e7be6371fa95a289262731f0440f3f19593eb5`.
Frozen input archive SHA256:
`dc0a9f6f26e6048a94f914cfdfd142aca90dd67edc613c5e6a8661c2e68c4007`;
all 11,013 deployed inputs match their per-file hashes. Later documentation
edits clarify CUDA scope and record results without changing runtime inputs.
Release, CUDA 12.9.1/sm120, ordinary compilation and ccache 4.5.1 are retained.
References: PySCF 2.14.0, Libxc 7.0.0, GPU4PySCF 1.8.1 and CuPy 13.6.0.
All real GPU work uses finite Slurm allocations with assigned visibility.
Node n3 is not used; n2's driver/library mismatch was recorded and avoided.

Raw logs, JUnit receipts, endpoint error JSON, source hashes and failed/cancelled
attempts are retained in the ignored sibling evidence directory
`../evidence-module-fast-fock-20261015`, not published as external assets.

## Revisit when

Additional providers/operators acquire independently validated lower-precision
consumers. Extend their module capabilities and shared scheduling rather than
adding method-name branches. Expanded spin/basis/resource domains require their
own executable capacity and independent numerical evidence.

## References

- [Earlier budget refinement](2026-10-15-pbe0-budget-refinement.md)
- `src/scf/fock_precision.hpp`
- `src/dft/cuda_ks_precision.hpp`
- `docs/user/application_precision.md`
- `tests/python/test_application_fock_cuda.py`
