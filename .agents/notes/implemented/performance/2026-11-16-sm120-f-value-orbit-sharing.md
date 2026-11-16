# Decision: share full-g f-value scalar orbits only on sm120

Status: implemented
Date: 2026-11-16

## Problem

The native full-g one-electron value owner repeats scalar f-component algebra
already related by coordinate permutations and Gaussian-center exchange. The
existing g lowering provides an independently checked scalar-IR implementation
of these transformations. Extending that implementation is preferable to a
second algebra, but smaller generated source alone does not establish a better
multi-architecture compilation or runtime artifact.

This qualifies a narrower alternative to the
[earlier broad f-sharing experiment](../../proposed/2026-11-16-f-shell-orbit-compilation-sharing.md).
That experiment's COSX and gradient candidates remain unadopted.

## Decision and invariants

Reuse g-value orbit lowering for f-bearing pairs only in the full-g header and
only under `__CUDA_ARCH__ == 1200`. Keep the original f lowering in the other
preprocessor branch. sm80, sm90, compute80 PTX, unknown future architectures,
standalone through-f headers and separately owned COSX consumers retain their
existing lowering. Guard f maps and their users together. Existing g lowering,
public signatures, angular inventory and all derivative generators are unchanged.

Keep rounded radial/Boys inputs and the existing mathematical IR. Permute
coordinate ownership and exchange Gaussian centers using the already-qualified
scalar transformations. Invalid-component behavior and bounded native launch
contracts remain intact. Do not remove angular, architecture, oracle, numerical,
profile, lifetime or forced-bounded coverage to obtain a smaller build.

## Rejected unconditional alternative

Sharing f values unconditionally in the native full-g owner makes its physical
header smaller, but the complete `80;90-real;120-real` object grows from
**16,889,528 to 20,666,600 bytes**, about **22.36%**. sm80 stack grows from
632/680 to roughly 1,488-1,520 bytes; sm90 grows from 632/672 to roughly
1,480-1,680 bytes. Its smaller PTX dump, 8,212,098 to 5,565,379 bytes, does not
justify these retained-image regressions.

Original/candidate request observations are 560.307/283.800 seconds, each with
one CUBIN hit and seven misses, and overlapping execution. Reaped-worker maximum
RSS is 3,081,908/1,064,836 KiB. These observations are not whole-wheel cold-build
speedups, nor sufficient reasons to accept the older-image resource regression.

Slurm job **17549** nevertheless checks the unconditional candidate's sm120 and
forced compute80-PTX arithmetic: six runs, each with **700** fresh independent
PySCF/Libcint vectors and **2,900** complete weighted S/Hcore/T/V launcher cases.
All pass; maximum independent absolute error is about **1.022e-13**. Keep this
evidence distinct from qualification of the selected architecture guard.

## Complete supported-image comparison

With the sm120 guard, the same complete owner object shrinks from
**16,889,528 to 13,343,416 bytes**, about **21.00%**. Every sm80/sm90 kernel
resource record and SASS section is byte-identical to the original. The complete
compute80 PTX dump is also byte-identical: **8,212,098 bytes**. Every selected
sm120 resource record and SASS section matches the independently exercised
unconditional candidate. No kernel identity is added or removed.

On sm120, registers remain 252/254/254/255. Three stacks remain 328 bytes;
shell-warp stack grows **344 to 368 bytes**. Constant-bank-2 use falls from
5,396 to 4,860 bytes. The physical generated header grows from
**4,341,862 to 4,612,871 bytes**, about **6.24%**, because it retains the original
branch. This is an architecture-selective compiler-work/object reduction, not
a physical-source-size reduction.

The guarded request observes 43.800 seconds, **three CUBIN hits/five misses**,
with every backend image cached. Its 511,988-KiB reaped-worker maximum and short
request are **not cold-build speed or memory improvement evidence**. Preserve
the serial wheel compilation and 12-GiB isolation cap; a worker maximum is not
aggregate container memory. These object-size measurements cover only this
owner, not the final wheel or the remaining Direct-JK compilation.

## Actual native integration and independent endpoints

The actual Release sm120 library/CLI builds and links with portable automatic
XC AOT. Actual generated headers match the reviewed candidate; all four native
kernel resources and SASS sections match its full-matrix sm120 image. The
incremental build observes 390 seconds with 17 CUBIN hits and 69 misses across
35 compiler requests. It is not a fresh supported-matrix wheel build.

Patch-stage library SHA256:
`617c772103ce9c21b198ed4a98e9c8bfaedb6574bfe7e1ab0a62dab89cdd06d4`.
Native source identity:
`05f75f386b6d4a417abe632ae77d9dbab7dd2032ea5983a04d1b1f3162ca917d`.
The isolated baseline is clean force-sharing code `feaddd7aa`, library SHA256
`ff2ea8bffc58a1b94faf47b392f98d2052983c51b1d84bdaf3865dc91b0503d1`.

Slurm job **17568** passes **64** ABBA complete independent PySCF endpoints:
RHF/UHF, Cartesian/spherical, cold/warm/moved/restored, including an f shell and
negative-coefficient contraction. Iteration, Fock-build and precision-work
counts agree. Maximum independent energy/force errors are about
**6.40e-14 / 8.04e-11**; maximum baseline/candidate differences are
**3.56e-15 / 1.50e-15**. The same allocation passes **53** native profile/safety
and forced-bounded high-l tests, with no skips or failures.

Slurm job **17552** independently qualifies the actual guarded full-matrix
weighted owner in four normal ABBA runs and original/candidate forced compute80
PTX runs. Each executes **700** independent vectors and **2,900** complete
S/Hcore/T/V launcher cases, **17,400** total. All pass, with maximum independent
absolute error **1.022e-13**. The case inventory retains all 25 angular pairs
through g, seven geometries, ordinary/diffuse-tight exponents, Cartesian and
mixed-three-term projections, negative two-primitive contractions and three
nuclear centers. The unchanged PTX branch is a portability control, not a PTX
runtime improvement claim.

Diagnostic execute-time median/maximum ratios are **0.9979 / 1.2777**. Retain the
largest outlier: moved RHF spherical **91.071 to 116.362 ms**. This is not a
uniform runtime improvement claim. Complete weighted-owner qualification,
public HF acceptance and installed-wheel qualification remain different scopes.

## Host gates and audit boundaries

The isolated actual-branch host oracle suite passes **938 tests**, covering raw,
normalized, contracted and spherical arithmetic against PySCF/Libcint. The host
fixtures exercise original and sm120 source branches explicitly; defining a
device macro for host arithmetic is not real-GPU acceptance. Through-f/COSX
retains its separate fixture. Diffuse/tight controls include both f and g.

The production host/compiler/CMake/catalog/CI run passes **1,977 tests** with
eight opt-in skips and one configure failure: this shell did not expose its
existing Conda OpenSSL/LibArchive installation to CMake. With the actual
`OPENSSL_ROOT_DIR` and `LibArchive_ROOT` supplied, and `GENERATIVEQC_NVCC` set to
the verified CUDA 12.9 compiler, the failed test and all eight skipped tests are
included in a passing **19-test** qualification run. This includes all seven
actual native CUDA graph configurations and the real multiarch object/link
contract. No production code, guard or test is disabled to resolve the local
prerequisite discovery failure. The changed-file full pre-commit run passes.

Compiler structure checks **605 modules** with no dependency errors. Complete
configured type checking passes with the previously qualified **ty 0.0.82**.
The shell's older default ty 0.0.15 reports 171 diagnostics, byte-identical to
the unchanged compiler in an equivalent temporary checkout; no unrelated typing
workaround is introduced. Authoritative audit writers update only current
source-identity fields. All scientific, historical, archival and acceptance
fields are unchanged.

## Follow-up: provenance-only stationary recompilation

This one value-generator edit also rebuilds 16 native stationary CUDA wrappers.
An isolated original/candidate PBE0 RKS emission has identical 128,914-byte
lengths and differs only in its binding's scientific source-identity literal.
The contract intentionally hashes complete compiler source families. This is
real provenance, not a spurious dependency that can simply be deleted.

A future ownership change could separate changing host identity metadata from
unchanged heavy CUDA implementation while retaining the complete identity and
runtime binding checks. Do not narrow source inventories, override identities,
or weaken integrity validation merely to obtain cache hits. This change does
not implement or claim that additional optimization.

## Evidence and revisiting

Retained local receipts: `/tmp/gqc-wheel-native-f-values/` for the unconditional
supported-matrix and weighted-owner experiment, and
`/tmp/gqc-wheel-native-f-values-sm120/` for the guarded matrix, dependency dumps,
cache statistics, integration comparison, independent oracle inputs, GPU reports,
host gates, unchanged type diagnostics and stationary-output diff. The original
all-consumer experiment remains under `/tmp/gqc-wheel-f-orbits/`.

Revisit other architectures or consumers only after complete image/resource
review and independent complete endpoint qualification. Smaller source, PTX,
one kernel or a cached request is not enough to broaden this schedule.
