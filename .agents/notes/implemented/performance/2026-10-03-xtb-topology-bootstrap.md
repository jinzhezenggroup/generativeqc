# Decision: make CUDA topology bootstrap structural

Status: implemented
Date: 2026-10-03

## Problem

CUDA preparation evaluated an artificial geometry on the CPU: coordination,
overlap, multipole S/D/Q, H0, ES2 and AES2 geometry caches, and external-point
potential. None of these seven evaluator/cache calls supplied the eventual
caller geometry. Setup then rebuilt geometry/AES2 on CUDA, temporarily forged
SCC convergence, executed a synthetic energy/force smoke, downloaded its outputs
and errors, and restored the fresh SCC initializer. The real first request
subsequently refreshed every geometry-dependent numerical value on CUDA.

This repeated scientific work was particularly expensive for large systems and
made production CUDA preparation depend on CPU evaluators without a scientific
consumer. A synthetic successful calculation cannot validate the caller's
as-yet unconsumed numerical inputs.

## Decision and invariants

Native preparation still builds all topology/parameter plans, descriptor bindings,
stable storage, and the fresh SAD multipoles. Initial numerical images are zero,
except for identity overlap. The seven CPU calls and their unused workspaces,
synthetic GPU geometry/AES2 setup, and hidden energy/force calculation are removed.
The runtime does not replace any compiler-owned mathematics with handwritten
arithmetic; it removes calculations with no production consumer.

The identity-overlap factorization and its asynchronous provider diagnostics
remain. Setup invalidates the placeholder factor's generation before publishing
the candidate. The externally visible numerical epoch starts at zero. The first
real request must refresh all scientific inputs, refactor the real overlap and
commit epoch one before SCC or energy/force publication. The retained factor is
therefore never accepted as the caller's overlap. No failed request may publish
placeholder science. Fresh SCC, FP64, numerical tolerances, transactional
replacement, provider validation and the bounded SCC Graph fallback are unchanged.

This supersedes the setup-smoke boundary retained by
[electronic matrix tiling](2026-10-03-xtb-electronic-matrix-tiling.md).

## Qualification

`tests/native/test_gfn2_cuda_bootstrap.cpp` exercises the real private CUDA
transaction with all seven independent tblite fixtures. It checks first, repeated
and topology-changing requests, host and device coordinate ingress, energy-only
to force selection, and matching fresh SCC iteration counts. It injects NaN,
finite coordinate overflow, coincident atoms and invalid atomic numbers on the
first request and during topology replacement, then checks recovery to prior and
new valid topology. A first-call iteration limit also checks terminal SCC failure
and recovery. Admission/refresh rejection preserves energy, force and diagnostic
sentinels; terminal SCC failure publishes diagnostics and NaN science.

Every healthy result uses independent 5e-7 Eh and 5e-7 Eh/bohr tblite gates.
The opt-in Python harness requires Slurm and compiles through ccache; its actual
entry point passes on n2. Host source guards exclude CPU evaluators and hidden
energy/force execution from this CUDA preparation owner.

The final source/provenance build also passes all 43 public/native-bootstrap
Python tests on n2. There are 22 passing host provenance/boundary/bootstrap tests
and one expected opt-in skip (exercised separately above); all pre-commit checks
pass. CMake and standalone harness compilation use ccache.

Both n1 RTX 5090 and n2 RTX PRO 6000 pass 41 public lifecycle/oracle/force tests and
all 110 comparator samples: seven fixtures plus water8/32/64 (24/96/192 atoms),
one cold, five repeated and five changed-geometry requests per case. Settings are
FP64, energy plus forces, fresh SCC, 300 K, Broyden 8/0.4, maximum 300 iterations,
and energy/charge tolerances 1e-10/1e-8. Every sample passes accuracy and SCC-count
gates. Energy is identical to the parent; maximum force change is 4.163e-17
Eh/bohr. Against xTBloom, maximum errors are 5.684e-14 Eh and 5.17e-15 Eh/bohr.

## Complete endpoint evidence and limits

Construction plus first call, milliseconds (one cold sample per case):

| GPU / case | Parent | Candidate | xTBloom |
| --- | ---: | ---: | ---: |
| n1 / water64 | 397.138 | 358.875 | 394.115 |
| n2 / water64 | 393.821 | 357.727 | 392.489 |
| n1 / water32 | 239.209 | 239.125 | 245.010 |
| n2 / water32 | 242.708 | 231.747 | 244.801 |
| n1 / water8 | 49.773 | 50.099 | 52.071 |
| n2 / water8 | 48.907 | 46.956 | 52.129 |
| n1 / H2O | 16.427 | 16.071 | 15.181 |
| n2 / H2O | 15.438 | 14.785 | 14.745 |
| n1 / HF | 13.844 | 13.614 | 12.358 |
| n2 / HF | 12.651 | 12.251 | 11.899 |
| n1 / first-process H2 | 546.478 | 801.242 | 522.325 |
| n2 / first-process H2 | 405.681 | 408.134 | 319.788 |

All ten first-singlepoint, repeated and changed medians beat xTBloom on each
GPU, but combined cold totals win only 4/10 on n1 and 6/10 on n2. The 801 ms n1
first-process sample is retained: initialization remains variable and sometimes
regresses. Cold means a new calculator in the same measurement process, not a
new process per case. Moving work between constructor and first call cannot
establish a cold-start win; compare their sum.

Warm performance is essentially unchanged versus the parent, including timing
noise/regressions. Water64 medians are 310.324 -> 310.290 ms on n1 (xTBloom
317.009 ms) and 309.793 -> 311.018 ms on n2 (xTBloom 318.594 ms). The improvement
targets topology preparation, not a warm scientific kernel.

A production-route Nsight probe of three 96-atom calls confirms the semantic
work change: H0/Pulay contraction invocations fall from four to three (median
30.352 -> 30.271 microseconds); integral force preflight likewise falls from four
to three (820.549 -> 821.365 microseconds). This verifies elimination of the
synthetic force call. Some device-launched SCC kernels are absent from Nsight's
summary, so this is not a complete count of SCC work. Per real request, all
scientific pair/AO evaluations and SCC iterations remain unchanged.

Ignored receipts in the task checkout:

- `.artifacts/{n1,n2}/topology-bootstrap/`: raw parent, candidate and xTBloom
  outputs, binary identities and full comparator reports.
- `.artifacts/bootstrap-qualification/`: native harness, independent oracle,
  sanitizer and pytest logs.
- `.artifacts/n2/bootstrap-{before,after}_cuda_gpu_kern_sum.csv`: production
  route invocation counts and kernel timings.
- `.artifacts/libgenerativeqc-bootstrap-qualified.so`, SHA-256
  `f8b4e59b6e7ed3fd0588eac8cce042e74795a28a557b1a915e6c36508d2cbc1f`.
- Final source/provenance build after comment wrapping and equivalent explicit
  braces, separately qualified in `.artifacts/bootstrap-qualification/n2-final.log`,
  SHA-256 `affe3faf3beb45bcd341e108eb3f58654bc14b5ed2188fe7616535d04c9556f1`.
- Parent `c3ff9636a` library SHA-256
  `df3b7f9ed857219c62e0f9b8c6bd1eb23c0cfdc96da2d0f381aae1ffa08f5252`.
- xTBloom `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, library SHA-256
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Rejected alternatives and revisit conditions

Retaining a synthetic molecule and optimizing its CPU evaluators would retain
work without a consumer. Reusing synthetic SCC convergence or overlap factors
would violate fresh-start and real-geometry contracts. Removing the identity
factorization as well is deferred: provider workspace/binding preparation and
its failure boundary need separate investigation and qualification.

Revisit the bootstrap if a new public consumer genuinely needs initial
numerical values, or if provider setup can be made structural without weakening
first-request validation. Remaining constructor and first-process gaps require
separate bounded lifetime/loader work and complete cold timing.

## Historical cold-timing qualifications

The later [PR #1739 comparator audit](https://github.com/jinzhezenggroup/generativeqc/pull/1739)
found that the old constructor timer also destroyed the preceding calculator.
The historical multi-case cold totals above therefore include preceding cleanup;
they are not isolated construction plus first-call measurements. The subsequent
comparator records result/calculator cleanup separately. The original rows are
retained unchanged, with no estimated cleanup subtraction. First-singlepoint,
warm/changed timings and scientific gates retain their original scope.

The later [PR #1742 lazy-loading correction](https://github.com/jinzhezenggroup/generativeqc/pull/1742)
found that xTBloom library identity verification triggered its lazy native/provider
load outside both timed phases. Historical first-process comparisons above thus
undercount reference loading. The corrected v3 comparator includes lazy loading
and keeps cleanup separate; its timings begin after Python package import, not
OS process launch. Warm/changed timings and numerical gates are unaffected. The
corrected six-run results in that later PR qualify a later stack state, not an
isolated remeasurement of this bootstrap change, and do not establish universal
startup superiority.
