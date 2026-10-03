# Decision: distribute H0/Pulay force shell pairs across CUDA blocks

Status: implemented
Date: 2026-10-03

## Problem

After AES2 peer scheduling and fresh device-fact queries, large complete GFN2
energy/force endpoints only narrowly beat xTBloom in repeated calculations.
The H0/Pulay force consumer assigned an entire system to one 128-thread block.
Its ordered shell pairs own disjoint AO blocks, so this unnecessarily serialized
independent work on one SM. A production-route Nsight probe of 96 atoms measured
the contraction at 1.146 ms median, with four calls across three public endpoints
(including initial setup). Device-launched SCC kernels are not all visible in
this trace; its timings do not represent the complete SCC profile.

## Decision

`method/gfn2_h0_force_schedule.py` selects at most 256 blocks of 128 threads per
system. The width is the ceiling of the rounded-up mean ordered shell-pair count
divided by 128, capped at 256. Host selection uses quotient/remainder arithmetic
to avoid overflow and does not download ragged metadata. Each system strides
over its actual device extent. Small means retain a single block; highly skewed
batches and systems beyond the cap remain complete through grid strides.

The compiler owns this scheduling choice. Native code owns traversal, descriptor
admission, one validation/seed scan per system, atomics and whole-system
publication. Every ordered shell pair still evaluates the same generated scalar
primal/VJP and traverses its AO block in the same order. No extra buffer, kernel
launch, host synchronization, D2H metadata, CPU oracle or prior-SCC state is added.

## Numerical and work invariants

- A system with S shells and M orbitals performs S² pair factor/adjoint
  evaluations and M² AO updates. Tiling does not repeat a source evaluation.
- Each ordered pair owns its AO outputs, which remain bitwise equal to the
  single-block route. The generated scalar helper bodies and equation hashes
  are byte-identical to the parent generator's output.
- CN and Cartesian adjoints retain the existing FP64 atomics. Their inter-pair
  order was already unspecified; extra blocks can change roundoff. Independent
  long-double analytic references and the single-block route use the gate
  `256 * epsilon(double) * max(1, abs(reference))` for the native fixture.
- Public molecular gates remain 5e-7 Eh and 5e-7 Eh/bohr against independent
  tblite/xTBloom references. All samples and SCC counts participate.
- Input failure, arithmetic failure, disabled requests, failed SCC status and
  a closed sequence preserve all public accumulators of the affected system.
  Other systems in the same launch still complete normally.

## Evidence

Native harness `tests/native/test_gfn2_h0_force_schedule.cu` uses s/p/d AO blocks,
multiple shells on the same atom, ragged batches and shell counts 11/12 around
the single-block boundary, 180/181/182 around the cap, and a 365-shell outlier
requiring grid strides. Padding with independent one-shell systems selects the
retained route without changing any original input. Each healthy member also
passes an independent analytic host reference that uses no generated helpers.
Both ordinary launch and two Graph replays cover late input/parameter failures,
invalid shell metadata and positions, overflow, coincident atoms, overflowing
coordinate differences, invalid seeds/masks, inactivity, failed SCC and sticky
sequence errors. Host selector tests cover exact tile tails and INT64_MAX.

The 15 host codegen/policy/provenance tests pass, with the opt-in device test
exercised separately through the Python test entry point on n2 (two passes).
That private environment required copied ccache 4.5.1 and `NVCC_CCBIN` selecting
its installed g++-11; the default gcc-12 lacked cc1plus. These environment setup
failures did not run uncached builds. Compiler ownership and pre-commit checks
pass. CMake compilation and isolated native harness compilation both use ccache.

The native harness passes on n2 RTX PRO 6000. On n1 RTX 5090, Compute Sanitizer
memcheck reports zero errors and racecheck zero errors/warnings. All execution
uses finite Slurm jobs on `main` with the appropriate typed GPU request.

Both GPUs pass 41 public runtime/lifecycle/tblite/force tests and all 110 samples
in the complete endpoint comparator (seven molecular fixtures plus water8/32/64,
one cold + five repeated + five changed calls each). These are FP64, fresh SCC,
energy plus forces, 300 K, Broyden 8/0.4, 300 maximum iterations and convergence
tolerances 1e-10/1e-8. All SCC iteration counts match. Energy is identical to the
parent and the largest force change is 4.86e-17 Eh/bohr. Against xTBloom, maximum
errors are 5.69e-14 Eh and 5.17e-15 Eh/bohr.

Repeated endpoint medians, milliseconds:

| GPU / atoms | Parent | Candidate | xTBloom |
| --- | ---: | ---: | ---: |
| n1 RTX 5090 / 96 | 124.619 | 124.374 | 129.293 |
| n1 RTX 5090 / 192 | 313.782 | 309.415 | 315.600 |
| n2 RTX PRO 6000 / 96 | 126.187 | 125.493 | 131.237 |
| n2 RTX PRO 6000 / 192 | 315.428 | 310.833 | 318.524 |

All ten repeated and changed-geometry medians beat xTBloom on both GPUs.
Small endpoints are essentially unchanged versus the parent (including minor
timing regressions); this schedule targets large force consumers. The n2
96-atom contraction median falls from 1.146 ms to 0.0304 ms with the same four
invocations. Endpoint gains are much smaller than this isolated kernel ratio.

Cold results are mixed. On n2, 192-atom construction plus first call improves
404.192 -> 394.556 ms but still trails xTBloom's 392.428 ms. On n1 it improves
404.056 -> 401.524 ms versus xTBloom's 392.287 ms; 96-atom cold total regresses
239.718 -> 247.102 ms in this single cold sample. Only 2/10 (n1) and 3/10 (n2)
combined cold totals beat xTBloom. This is not evidence of universal startup
superiority. Cold denotes a new calculator within the measurement process,
not a new process for each molecule.

Historical cold-timing qualification from the subsequent #1739 comparator audit:
the multi-case protocol used here included destruction of the preceding
calculator inside the next construction timer. The n1 96-atom
239.718 -> 247.102 ms observation (+3.08%) is therefore inconclusive as an
isolated current-molecule cold regression. It is not established as noise or a
repeatable regression, and #1739 does not show that it was fixed or disproved:
its corrected comparison starts at #1735, which already contains this H0-force
change. The historical rows above retain their original protocol; no estimated
cleanup is subtracted and they are not corrected measurements. First-process
and first-singlepoint evidence, warm/changed timings, numerical accuracy, SCC
counts and work counts retain their original scope. See the
[PR #1733 review](https://github.com/jinzhezenggroup/generativeqc/pull/1733#pullrequestreview-5396967265).

Preserved ignored receipts in the task checkout:

- `.artifacts/{n1,n2}/h0-force-tiles/`: raw parent/candidate/xTBloom JSON and
  comparator reports (including constructor plus first call).
- `.artifacts/h0-force-qualification/`: native oracle, sanitizer and compiler
  cache receipts, generated header and parent scalar-output comparison.
- `.artifacts/n2/h0-force-{before,after}_cuda_gpu_kern_sum.csv`: production-route
  diagnostic timing and invocation counts.
- Candidate `.artifacts/libgenerativeqc-h0-force-tiles.so`, SHA-256
  `cfa9440ef10f7f060d9d1a68dba7b89aa015c3ca4ef29237267080ffcf19f385`.
- Final dependency/provenance metadata refresh, with unchanged CUDA objects:
  `.artifacts/libgenerativeqc-h0-force-final.so`, SHA-256
  `df3b7f9ed857219c62e0f9b8c6bd1eb23c0cfdc96da2d0f381aae1ffa08f5252`.
  Its build log and separate public endpoint qualification are retained under
  `.artifacts/h0-force-tiles-build-final.log` and
  `.artifacts/h0-force-qualification/n2-final.log`.
- Parent `783c11ebe` binary SHA-256
  `8f311992409446d3bb8d47b4050b379aff44d3ee19bb20e915531b968b6ad5eb`.
- xTBloom `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3` binary SHA-256
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Rejected alternatives and revisit conditions

Duplicating full input/seed validation in every tile would multiply scans and
race on scratch initialization. Keeping the existing separate phases avoids
both problems. Triangular pair sharing, reassociated AO reductions and atom-owned
reductions would change more science or work/data movement than this scheduling
increment; they need separate evidence rather than being folded into this change.

Revisit the cap or mean-based choice if skewed production batches show a measured
endpoint bottleneck. The cap bounds launch resources, while grid strides bound
storage and preserve complete work. Do not infer a need to cache scientific state
or device ordinals from remaining cold/startup costs.
