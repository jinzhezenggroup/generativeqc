# Decision: share generated GFN2 primitive work across Cartesian consumers

Status: implemented
Date: 2026-10-02

## Problem

The generated S/D/Q primitive helpers separately lowered every Cartesian AO
pair. CPU traversal evaluated the same Gaussian prefactor and recurrence
intermediates up to 36 times for a primitive shell pair. CUDA lanes selected
different component switch arms containing repeated prefactor expressions.

The goal remains to exceed xTBloom complete-endpoint performance. This change
is an incremental improvement, **not evidence that the goal has been reached**.

## Decision

Intern a shell's Cartesian graphs together using the existing scalar Graph.
For CPU, emit one bounded shell-block evaluator with CSE shared across all
outputs. Store each completed output immediately and keep native contraction
order, screening, spherical transforms and origin translation unchanged.
CPU scratch is at most 36 primitive records (11,520 bytes) reused across the
primitive loop. It does not grow with molecule size.

For CUDA, retain the existing pair-per-lane schedule. Hoist exactly the
intersection of reachable DAG nodes before the Cartesian component switch.
No union of branch-exclusive arithmetic is speculatively executed. Scalar
emitter child scopes inherit definitions while retaining independent CSE maps
and non-shadowing temporary numbering. The checked primitive API remains
available; no native handwritten recurrence or reduced precision is introduced.

## Work and numerical invariants

- One Gaussian `exp`/`pow` prefactor per CPU primitive shell block instead of
  one per Cartesian pair (1, 3, 6, 9, 18 or 36, depending on shell orders).
- The CUDA artifact contains 36 prefactor expressions across its four modes
  and nine angular classes, versus 400 previously. This is a source/control-flow
  count, not a claim of 11x fewer scalar operations per active lane.
- The 96-atom GPU profile retains 5 integral-value launches and 10 integral-force
  launches over five complete calls. The source schedule does not add launches.
- Every endpoint starts fresh SCC with modified Broyden (history 8, damping 0.4),
  300 K, 300 maximum iterations, energy tolerance 1e-10 Eh and charge tolerance
  1e-8. All repeated and changed-geometry samples retain iteration counts.
- Public s/p/d bounds, ket multipole origin and all FP64 mathematics remain.

## Evidence

Baseline: master `b9c626d15`, plus the same generic-only registry link repair
used by the candidate. `GENERATIVEQC_ENABLE_AOT_SHELLS=OFF` previously lacked
`preferred_streaming_fock_shell_class_mask`; its empty implementation selects
no HF shell classes and is unrelated to xTB execution. The subsequent master
`cdd9e56cb` adds only a GPU-free cost-analysis tool and its tests/note.

Comparator: xTBloom main `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`.
Both builds use Release, CUDA 12.9.1, sm_120, no fast compilation, and the same
LP64 scipy-openblas32 0.3.34 provider. CPU measurements use one BLAS thread.
GPU runs use n5's RTX 5090 through `srun --partition=main --gres=gpu:5090:1`,
with finite limits and Slurm device visibility preserved (qualification job 1360).

The high-level xTBloom call also returns atomic charges. Both endpoints return
host energy and forces; timing includes preparation, transfer and publication.
One cold call, five same-geometry calls and five non-rigid changed-geometry
calls are recorded for each of seven committed tblite fixtures and water
clusters of 24 and 96 atoms. Geometry identity, every numerical sample and loaded
binary hashes are retained by `benchmarks/compare_xtbloom.py`.

| Case | CPU warm speedup vs master | CUDA warm speedup vs master | CUDA warm speedup vs xTBloom |
| --- | ---: | ---: | ---: |
| H2 | 0.99 | 1.01 | 0.35 |
| H2O | 1.04 | 1.04 | 0.61 |
| NH3 | 1.06 | 1.04 | 0.62 |
| CO | 1.11 | 1.06 | 0.64 |
| HF | 1.04 | 1.05 | 0.52 |
| HCl | 1.10 | 1.06 | 0.64 |
| SiH4 | 1.12 | 1.04 | 0.68 |
| Water-8 | 1.08 | 1.01 | 0.76 |
| Water-32 | 1.02 | 0.99 | 0.52 |

These short samples establish an incremental improvement on several cases;
differences near 1% are noise-scale and are not claimed as wins. Water-32 remains
about 248 ms versus xTBloom's 130 ms on CUDA. Its Nsight profile shows the
integral-value kernel improving from 1.231 to 0.899 ms/launch (1.37x), and the
integral-force kernel from 1.791 to 1.519 ms/launch (1.18x). Complete endpoint
timing takes precedence over those kernel measurements.

Across all measured repeats (including cold and changed geometry), CPU
candidate results are bitwise identical to baseline. CUDA maximum differences
are 0 Eh and 4.17e-17 Eh/bohr. Versus independent xTBloom, maximum CPU errors are
1.14e-13 Eh and 5.72e-15 Eh/bohr; CUDA errors are 5.69e-14 Eh and 3.00e-15
Eh/bohr. SCC iteration counts agree throughout. Benchmark acceptance gates are
5e-7 Eh and 5e-7 Eh/bohr, applied to every sample regardless of timing or SCC
iteration classification.

Compiled generated outputs cover all 100 Cartesian s/p/d pairs at three
geometries/exponent combinations against independent Gauss-Hermite quadrature;
ket gradients additionally use displaced-quadrature finite differences. CPU
complete endpoint/oracle/force tests pass (21 passed, 10 CUDA opt-in skips),
and the allocated CPU/CUDA endpoint/oracle suite passes all 23 tests.
The compiler/benchmark/dependency suite passes 28 tests. Compute Sanitizer
memcheck on all seven CUDA tblite fixtures passes with zero errors on n5.

Measured library SHA-256:

```text
baseline: dd9cb00baf2828d91e9e2e65267d791695d590715765772d5855e318eb1e0ca5
candidate: f18003854ad693ab8253701127036fc0149781d093e096b050874a7f8891f6e8
xtbloom: 6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f
```

Raw local reports and profiles are ignored artifacts under `.artifacts/` and
`.artifacts/n5/` in the qualification worktree, also retained at
`n5:/home/jzzeng/xtb-perf-20261002/qc/.artifacts/`. Reproduce measurements with
`benchmarks/compare_xtbloom.py --engine generativeqc|xtbloom --device cpu|cuda
--output <report.json>`, setting the explicit native-library and source paths.
Compare reports with `--reference <reference.json> --candidate <candidate.json>
--output <comparison.json>`.

## Rejected alternatives and remaining work

Moving common expressions outside switches alone gives no meaningful CPU
endpoint gain (measured ratios 0.99–1.01); CPU needs reuse across successive
Cartesian consumers. Keep CUDA's parallel ownership instead of serializing a
whole shell onto one lane merely to reuse the CPU lowering.

The public endpoint still trails xTBloom. Follow-up profiling must include CUDA
graph nodes, complete preparation/SCC/force work and retained runtime state.
The follow-up five-call trace records 2,275 visible candidate kernel instances
versus 1,601 in xTBloom; candidate integral-force/eigensolver preparation runs
10 times versus 6. This points to retained-state/setup work amplification in
addition to slower primitive arithmetic. Device-launched SCC graph coverage
must be checked before treating visible kernel totals as complete SCC counts.
Do not extrapolate these primitive wins to whole-method superiority or change
SCC tolerances/precision to obtain a faster comparison. CPU `perf` sampling on
n3 was unavailable (`perf_event_paranoid=4`); this is not numerical evidence.

When moving measurements between machines, xTBloom's Python loader may preload
an old `/usr/local/cuda` before the requested toolkit. The n5 qualification
environment uses a private `nvidia/qualified_cuda/lib` link to CUDA 12.9.1 so
both libraries resolve the same runtime cohort. Preserve binary hashes and
inspect the loaded cohort rather than assuming `LD_LIBRARY_PATH` wins.
