# Opportunity: spatial bisection for DF r2SCAN-3c cold endpoints

Status: proposed; existing opt-in measured, automatic policy unchanged
Date: 2026-11-10

Superseded by the qualified bounded automatic default in
[`2026-11-10-default-large-semilocal-xc-bisection.md`](../implemented/performance/2026-11-10-default-large-semilocal-xc-bisection.md).
The measurements below remain the earlier opt-in pilot, not the new-default cohort.

## Question and result

Can the master DF r2SCAN-3c 96-atom cold endpoint save about 0.5 seconds?
For the README water32mer workload on the measured RTX 5090, the existing
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` opt-in saves **1.761553 s / 7.443%**
in a small complete-endpoint pilot. No new kernel is necessary to demonstrate
the opportunity. This does not qualify a global automatic-default change.

## Source and measurement boundary

- Master: `d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`, freshly fetched.
  Python and manifest inputs are exported from this exact revision.
- Reused native library was built at
  `805de425a7c9cd323bc3353d466493cf4eadd397`; its SHA256 is
  `21838a057f9b6f3ce8b025ec94e289ce29ff30827b1afd3da23f95f1fa8b2741`.
  Git object comparisons verify identical `src`, `include`, `cmake`, and root
  `CMakeLists.txt` inputs at these revisions. This is not a newly rebuilt or
  relabelled d35 binary; the intervening change adds Python D3 Hessian support.
- Node n4 / Slurm node4, one RTX 5090, CUDA 12.9.1, one OpenMP/OpenBLAS/MKL
  thread. All GPU execution uses finite Slurm `main --gres=gpu:5090:1`
  allocations with scheduler-assigned device visibility intact. No n3 GPU work.
- The maintained `benchmarks.readme_defaults` Standard DF endpoint measures
  constructor, preparation, native cold SCF, analytic forces, D4/gCP and host
  export. Process/import/context startup and empty-cache compilation are not
  included. The worker rejects PySCF/GPU4PySCF imports in native processes.
- Orbitals: 1,088 spherical def2-mTZVPP AOs; auxiliaries: 6,816; native stock
  grid: 1,078,336 points. Moving-grid and auxiliary-basis response remain on.
  FP64, initial guess, grid, screening and convergence gates are unchanged.

## Clean endpoint evidence

Slurm1552 executes center-a, bisect-a, bisect-b, center-b. Slurm1554 executes
bisect-c then center-c. Every population is a fresh process and owner.

| Population | Default center cold / s | Bisection cold / s |
| --- | ---: | ---: |
| a | 23.640100255 | 21.905314256 |
| b | 23.666867329 | 21.892718642 |
| c | 23.677152957 | 21.907721447 |
| median | 23.666867329 | 21.905314256 |

All six cold calls have 10 SCF iterations and 10 reported Fock builds. Each warm
call has one iteration/build and uses its own engine-local frozen cold density.
The a/b processes have one warm call each; c has three. Median process-level
warm medians are 6.820129 s default and 6.595051 s bisection. This mixed warm
sample count is a pilot, not the final all-three-warm qualification protocol.

All 16 clean E/F exports pass paired gates of energy 1e-8 Eh, force max
1e-7 Eh/Bohr and force RMS 3e-8 Eh/Bohr. Observed maxima versus center-a are
9.10e-13 Eh and 3.10e-12 Eh/Bohr. All pass the retained independent GPU4PySCF
default-grid comparison gates of energy/atom 1e-5 Eh, force max 1e-4 Eh/Bohr
and force RMS 3e-5 Eh/Bohr; maximum force discrepancy is about 1.35e-6 Eh/Bohr.
The independent input geometry is checked exactly. This is neither a new
dense-grid reference nor a displaced-geometry/grid-convergence certificate.

## Why it helps

Separate intrusive center/bisection profiles use Slurm1551/1553, not clean
timing populations. The cold explicit XC work per Fock changes as follows:

| Census | Center | Bisection |
| --- | ---: | ---: |
| Point-active-AO visits | 300,677,440 | 262,959,168 |
| Point-active-AO-square sum | 89,166,908,992 | 66,625,602,624 |

The square-work proxy falls 25.28% and AO visits fall 12.54%. All quadrature
points/weights remain present; this groups spatially nearby points so each
tile's existing 1e-16 active-AO selection needs a smaller union. There are still
10 cold XC submissions, the same occupied/strict handoff pattern, and the same
32-tile point batches. No FP32 conversion or density-GEMM provider is introduced.

Inclusive cold XC event time falls 9.475 to 7.693 s: density projection 3.705
to 2.980 s, potential contraction 4.199 to 3.281 s, AO evaluation 1.111 to
0.970 s. DF J remains about 2.44 s and stationary force about 5.02 s. These
CUDA-event regions include submission gaps and overlap nested operations;
they are attribution, not additive kernel-busy times or speedup evidence.

## Decision and next qualification

Prefer investigating bounded automatic spatial-schedule admission before
new DF integral kernels, reduced grids, relaxed convergence or lower precision.
Retain the current center/canonical routes, optional host/device capacity checks,
resource-ledger ownership and canonical grid-response force view. Admission must
depend on generic work/resource properties, not molecule or benchmark names.

Before automatic promotion, run balanced complete cold/three-warm cohorts and
48/96 plus small/large, other semilocal functionals, moved geometry, and resource
refusal/lifetime cases on the relevant devices. Reuse the existing bisection
tests and stronger independent numerical qualification where applicable.

Task-local complete JSON exports, traces, allocation receipts, runners and the
recomputing `summarize.py` remain under
`.artifacts/df-r2scan3c-cold-20261110/`; the remote working directory is
`n4:/data/jzzeng/df-r2scan3c-cold-20261110/`. These ignored artifacts are local
investigation evidence, not newly published archival qualification. Preliminary
deployment failures for missing exported manifests/compiler assets are not
counted as endpoint passes; surviving deployment logs are retained separately.

Historical mechanism and earlier, different-source qualification:
`benchmarks/results/xc-spatial-bisection-20261023/README.md` and
`.agents/notes/proposed/2026-10-23-spatial-tile-bisection.md`.
