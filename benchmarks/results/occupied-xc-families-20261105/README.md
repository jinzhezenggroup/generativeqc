# Occupied-XC families and handoff costs

This capsule qualifies a bounded FP64 **initializer**, not a new point
functional or a relaxed final-state certificate. Default unset admission now
works by ingredient layout for LDA, GGA and tau-meta-GGA, including their
supported hybrid semilocal parts. It does not claim all Libxc products,
Laplacian/nonlocal/response consumers or all inputs are accelerated.

## Scientific image

- Source: `3c41b36142d2fe52fc7b3cb7de89e42fef670b42`.
- Native source identity:
  `5db4c208f23bc6215e03241dd08214258c69995291b8c8aa99a85b3b47adc8a3`.
- Library SHA-256:
  `472e00282a8795b4075ddd6900464a40c09217de7e6b163bbabec0ebe7e3fb5b`.
- Release CUDA 12.9 / sm_120, CUDA AOT off, verified sccache 0.16.0
  CXX/CUDA launchers. Retained before/after cache statistics describe the
  incremental qualification build, not a clean-cache performance claim.
- RTX 5090 / 32 GB, finite Slurm `main` allocations. Node4 uses driver
  580.178.04 and Python 3.13.11; node5 uses driver 570.86.16 and Python 3.13.9.
  Both use CuPy 13.6.0 and PySCF 2.14.0, with one BLAS/OpenMP thread.
  Scheduler-assigned device visibility is preserved.

Primary family populations use node4; independent confirmations and the original
96-atom workload use node5. Every matrix/default comparison stays on one host;
no samples or clocks are pooled across nodes or scientific images. The later
documentation/evidence commits do not change this scientific-source identity.

## Complete endpoint results

Each population contains three alternating, fresh-process pairs of actual unset
default versus `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=0`. A family process measures
construction/preparation plus cold E+F, then frozen warm, displaced E+F and
displaced warm. Timers include native SCF, analytic moving-grid forces and export,
but not independent oracle work or receipt serialization. No traced call is
timing evidence. The predeclared floor is `max(2%, 2 * sum(relative MADs))`.

There are 624 final-image family E+F observations: seven functionals times
RKS/UKS on 3 atoms, M06-2X/12 RKS, six 48-atom RKS methods, and five separate
confirmation populations. The original r2SCAN-3c/96 protocol retains another
twelve cold/frozen-warm E+F observations under its established qualifier.

| Cold population | Matrix median (s) | Default median (s) | Less time | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Original r2SCAN-3c/96, node5 | 30.449281 | 29.303447 | 3.763% | Above 2% floor; 10/10 Focks |
| M06-2X/48, node4 | 9.861291 | 9.620451 | 2.442% | Above 2% floor; 10/10 Focks |
| M06-2X/48, node5 confirmation | 10.069907 | 9.847389 | 2.210% | Independently above 2% floor |
| M06-2X/12, node4 | 2.563963 | 2.552886 | 0.432% | Neutral, no former cold restart penalty |
| MN15/48, node4 | 16.565734 | 16.354266 | 1.277% | Below floor; not a speedup claim |
| LDA/48, node4 | 5.216277 | 5.169899 | 0.889% | Below floor; node5 confirmation also neutral |
| PBE/48, node4 | 4.822132 | 4.807869 | 0.296% | Below floor; node5 confirmation also neutral |
| PBE0/48, node4 | 7.526629 | 7.518476 | 0.108% | Below floor; node5 confirmation also neutral |
| B3LYP/48, node4 | 7.631709 | 7.611981 | 0.259% | Below floor; not a speedup claim |

Original frozen warm is `11.340720 -> 11.302938 s` with one Fock on both
routes, neutral rather than an extra speedup claim. All primary final family
phases remain within the retained regression floor. Keep the counterexample:
the separate PBE0/3 RKS confirmation flags frozen warm at **2.045% slower**
(about 1.3 ms); the primary PBE0/3 warm population is neutral. This isolated
disagreement is retained, not erased, relabelled or used for a functional
exclusion. Admission is not a universal per-input profitability guarantee.

## Semantic work and discarded candidates

Three independent costs could hide the saved XC contractions:

1. Candidate `8299cd88d1400931e67958e1ca6181912f49e196` discarded useful
   strict DIIS entries after six strict sweeps. Retained M06-2X UKS/3 pairs
   show cold 24 versus 25 Focks and displaced 18 versus 20. Natural bounded
   ring ageing preserves the strict predictor and restores matching counts.
2. Candidate `ca8855d9dc5d64ffec46f011edf295ce04700811` re-prepared an
   unchanged exact ERI lease. Separately traced M06-2X RKS/3 receipts count
   **105950 -> 52975** actual `coulomb_recurrence` productions with identical
   SCF/Fock counts and paired observables. XC-only handoff now preserves the
   existing lease; genuine precision transitions still replace it.
3. Candidate `4dd6b34b53b882688e18e4a0036acc20326a5704` still disabled
   incremental J/K at XC handoff. Cold M06-2X/48 had 32 versus 48 structured
   basis actions despite ten Focks on each route. Final matrix/default both
   have **32 actions, six delta builds, three anchor builds and one final full
   build**. The ordinary full-energy refinement and canonical gates still
   forbid deltas. The XC-only handoff does not alter J/K arithmetic or anchors.

The capsule also retains every completed pre-incremental primary/confirmation
population, including neutral and regressing phases, under its separate image.
Some pre-incremental timings show abrupt warm-latency changes as well as cold
changes; they are not adjusted or pooled with final confirmations. Traced
semantic work establishes the J/K defect independently of those clocks.
The earlier two campaigns were stopped for the diagnosed fixes; only the
declared complete DIIS counterexample and source diagnostic qualify here.
Incomplete/stopped observations supply no accepted performance population.

The GGA implementation projects values once and reconstructs `C(C^T phi)` in
the existing feature reduction, using the original AO gradients. It admits
only `2 * rank < active`; LDA and tau-meta-GGA require `rank < active`.
Existing dead scratch supplies the panels. Normal admitted points do not
perform a dense original-D comparison; poisoned-density GPU fixtures protect
that property. Nonfinite points retain the original-D fallback. Strict final
features, full-density J/K qualification, canonicality and forces are unchanged.

## Numerical and execution checks

- Small independent CPU PySCF default-grid oracles and paired matrix controls
  keep `2e-9 Eh` / `1e-7 Eh/Bohr` gates. Seven functionals (LDA, PBE, PBE0,
  B3LYP, r2SCAN, M06-2X, MN15) cover RKS/UKS, original and displaced geometry.
  Largest primary small independent errors are `6.83e-13 Eh` and
  `8.30e-10 Eh/Bohr`.
- Large 48-atom GPU4PySCF full-Fock references at grids 5 and 6 independently
  qualify both geometries. Their mutual stability must pass half the existing
  `1e-5 * atoms Eh`, `1e-4 Eh/Bohr` maximum and `3e-5 Eh/Bohr` RMS gates.
  Every native vector passes both references; matrix/default pairs still keep
  the tighter `2e-9` / `1e-7` gates. Native grid/solver settings remain paired.
- Original r2SCAN-3c/96 reuses the checksum-pinned independent dense references
  and unchanged qualifier from `readme-final-20261022`; this is static-geometry
  qualification, not a newly established trajectory accuracy claim.
- Final-image scheduled GPU suite: **320 passed** (272 emitted projection
  fixtures, 32 native lifecycle fixtures, 16 independent/control endpoints).
  Memcheck, racecheck and synccheck: **36 cases each, zero findings**.
- Final host/controller/native-contraction suite: **278 passed** before the
  additional retained-capsule tests. Compiler structure: **586 modules,
  zero dependency errors**. The native Libxc catalog remains explicitly stale
  with zero currently qualified public products; this change does not promote it.

Diagnostic generation lists are split by real solve boundaries and end with a
strict matrix comparison tail after any provisional epoch. XC evaluation counts
are not blindly equated to Fock totals: a final full Fock probe can reuse an
already strict XC result. Separate trace work, iterations and Fock counts remain
available, instead of inserting synthetic trace entries to make counts match.

## Replay and reproduction

Offline replay needs NumPy for the existing original-workload qualifier, but
does not load a native library, CUDA, PySCF or GPU4PySCF:

```bash
PYTHONPATH=.:python python benchmarks/results/occupied-xc-families-20261105/replay.py
PYTHONPATH=.:python python -O benchmarks/results/occupied-xc-families-20261105/replay.py
```

It rechecks image/host separation, complete force vectors, independent and
paired gates, grid stability, semantic work, actual controls and original
reference checksums. Corruption probes and regression tests remain active under
optimized Python. `endpoints.json.xz` retains the exact execution scripts in
`commands`; raw profiler events, incomplete receipts and build outputs stay
outside tracked results. Adapt the raw directory/library paths, bind the
scientific source and native image, and run the scripts through finite `srun`
allocations. Generate new independent references with
`benchmarks.occupied_xc_functionals oracle`; never supply them as production
densities. See the [decision note](../../../.agents/notes/implemented/performance/2026-11-05-occupied-xc-feature-families.md)
and [current contract](../../../docs/developer/xc_contractions.md).
