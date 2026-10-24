# Decision: reuse incoming residuals in native CUDA DF force amplitude solves

Status: implemented
Date: 2026-10-11

## Problem

#2211 qualified weighted Anderson mixing for energy-only CUDA DF owners but
excluded owners retaining molecular source response. Force calls therefore
reevaluated outgoing Jacobi trial equations before the unchanged independent
physical replay, even though source retention does not change CC equations.

## Decision

Enable the existing option for native CUDA DF amplitude owners whether or not
they retain source response. Only the forward T solve changes. Do not
differentiate DIIS history, change Lambda/response equations, reuse stale
residuals, or change precision, Q caps, storage admission or provider policy.
The Lagrangian force depends on the accepted amplitudes and separately
qualified adjoints, not the iteration trajectory used to reach that root.

CPU, conventional and supplied-problem defaults remain unchanged. Explicit
`SolverOptions::diis_input_residual=false` retains the original internal path;
no-DIIS execution remains the original Jacobi path. Every observed/final state
and the independent expanded physical replay still evaluate actual amplitudes.

## Evidence

The measured full-library source is `34dd51ed3ebc5b03b17f095de2cd7b39903923c8`
plus the retained one-file overlay, not an inferred latest-master library.
The clean baseline is compiler-cached; the candidate rebuilds its changed
method owner and build-identity consumer without rebuilding scientific CUDA.
Both arms use byte-identical endpoint and force-probe binaries.

Fresh ethane230/o9/v221/Q488 cold force ABBA, Slurm 2898 on node2, ordinary
history 8, CC Q8, Lambda Q32, strict FP64, optional cuBLASLt off, 64 GiB budget:
complete process medians are 275.829710826 and 250.339765314 seconds, a
descriptive 9.2412% reduction. CCSD medians are 61.072098459 and
35.534116752 seconds. Observations decrease 20 to 19 and primal evaluations
38 to 19. Lambda actions remain 22; exact orbital actions remain 13.
Complete logical numeric capacity remains 19,509,839,897 bytes; this is not
physical peak VRAM or a universal RSS claim.

Native zero-problem work queries in separate Slurm 2899 reproduce every
published CC work counter from per-primal and independent-replay counts.
Contraction terms decrease 26,799,066,988,432 to 15,735,124,233,906;
GEMMs decrease 54,862 to 34,019. Query timing is not endpoint evidence.

All original same-Hamiltonian gates pass: total energy 1e-8 Eh, triples 1e-10
Eh, fresh R1/R2 1e-10, and independent directional forces 3e-7 Eh/bohr at both
retained FD steps. Maximum large directional error is 7.11e-9 Eh/bohr; the
control/candidate force-vector difference is 6.26e-12 Eh/bohr.
Independent small CCSD/CCSD(T) force FDs, all-coordinate water, auxiliary-g,
fixed-rank, public force/energy and failure-publication checks pass in both
arms. Candidate matrix: 38 cases; baseline: 26 already-passing public/API
cases plus 12 force cases after fixing a missing staged SONAME alias.
Memcheck reports zero errors. CPU incoming-residual checks: four passed.

Original loader failure and a result-reader field-name failure are retained.
The separately completed reader-failure baseline is a validation pilot,
excluded from fresh ABBA. Neither independent small matrix is redundantly
rerun after the reader repair. Four endpoint observations do not establish
statistical, universal convergence or global speedup claims.

## Revisit when

Change the policy only with new physical convergence and complete-force
evidence if diagonal preconditioning/damping varies within a solve, a
qualified force case regresses, or another backend/consumer is admitted.

## References

- #2211 and the preserved energy-only predecessor decision:
  `2026-10-10-df-cc-input-residual-mixing.md`.
- `benchmarks/results/df-cc-force-input-residual-20261011/`.
- `src/methods/rccsd_method.cpp` and `tests/python/test_df_cc_native_solver.py`.
