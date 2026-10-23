# Decision: scope the range-DF benchmark selector to one invocation

Status: implemented
Date: 2026-10-23

## Problem

PR #2526's core-b worker reported four fitted WB97M-V Hessian failures at
the native/Python Fock provenance check. The complete Hessian module passed
alone with the same source-built CPU library. Running
`test_range_separated_df_benchmark.py` immediately before
`test_fitted_hvp_matches_reconverged_energy[False]` reproduced the failure:
24 benchmark cases passed and the Hessian failed.

The benchmark's callable `main()` assigned
`GENERATIVEQC_DF_RANGE_EXCHANGE=df` without restoring the caller's environment.
Later calculations consequently prepared a fully fitted LR operator instead
of the default full-range DF plus exact LR operator. The Fock provenance gate
correctly rejected that different Hamiltonian. This was process state leakage,
not evidence that the Fock gate or the full-rank Hessian oracle was inaccurate.

## Decision and invariants

Scope the selector to `main()`'s measurement and restore the exact prior string
or absence in `finally`, including preparation errors and `KeyboardInterrupt`.
The selected operator remains active throughout preparation, warm execution,
and geometry replay. Output reservation and failure journals retain their
existing behavior and timing boundaries. This process-global benchmark entry
point is not intended for concurrent in-process campaigns.

Do not weaken the `2e-8` Fock agreement gate, alter the numerical implementation,
or remove the earlier explicit rejection of rank-truncated fitted Hessians.
Clearing the environment only in tests would hide the callable runner's leak.
An isolated Hessian pass alone cannot establish independence from preceding
tests; the regression validation must include the benchmark prefix in the same
Python process.

## Evidence and reproduction

The original source was `37a7ca142064d65e2fca3411588b6eeb85c6ca8f`.
The Release/O3 CPU library (scalar linear algebra, CUDA off, verified ccache)
had SHA-256
`efc33a41d3942a24e991a30305397f65f0d887667d2d272f49e47c73cfbc5d05`.
Its standalone Hessian module passed all 13 cases before this change.

The regression command, using that unchanged library and one CPU/BLAS thread,
is:

```bash
PYTHONPATH=python:tests/python \
GENERATIVEQC_LIBRARY="$PWD/build/cpu/libgenerativeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/python/test_range_separated_df_benchmark.py \
  tests/python/test_wb97mv_df_hessian.py -q
```

The combined run passed all 66 cases (53 benchmark and 13 Hessian) in 77.04 s.

Benchmark tests cover both selectors, absent and preexisting environment values,
successful energy/force campaigns, preparation/runtime failures, import failure,
and interruption. The Hessian tests retain their independent reconverged-energy
finite-difference oracle and their original rank and Fock acceptance gates.
Verbose failure and validation receipts remain in ignored local artifacts;
the command above reconstructs the behavioral check without those artifacts.
