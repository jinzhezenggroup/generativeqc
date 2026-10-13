# Experiment: native same-owner WB97M-V density seed

Status: diagnostic prototype, default off; no default or benchmark promotion
Date: 2026-10-14

The source-matched bilateral integration `7001858f1` still reaches the original
1800-second whole-point limit in cold/0 at 96 atoms (Slurm 8241). A finite trace
measures about 82.6 seconds per exact VV10 feature sweep. Force ingredient reuse
does not remove these cold SCF sweeps.

`GENERATIVEQC_VV10_NATIVE_SEED=1` tests a bounded FP64 initial iterative stage
inside the same prepared native CUDA KS owner. It uses the target AO basis,
moving grid, semilocal functional and exact range-separated J/K. Only the
nonlocal term is omitted while forming a provisional density. No CPU/reference
solve, oracle density, new owner, coarser grid or precision change is involved.
The stage ends at maximum residual below 1e-9, after 16 provisional iterations,
or when two iterations remain in the original total budget. Resident warm
replays skip this stage. UKS, lower precision, incremental J/K and device chunks
are outside the experiment.

The provisional stage cannot publish convergence, a final state, warm cache or
forces. Its transition clears DIIS and the energy baseline. At least two full
WB97M-V Focks are then required, with unchanged energy, density, maximum/RMS
physical residual and final-state validation gates. All provisional iterations
remain in the original 100-iteration budget and complete endpoint timer.

This experiment deliberately emits per-iteration seed/exact counts to stderr.
These are diagnostic runs, not ordinary-driver publication evidence. Promotion
would require stable semantic work diagnostics, failure/recovery and tight-budget
tests, independent cold/warm/moved E/F gates, larger controls, a bounded fallback
policy and measured complete endpoints. An initial guess can change a nonlinear
SCF basin, so numerical equivalence must be demonstrated, not presumed.

Refs #2426. Accuracy-only cold/moved oracles are separate from reference timings.
