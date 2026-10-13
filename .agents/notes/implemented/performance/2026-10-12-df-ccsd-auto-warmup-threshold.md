# Decision: qualify CCSD warmup by complete time, then default to bounded auto

Status: implemented
Date: 2026-10-12

## Problem

The existing FP32 auxiliary warmup was opt-in and had been measured against an
older master with different reference/CCSD schedules. Waiting for approximate
convergence is not the objective: conversion, the strict DIIS restart, strict Q
tiling, independent replay and subsequent response work all contribute to the
complete endpoint. A looser switch can reduce FP32 evaluations while increasing
more expensive FP64 work. Uniformly enabling warmup also penalizes small systems.

The frozen baseline for this decision is master
`2a6de26eea1f83e9fd1e8df01c189fea44058b6b`, fetched on October 12, 2026.
Earlier warmup evidence is not reused as a performance claim against that master.

## Decision

- Native CUDA correlation-only-DF endpoints default to automatic warmup selection.
  At least 9 occupied and 128 virtual orbitals request the existing mixed route.
  Smaller/low-occupied systems remain FP64. The existing provider, dimension and
  complete-memory admission still owns whether a requested route can execute.
- Native internal requests use `std::optional<bool>`: absent means auto, `true`
  forces the existing diagnostic candidate, and `false` retains the FP64 ablation.
  CPU/conventional methods and supplied-Hamiltonian `SolverOptions` defaults do
  not change. The public C ABI and physical precision selector do not change.
- Keep the switch at `100.0 * 1e-9` (approximately `1e-7`) and the 24-evaluation
  upper bound. This is a warm-start stopping rule, never a convergence certificate.
  FP64 strict polishing, fresh strict history, pair admission/projection and the
  independent expanded replay remain mandatory. Lambda/(T)/orbital/nuclear
  precision policies remain unchanged.
- Expose the switch as an appended benchmark/native control without weakening
  the finite/separated-stage admission inequality. Benchmark position 30 accepts
  `auto`, `0` or `1`; position 32 overrides the residual target. Existing explicit
  numeric ablations retain their argument positions and meaning.

The auto boundary is a conservative rollout bracket, not a fitted universal GPU
crossover. Occupied count 9 and virtual spaces 135/221 are the measured winning
domain; virtual spaces up to 87 are losing or neutral. The boundary avoids
promoting the candidate on those known losing cases. Performance qualification
currently covers RTX 5090, not every CUDA device or every point between samples.

## Measured evidence

The retained [qualification records](../../../../benchmarks/results/df-ccsd-auto-20261012/README.md)
bind source patches, actual binaries, molecular inputs, Slurm allocations and
unrounded endpoint/work records. CUDA work runs through finite `srun` allocations
on n1/n4; n3 under maintenance is not used. Compilation uses verified
ccache and is outside the measured native endpoint boundary.

All comparators use conventional exact RHF plus correlation-only DF, DIIS 6,
automatic CCSD Q tiling (limit 32), the original strict convergence tolerances,
and FP64 (T)/response arithmetic. No imported scientific state enters the native
endpoint. Cold calls use fresh processes; final auto/lifetime checks additionally
exercise same-process reuse and a `1e-4`-bohr geometry displacement.

The coarse ethane/aug-cc-pVTZ scan tries `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-7`;
the fine scan adds `2e-7`, `3e-7`, `5e-7`. At `1e-3`, complete energy time rises
49.794 -> 51.294 seconds. `1e-5` also loses (50.212 seconds). `1e-7` gives
47.583 seconds and 23.324-second CCSD; `3e-7`/`5e-7` are essentially tied in
complete time but their approximately 23.43-second CCSD stage is not faster.
Three alternating-order complete E+F repeats compare FP64, `1e-7` and `5e-7`;
the retained summary, not the pilot alone, is the complete force qualification.
The three-repeat medians are 249.523569 -> 247.303254 seconds for complete E+F
and 25.661352 -> 23.326304 seconds for CCSD (0.8898% and 9.0995% reductions).
The `5e-7` candidate takes 247.337432 / 23.437707 seconds despite one fewer
evaluation. At `1e-7`, maximum paired errors are `3.183231456205249e-12` hartree
and `4.801936626108727e-12` hartree/bohr; no measured repeat is excluded.

Three-repeat complete E+F checks also cover water24, methane34, ethane58,
water92 and ethane144. Water24/methane34/ethane58 warmup increases CCSD time
despite tiny total-endpoint changes. Water92 CCSD increases 0.583 -> 0.661
seconds. Ethane144 CCSD falls 5.905 -> 5.374 seconds, and complete E+F falls
54.676 -> 54.117 seconds. These distinguish the native auto default from a
blanket `df_mixed_precision=true` default.

CCSD evaluation counts are not the objective: ethane144 changes from 18 FP64
evaluations to 11 FP32 + 9 FP64, while ethane230 at `1e-7` uses 12 FP32 + 10
FP64 instead of 18 FP64. The approximate schedule may increase total work; every
record retains GEMM summands, casts, packing, pair admissions and strict replay.
No reduced-total-work or memory-peak claim follows from these timings.

## Scientific and compatibility gates

- Independent determinant-space CPU checks retain the physical equation oracle.
- GPU threshold tests cover `1e-2` through `1e-7`, strict final amplitudes/energy/
  replay, finite threshold admission, hard warmup caps, asymmetric initial states,
  resource refusal and sticky nonfinite arithmetic.
- H2/water/LiH complete molecular calls compare to pinned PySCF 2.14.0 energies
  and independent two-step directional energy finite differences. Explicit
  `1e-3`/`1e-5` switches and the unchanged default retain the force gates.
- Final compiled auto tests retain FP64 on small molecules, admit warmup on the
  measured larger domain, and preserve the public energy/force endpoint. Lifecycle
  comparisons gate cold, same-process and moved-geometry outputs separately.
- Every measured repeat must pass the original independent CCSD replay limit
  `1e-10`, response limits `1e-8`, paired energy error `2e-9` hartree and paired
  force error `1e-7` hartree/bohr. No inaccurate repeat is excluded for timing.

The frozen explicit-policy timing binary and the final default-policy binary are
different artifacts and have separate hashes/source patches. Final default-on
checks are retained separately; do not describe a later rebuilt binary as the
one measured in the frozen matrix.

## Rejected alternatives and revisit conditions

Do not select the earliest switch or the fewest counted iterations without full
timing. Do not retain approximate DIIS rows, relax final tolerances, suppress
independent replay, or retry a numerical failure under FP64 to conceal it.
Do not enable this schedule on every small molecule merely because its mixed
residual eventually passes a strict certificate.

Revisit the auto boundary and switch if a different device, occupied domain,
larger virtual/Q space, damping/history policy or resource ceiling changes the
complete endpoint. Include independent scientific gates and actual work counts;
kernel throughput alone does not justify extending the default domain.
