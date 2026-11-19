# Decision: full-CTA Direct RHF energy and default block convergence walk

Status: implemented
Date: 2026-11-19

## Problem

Each Direct RHF iteration contracts `nbf x nbf` matrices twice in single-block
kernels:

- **Energy:** `compute_energy_kernel` runs one 32-thread warp per system.
- **Convergence walk:** the residual/RMS/copy walk also runs on 32 threads.

At 768 AOs they cost 89 ms and 179 ms of a cold README Direct HF96 call on
H100. The opt-in 256-thread walk (#3042, proposal
`../../proposed/2026-11-07-rhf-block-convergence.md`) was qualified as a
default-off component and left awaiting complete endpoint and control evidence.

## Decision

- **Energy:** `compute_energy_kernel` folds warp sums in fixed warp order when
  launched wider. RHF launches it with `kEnergyReductionThreads = 512`. One
  warp keeps the original arithmetic, and UHF keeps its 32-thread launch.
- **Walk default:** `GENERATIVEQC_HF_BLOCK_CONVERGENCE` defaults to the
  256-thread walk.
- **Walk domain:** the walk is unchanged, and so is its domain: restricted,
  nonmixed, nonincremental quartet Direct without physical-reference export.
  `0` restores 32 threads and invalid values reject. The control remains part
  of resource identity.
- **Arithmetic:** the energy sum and the RMS are reassociated. Lane-zero
  convergence policy, refinement and every tolerance are unchanged.

## Evidence

`benchmarks/results/h100-direct-rhf-block-reductions-20261119/`.
- **Timings:** README Direct HF96 cold 6.65-6.66 → 6.42-6.44 s and warm
  1.400 → 1.372-1.378 s. HF48 cold 4.23-4.29 → 4.18-4.20 s.
- **Correctness:** iteration and Fock counts are identical, every arm
  qualifies, and same-phase forces agree within 5.7e-13.
- **Native controls:** the 32/256 matrix-walk, refinement and cached-policy
  refusal controls pass.
- **Memcheck:** clean.

## Revisit when

Extend the walk to UHF, mixed precision or incremental owners only with their own
matrix/lifecycle controls and endpoint qualification. Never relax a convergence
limit to accept a different reduction trajectory.
