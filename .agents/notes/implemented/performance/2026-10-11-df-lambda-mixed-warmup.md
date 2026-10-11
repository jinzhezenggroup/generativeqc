# Decision: auxiliary-only mixed Lambda warm start with a fresh strict solve

Status: implemented, experimental opt-in; endpoint qualification recorded below
Date: 2026-10-11

## Problem

The first CCSD warmup/scratch-sharing candidate improves complete E+F, but leaves
Lambda near 79 seconds for the recorded ethane case. A diagnostic Nsight Systems
run of that frozen candidate, Slurm job 7541, shows large FP64 GEMMs dominate
global kernel time. This profile is not matched endpoint timing or a Lambda-only
kernel attribution. IR analysis identifies two essential large auxiliary GEMMs;
hoisting Q-independent auxiliary packs removes only about 4% of packing and
introduces new retained storage/lifetime complexity.

## Decision

Uniformly retype the existing compiler-owned auxiliary transpose to FP32 for a
bounded GMRES warm start, leaving the primal, core adjoint, RHS, preconditioner
and host Krylov arithmetic FP64. Widen every Q output independently into the
original ordered FP64 accumulation. Pedantic provider arithmetic and existing
round-to-nearest/sticky-error checks remain mandatory.

Only a finite proposal crosses into a new original FP64 GMRES solve, with empty
Krylov history and a fresh physical action. Finite warmup exhaustion/stagnation/
breakdown cannot accept a result; nonfinite failures propagate. Strict residual
tolerances, independently expanded audits and all final parameter/factor/force
VJPs are unchanged. The experiment remains default-off and separately controlled
from CCSD mixed preconvergence.

Halve the requested strict Q tile only for this opt-in. FP32 scratch uses the
front of the existing shared strict/audit lease. Immutable FP32 factors/T1/tau
and dynamic cut seeds use a tail beyond the maximum scratch needed by FP32
auxiliary, FP64 core (including retained-core dynamic scratch) and FP64 tau
pullback. No extra owned device allocation is introduced. The single-use epoch begins
after fresh strict primal/RHS checks and ends permanently before strict
replay/audit/VJP stages may overwrite the tail. Optional descriptor/storage
refusal retains bounded strict actions; the host warm-proposal and descriptors
are included in the existing numeric budget.

## Rejected alternatives

- Hoisting Q-independent packs alone does not address essential FP64 GEMMs and
  offers limited work reduction relative to the added cache ownership.
- Lowering final Lambda or force arithmetic changes the scientific certificate.
- Reusing approximate Krylov history or residuals as strict state contaminates
  the boundary and hides a required exact physical action.
- Allocating a second FP32 arena or full input cache defeats memory reduction.

## Invariants

- Preserve the original Hamiltonian, strict tolerances and independent oracle.
- Default FP64 consumers retain their existing schedule and precision.
- Charge all mixed and strict actions/iterations, conversions and packing work.
- Never overlap cached warmup inputs with retained-core dynamic scratch.
- Strict audits/parameter/factor stages cannot execute during the mixed epoch.
- Report arena ownership separately from sampled occupancy or allocator peaks.

## Evidence

The source build and initial Slurm job 7554 pass 64 targeted compiler/native/
complete-endpoint tests, including independent H2/water/LiH energies and two-step
directional finite differences. CPU-only lifetime/compiler coverage passes
56 tests with 27 expected GPU skips. Native coverage includes finite two-step
exhaustion, exact optional-storage refusal and nonfinite nonpublication.
Matched endpoint measurements and final-source sanitizer receipts belong in
`benchmarks/results/`, not profiler timings.

The clean frozen candidate `04cdb116bcd377788ace975e65fd08f02becf4c0` receives
64 passing targeted tests, 167 CPU passes/27 expected skips and zero water
memcheck errors/leaked bytes in Slurm job 7560. One complete cold triplet observes
E+F 293.210665477 → 289.226577167 seconds and Lambda 78.710987019 → 74.766767397
seconds versus the prior CCSD-only mixed PR. FP64 Q16 alone takes
305.645875821/91.193064252 seconds for E+F/Lambda, respectively. Both Q16
consumers own 6,212,893,888 Lambda device bytes versus Q32's 10,324,999,872;
the storage reduction is tiling, not FP32 arithmetic itself. Whole-endpoint
sampled peak remains 27,547 MiB. Total Lambda actions increase 22 → 25.

This is exploratory timing, not a repeated/significant performance certificate.
The campaign stops after one triplet to preserve remote master integration;
the user then requests no repeated validation. Later integration and the
mixed-only option admission guard are not requalified by earlier receipts.
Keep that version boundary explicit and leave the experiment default-off.
Full values, oracle errors and source identities are retained in
[the numerical publication](../../../../benchmarks/results/df-lambda-mixed-warmup-20261011/README.md).

## Consequences and revisit conditions

Warmup can add iterations, conversions and another exact initial action; faster
FP32 kernels alone do not imply a faster complete endpoint. The smaller strict
tile may trade launch count for storage. Revisit thresholds, tiling or default
policy only with new source-matched complete E+F cohorts, original independent
gates and explicit total work/memory accounting, especially for difficult
response cases and additional devices.

## References

- [Current precision contract](../../../../docs/developer/df_ccsdt.md)
- [Prior CCSD/scratch decision](2026-10-11-df-ccsd-mixed-warmup-shared-lambda-scratch.md)
- `tests/python/test_df_lambda_mixed_precision.py`
- `tests/python/test_df_triples_mixed_endpoint.py`
