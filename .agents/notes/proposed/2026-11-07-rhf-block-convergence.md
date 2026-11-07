# Decision: opt-in one-block RHF convergence matrix walk

Status: proposed; local source preparation, CUDA/endpoint qualification pending
Date: 2026-11-07

## Decision

`GENERATIVEQC_HF_BLOCK_CONVERGENCE=1` requests one fixed256-thread block per RHF
system for the residual maximum, density-difference RMS and conditional density
copy. Unset/`0` keeps32 threads; other spellings reject argument admission. The
request resolves active only for restricted nonmixed nonincremental quartet
Direct without physical-reference export. Unsupported domains keep32. No AO
cutoff or timing-tuned threshold is introduced.

The existing prepared native plan freezes the resolved choice and refuses a
changed choice before reusing its graphs. Python resource/restart schedule
identity also records the exact control. A separate CUDA-free private policy
declaration keeps unused declarations out of the broad RHF tuning header and
its expensive integral consumers. This is an ordinary dependency boundary;
compiler caching and complete source identity remain authoritative.

## Arithmetic and state

The32-thread specialization keeps the incumbent matrix loops and warp reduction.
The256-thread specialization uses per-thread FP64 accumulation, fixed warp trees,
eight shared sum/max entries and a fixed first-warp tree. Nonfinite physical
residuals map to positive infinity before reduction. Null or approximate-stage
residual handling remains unchanged. Lane zero runs the same current-main
physical convergence and incremental energy-refinement policy. A shared flag
and block barrier publish the conditional density-copy decision to every warp.

RMS summation order changes; bitwise RMS or identical SCF iteration counts are
not claimed. Energy, density and stationarity limits remain unchanged, including
explicit tighter requests. Retained converged density stays paired with its
physical Fock; continuing items accept the next density. Inactive systems leave
all matrix and scalar state untouched. No global scratch or allocation is added.
The global matrix reduction width, UHF, DF, energy and force-validation consumers
are unchanged.

## Focused qualification

The existing force-convergence fixture offers explicit
`--rhf-block-convergence` for standalone CUDA execution without impersonating a
Slurm allocation. It checks32/256 widths, partial matrices and768 AOs, direct and
captured launches, complete density copies, independent scalar RMS arithmetic,
null/coarse/nonfinite residuals, inactive state and current-main incremental
transitions. The original fixture mode remains available. The selected
`test_uhf --rhf-block-convergence-cache` mode checks actual cached RHF refusal
in both directions and recovery of the unchanged warm owner, without rerunning
the historical CPU/UHF oracle suite. All device execution remains pending.

The RMS fixture's finite/nonnegative arithmetic check uses a positive-sum
roundoff bound based on per-thread term count and fixed reduction depth. That
comparison tolerance never enters the production physical stopping policy.

## Advancement boundary

This is a composable component candidate, not a claim to close the complete HF
or fixed30 gap. Actual focused CUDA/sanitizer controls and one source-bound
complete endpoint off/on cold-plus-three-warm comparison decide usefulness.
Preserve failures, strict force diagnostics, independent physics gates, actual
iteration/Fock work and each warm guard; do not tune a failed fixed proposal.

References: issue #3004; `src/scf/cuda/scf_convergence_policy.cuh`;
`docs/maintainer/performance_engineering.md`.
