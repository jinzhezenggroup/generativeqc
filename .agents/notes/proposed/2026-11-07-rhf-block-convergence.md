# Decision: opt-in one-block RHF convergence matrix walk

Status: proposed for merge; default-off candidate qualified on measured513288
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
the historical CPU/UHF oracle suite. These selected real linked controls passed
on the measured513288/e3ef core; selected racecheck/synccheck also passed. No
old integral suite or memcheck repeat is claimed.

The RMS fixture's finite/nonnegative arithmetic check uses a positive-sum
roundoff bound based on per-thread term count and fixed reduction depth. That
comparison tolerance never enters the production physical stopping policy.

## Advancement boundary

This is a composable component candidate, not a claim to close the complete HF
or fixed30 gap. Actual focused CUDA/sanitizer controls and one source-bound
complete endpoint off/on cold-plus-three-warm comparison decide usefulness.
Preserve failures, strict full-force gates, independent physics gates, actual
iteration/Fock work and each warm guard; do not tune a failed fixed proposal.

References: issue #3004; `src/scf/cuda/scf_convergence_policy.cuh`;
`docs/maintainer/performance_engineering.md`.


## Measured component evidence

The [retained three-pair evidence](../../../benchmarks/results/h100-rhf-block-convergence-20261107/README.md)
compares32/256 threads in fixed off/on, on/off, off/on execution order on the same
H100/core. All24 original independent E/F gates,12 strict full-force comparisons
and nine individual warm guards pass. Every cold executes15 molecular Focks
(8 proposal+7 strict target), and every warm executes one. No work equality was
required to admit a changed reduction trajectory.

Median paired cold reduction is2.63859281%, with median saved0.259096243s.
All six cold times are retained; the first5.273% result is not singled out as a
kernel speedup. Its wall-time saving exceeds the separate prior intrusive
component time, so timing noise/source/device differences prevent that causal
attribution. The two counterbalanced additions retain smaller gains. The
original failed-before-native purpose check remains a failed historical receipt,
with a separate reviewed metadata-only repair and fresh successful populations.

The measured source remains513288b77bb6a654cffa33c9aa3aa6b39d134c16, native013884,
coree3ef, GPUfe8. Adding this evidence does not measure the publication tree.
This supports a scoped default-off component; no new G4 comparison, automatic
promotion or original fixed30 completion follows. Keep the original campaign
and denominator requirements unchanged.

## Post-measurement master integration

Integrate master `7704dd93dec18db0cf506d78ce1c76f0d6d40fe1` after #3039's
shared-provider migration. The RHF driver and private policy changes compose
without a source conflict; only the generated Libxc qualification report and
snapshot require regeneration. Recompute their checked-source identities with
the canonical catalog tool, retaining stale status, zero qualified products,
all original measurements and unchanged acceptance gates.

The combined native input identity is
`2c113229e644417e47abaf2d30a75a579a2f8096fe08e04a3ca66831cb4d80cc`.
This is not the measured `013884` image. Focused host integration checks and
offline replay do not freshly qualify the combined GPU source or relabel the
original three H100 pairs. The control remains default-off, and normal remote
CI and merge-queue checks remain required.
