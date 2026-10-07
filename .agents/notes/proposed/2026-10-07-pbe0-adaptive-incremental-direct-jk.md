# PBE0 cold SCF: admit incremental Direct-J/K only after late-SCF density contraction

Status: proposed; implementation and regression coverage are in the associated PR. No default promotion or performance claim.

## Motivation

The retained 96-atom exact-direct PBE0/def2-SVP composition spends about 221 s of its 253 s cold endpoint outside the force endpoint, while compile/cache setup is only about 3 s. The dominant cold opportunity is therefore repeated SCF Fock work rather than another force/JIT micro-optimization.

The older shared incremental CUDA J/K experiment (#1803) showed a useful K delta-work collapse but no endpoint win and a moved-geometry convergence regression. Current master already routes delta density through the shared shell-pair density bounds and quartet compaction, so this follow-up does not add another J/K implementation. Instead it prevents screened delta builds from being used while the accepted density is still changing substantially.

## Policy

The existing shared policy still:

- requires exact Direct J/K,
- rejects conflicting mixed-precision iterative Fock,
- keeps the screened lower to at most one accepted delta update before a full refresh,
- performs strict full-density physical finalization.

This change adds an optional positive density-RMS threshold. For a density-screened CUDA lower, the first build is full and a later build is also full while the previous accepted density RMS is above the threshold or nonfinite. Only late-SCF iterations at or below the threshold may use ΔD. A zero threshold preserves the previous cadence-only behavior.

The threshold is part of CUDA bucket execution identity because it is captured as a graph kernel scalar. Actual full/delta build counts are retained on device and downloaded; diagnostics no longer infer the build split from the nominal cadence.

## PBE0 experiment boundary

A benchmark-only DFT selector is intentionally narrower than the public API:

- `GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK=1`
- optional `GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL`
- optional `GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD`

It accepts only strict-FP64, exact-direct, CUDA, restricted PBE0. Production defaults remain unchanged.

## Qualification plan

Use one frozen source/library and the existing complete 48/96-atom PBE0/def2-SVP protocol. Retain cold, five warm, moved and five moved-warm endpoints, exact vectors, actual SCF histories and incremental work diagnostics. Compare:

1. ordinary control,
2. cadence-only incremental control (RMS threshold 0),
3. adaptive thresholds selected before timing (initial sweep candidates: 1e-2, 1e-3, 1e-4).

Do not normalize by SCF iteration count and do not attribute a cold difference when the trajectories differ. Promotion requires a reproducible complete-endpoint benefit without a moved/convergence regression; otherwise retain this as negative evidence.

## Current validation scope

Host policy and bucket-identity tests cover the new option. The CUDA regression requires the gate to replace early delta builds with full builds, preserve the final state, reconcile actual build counts, and rebuild the cached graph when the threshold changes. Fresh GPU endpoint timing is still required before any default decision.

Agent: ChatGPT
Model: GPT-5.6 Sol
