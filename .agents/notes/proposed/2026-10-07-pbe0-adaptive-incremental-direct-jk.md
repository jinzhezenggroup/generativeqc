# CUDA KS: admit incremental Direct-J/K only after late-SCF density contraction

Status: proposed; implementation and regression coverage are in the associated PR. No default promotion or performance claim.

## Motivation

The retained 96-atom exact-direct PBE0/def2-SVP composition spends about 221 s of its 253 s cold endpoint outside the force endpoint, while compile/cache setup is only about 3 s. It is the first qualification workload, but the implementation is deliberately not a PBE0-specific algorithm.

The older shared incremental CUDA J/K experiment (#1803) showed a useful K delta-work collapse but no endpoint win and a moved-geometry convergence regression. Current master already routes delta density through the shared shell-pair density bounds and quartet compaction, so this follow-up does not add another J/K implementation. Instead it connects the shared controller to CUDA KS and prevents screened delta builds from being used while the accepted density is still changing substantially.

## Operator-level policy

Incremental execution is admitted by the resolved Fock-provider contract rather than by a named functional. The common policy:

- requires strict-FP64 exact Direct J/K,
- rejects conflicting mixed-precision iterative Fock and density-fitted providers,
- keeps a density-screened lower to at most one accepted delta update before a full refresh,
- derives screening cadence from the immutable prepared provider rather than a caller-supplied option copy,
- performs strict full-density physical finalization.

Only the linear Coulomb and exact-exchange terms consume the anchor-relative density:

`J[D] = J[D_anchor] + J[Delta D]`

`K[D] = K[D_anchor] + K[Delta D]`

Semilocal XC, meta-GGA state, VV10/rVV10 and other nonlinear density consumers continue to evaluate the full current density. Compatible range-separated exact exchange follows the same delta-density rule.

This makes the mechanism applicable to any CUDA KS composition whose resolved operator graph uses a compatible exact Direct provider, including pure semilocal RKS/UKS (incremental J), global hybrids (J plus full-range K), and compatible RSH/nonlocal compositions (J plus exact SR/LR K). A named functional does not receive special numerical behavior.

## Late-SCF gate

The optional positive density-RMS threshold is evaluated on the preceding accepted density update. For a density-screened CUDA lower, the first build is full and a later build is also full while the previous accepted density RMS is above the threshold or nonfinite. Only late-SCF iterations at or below the threshold may use Delta D. A zero threshold preserves cadence-only behavior.

Actual full/delta build counts are published through the ordinary DFT result adapter. Strict-FP64 incremental owners also reserve the bounded final-closure history before retained-host resource bytes are reported.

Incremental trajectories currently bypass the CUDA KS chunk/Graph replay route. Capturing the anchor controller inside that route requires separate qualification; disabling it here preserves one execution owner and avoids claiming graph compatibility prematurely.

## Benchmark controls

Generic controls:

```text
GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK=1
GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL=<unsigned>
GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD=<nonnegative finite double>
```

The historical PBE0 spellings remain accepted as compatibility aliases for retained experiments:

```text
GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK=1
GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL=<unsigned>
GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD=<nonnegative finite double>
```

Production defaults remain unchanged.

## Qualification plan

PBE0 remains the first performance gate because retained cold/warm/moved evidence already exists. Use one frozen source/library and the complete 48/96-atom PBE0/def2-SVP protocol. Retain cold, five warm, moved and five moved-warm endpoints, exact vectors, actual SCF histories and incremental work diagnostics. Compare:

1. ordinary control,
2. cadence-only incremental control (RMS threshold 0),
3. adaptive thresholds selected before timing (initial sweep candidates: 1e-2, 1e-3, 1e-4).

Do not normalize by SCF iteration count and do not attribute a cold difference when the trajectories differ. Promotion requires a reproducible complete-endpoint benefit without a moved/convergence regression; otherwise retain this as negative evidence.

Before broad default promotion, numerical parity should also cover representative pure semilocal RKS/UKS, another global hybrid, and a range-separated/nonlocal composition. Those tests qualify composition breadth; they do not require separate named-functional implementations.

## Current validation scope

Source contracts guard capability-based admission, full-density XC/nonlocal evaluation, exact J/K delta routing, provider-owned screening cadence and strict final full builds. Native CUDA exact-exchange tests exercise RKS and UKS, permissive versus blocked density-RMS gates, real full/delta build counters, endpoint parity, native screening-option mismatch, and bounded final-closure resource accounting.

Fresh GPU endpoint timing is still required before any default decision.

Agent: ChatGPT
Model: GPT-5.6 Sol
