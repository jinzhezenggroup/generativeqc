# Decision proposal: capability-based HF shared-x/y admission

Status: proposed; host implementation does not confer endpoint qualification
Date: 2026-11-08

## Problem

The previous automatic policy identified exact SM80. Architecture identity is
neither a scalar-FP64 throughput class nor evidence that a compiled companion is
available. An ISA whitelist also blocks independently qualified future profiles.
The user explicitly rejected that design.

This supersedes the architecture-specific dispatch proposal in
`2026-11-08-sm80-shared-rys-xy-qualification.md`, not its historical evidence or
the independent HF48 accuracy failure.

## Decision

Separate three inputs:

- Compiler availability follows the selected manifest's unweighted capability
  and validated target plan, without a second architecture whitelist.
- Automatic runtime selection requires the current visible device's reported
  FP32:FP64 scalar-throughput ratio in `(0, 2]`. This is a conservative
  full-rate-FP64 class, not a matrix-acceleration or GPU-name predicate.
- Both automatic and explicit requests require compiled capability, 32-lane
  warps, and resources for the fixed family's largest 192-thread block. The
  generated profile still owns per-kernel register/shared-storage legality.
  Existing method, precision, storage, output and operator restrictions remain.

Unknown/reduced-rate throughput keeps the incumbent automatically. Explicit
`1` bypasses throughput profitability only, not kernel availability or legality;
explicit `0` remains a rollback. Device facts are queried afresh, including the
throughput ratio, with the existing full-property provider fallback. No global
ordinal cache, GPU timing calibration, CPU scientific work, or new control is
introduced. Weighted-Z has separate scientific qualification and is unchanged.

## Rejected alternatives

- Renaming the SM80 gate to an A100/product-name gate preserves the same flaw.
- Enabling every compiled companion ignores low-rate FP64 and the RTX5090 guard.
- Treating full-rate FP64 as universal performance evidence is also incorrect.
  It is an admission policy, not a claim that an unmeasured device is faster.

## Evidence and remaining gates

Executable host tests vary ISA identity independently of throughput, exercise
unknown/reduced-rate resources and explicit overrides, and preserve unavailable
capability fallback. Synthetic compiler inventories verify a declared unweighted
capability is not vetoed by ISA name; they do not qualify those targets.

Fresh real-device resource queries and complete A100/RTX5090 endpoint checks
are distinct gates. Retain the unchanged 3% paired
warm guard and independent E/F gates. Historical SM80-image timings must not be
relabeled as measurements of this change. H100 has no new endpoint evidence.

Host validation: 81 tests pass, one opt-in CUDA test is skipped by pytest;
standalone real-device qualification is separate. Compiler ownership checks
cover 587 modules with zero errors. The host matrix varies major/minor identity
independently of ratios -1/0/1/2/3/32/64, checks 191/192-thread resource boundaries,
and verifies explicit requests cannot override missing capability/resources.

The standalone production device-facts owner is compiled through verified
sccache 0.16.0 / CUDA 12.9. Slurm16822 passes the full-property oracle on the
allocated RTX5090, including repeated/invalid-ordinal/device-preservation
checks; its observed ratio is 64. LSF229194 on gpu22 likewise reports DONE/exit0
for the same source's dynamically linked probe, with finite five-minute
exclusive GPU allocation. Its stdout was not recovered from the login-local
output path; the scheduler success is retained but no raw A100 ratio or endpoint
timing is inferred. CPU transport229196 does not repeat the GPU calculation.

Frozen query-source archive SHA256 is
`302bd47bbc0c5f8898608da9d74c266a718f6bbf1c3bc287cd0b4f760bfbcc04`.
Slurm's statically linked probe ELF is
`b8fe35ef56a630dae9877481c8c28bf9f3726624db5ddf91c2f3f19f28041ce3`;
LSF's dynamically linked probe ELF is
`bdf4103b2a130afa7d1b167e87e688a8dd8970b7e38b81541e94a70a4a9832de`.
These are resource-query executables, **not libgenerativeqc or molecular endpoint
qualification**. Source/logs are retained in the ignored task artifacts. The
revised complete endpoints and default HF48 independent failure remain open;
the PR stays draft. Latest integrated master is b6c20e73b, not the historical
measurement base.

## Post-measurement master integration

Masterfea412df1 introduces the canonical `CudaTargetInfo.fp32_fp64_ratio` with
clock and memory-bus facts. Migrate this branch's policy and fixture consumers
directly to that field, rather than retaining a second ratio or a compatibility
alias. Preserve master resource-query fallbacks and both new missing-attribute
cases, plus the branch's repeated/current-visibility ratio variation checks.
This is still capability/throughput/resource selection, not an ISA whitelist.

The historical complete-ring DIIS experiment at source682561e9/ELFc368db4b is
rejected on both scheduled cards and removed; its full observations are at
`benchmarks/results/hf-fp64-predictor-rejection-20261108/`. That source predates
this master integration. Its HF96 science/performance observations are not
current-merge, HF48 or full-README qualification. Keep the PR draft and retain
the independent default HF48 failure and original acceptance gates.

The clean merge index passes101 focused host checks with one opt-in GPU skip;
no new GPU endpoint is inferred from these host checks. Regenerated Libxc
metadata preserves historical measurements as stale, not freshly qualified.

## References

- Issue #2976 and draft PR #3048.
- `benchmarks/results/hf48-control-isolation-20261108/` preserves the unresolved
  default HF48 force-accuracy diagnostic and coupled convergence controls.
