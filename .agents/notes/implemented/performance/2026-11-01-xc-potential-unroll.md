# Decision: bound CUDA XC potential loop unrolling

Status: implemented
Date: 2026-11-01

## Problem

NVCC fully expanded the constant-size inner point loops of the tiled potential
consumer. The measured SM120 FP64 generic and diagonal kernels each required
168 registers per thread for the default 16x16 tile. The register budget allowed
only one 256-thread CTA per SM. The explicit experimental 32x32 build required
255 registers and 176/184 bytes of stack; real launches failed with `too many
resources requested for launch` on both A100 and RTX 5090. Thread/shared-memory
admission alone had not detected this compiled resource limit.

The preceding cooperative AO-map discovery experiment did not establish a
complete-endpoint benefit and was reverted. Its source patch and all 24 original
qualifications remain in the separate rejected-evidence family; do not repeat it
solely because map comparison looks expensive in isolation.

## Decision

Limit each of the six potential point loops to an unroll factor of four. Keep
the single accumulator, ordered explicit FMA operations, finite-domain admission,
overflow/cancellation fallback, maps, points, dtype and allocations unchanged.
The production matrix tile remains 16. There is no GPU-name or architecture
whitelist. The compiler can keep fewer panel operands live while other warps
hide dependency latency.

Expand the independent CUDA contraction probe to all three existing supported
matrix schedules (8/16/32), each with FMA contraction enabled and disabled. Its
launch dimensions come from the same selected schedule as the production body.
The test accepts an actual Slurm or LSF allocation identity; ECNU qualification
never fabricates Slurm provenance.

## Invariants

- Do not split the single reduction into independent accumulators, reassociate
  the two cross legs, relax finite guards, or substitute clipping for rejection.
- Retain complete cold energy+force timing and frozen warm timing, numerical
  oracles and semantic work counts. Reduced registers are not a speed claim.
- Alternative matrix tiles remain explicit experiments until separately
  qualified. Passing the 32x32 launch regression does not promote that tile.
- Preserve source and binary identities separately from build configuration.
  A CMake tile override can change generated code without changing source identity.

## Evidence

The frozen measured parent is source identity
`b6b63f9063949ed70191c54b4d0f0b7efb788957274eb680c98f4242bbcf74f3`;
the candidate is
`ce5f498a8d173d214bc46d686256caba37836541445f6a089601a550066d6ad0`.
Both use the same unmodified scientific workload, default quadrature, SCF tile256,
force tile1024, eight reserved CPU slots and single-threaded math.

The final design uses two counterbalanced source pairs per device: candidate,
parent, parent, candidate; each process performs one cold and three frozen warm
complete E+F calls. All seven README functionals are independently qualified.
GPU4 comparisons reuse explicitly retained reference populations on the same
physical device locator, in different allocations. They do not establish a
universal ranking or a confidence interval. Raw samples and variability remain
available, including pilot changes within the practical 1% regression-review
noise tolerance.

All 21 final method/device cohorts complete both source pairs, with 336 complete
E/F calls. Every cold and warm median decreases in this sample: A100 cold
2.80–13.98%, RTX 5090 cold 0.30–3.70%, PRO 6000 cold 0.39–3.73%. The unchanged
independent gates pass; paired target/preliminary work, grids and J/K/nonlocal
work remain equal. These measured reductions include effects within ordinary
noise and are not universal regression bounds.

All three devices pass 510 independently referenced contraction cases. The
Blackwell controls additionally pass two RKS/UKS occupied-XC admission/rollback
checks against CPU oracles. Native point/ragged and matrix/spin/variational/
capture suites pass; Blackwell synccheck and memcheck each report zero errors.
The publication checkout also passes 73 host compiler/contraction tests and the
complete pre-commit suite.

Larger 120-atom controls use an explicit 8 GiB DF workspace and compare complete
cold, frozen warm, moved and frozen moved-replay endpoints. The first driver's
last `None` coordinate argument actually returned to the original geometry;
retain its original v1 label and move/return evidence. Corrected v2 populations
record each actual coordinate array and repeat the moved coordinates explicitly.
Separate successful v2 controls are required on every device. The 1/4 GiB memory
refusals are retained; increasing the test allowance does not weaken numerical
or resource-admission rules.

The remaining A100 cold gaps against retained GPU4PySCF are PBE 3.56%,
PBE0 8.84%, B3LYP 38.86%, r2SCAN-3c 32.56%, M06-2X 28.86% and WB97M-V 50.21%.
Warm PBE0/B3LYP remain 0.48%/1.93% behind. Blackwell r2SCAN-3c fresh-reference
controls explicitly condition both artifact caches before the two measured
ABBA pairs. Their original samples and the n1 cache-fill outlier remain visible.
This patch does not claim the all-functional objective is complete.

A separate build-configuration pilot of tile32 with the same unroll4 source
regresses cold 2.67–4.51% and warm 1.03–1.46% on four n1/5090 functionals.
All eight native populations qualify; their distinct CMake/generated/library
hashes distinguish the two builds. The later unsupported GPU4 command fails
before timing and remains recorded as job 15031 exit 2. See the
[rejected tile32 default](../../rejected/2026-11-01-xc-potential-matrix-tiles.md)
for the independent cache-conditioned reference controls. The subsequent n1
8x8 pilot also independently qualifies all eight populations but regresses
cold 0.16–3.05% and warm 0.07–1.99%; it is not promoted.

The publication branch includes #2910's later provenance/tuning-publication
repairs. Native implementation, CMake, runtime Python, DFT compiler and generation
tool files remain byte-equal to the frozen measurement checkout. The archive
retains that one-off boundary comparison and original identities; it does not
rewrite receipts to the publication commit or pin implementation hashes in CI.

The candidate SM120 default16 kernels require 64/56 registers and no stack.
Standalone emitted32 resource diagnostics also show 64/56 on SM120 and 48/56 on
SM80; those diagnostic compilations are not default16 endpoint evidence.

The original A100 pilot 224133 used a short source revision and was rejected by
the normal provenance gate. Its complete raw outputs are retained, never
rewritten as qualified receipts. Matched attempt 224134 built both binaries but
failed before its first completed endpoint because its source snapshots omitted
the pinned Libxc input tree. The repaired deployment copies the immutable
upstream/assets tree and records fresh complete populations. Neither failure is
hidden as a valid timing population.

## Consequences and revisit conditions

This is a portable generated-kernel scheduling change with no new runtime
selection policy or storage. The test matrix protects actual supported launch
behavior, not a fixed compiler register count. Recheck endpoint performance and
launch feasibility when changing compiler/toolkit, unroll extent, tile, panel
layout or fallback logic. Broader hardware promises require measurements on
that hardware; sampled A100, RTX 5090 and PRO 6000 results do not prove absence of
regressions on every GPU.
