# Decision: scope packed Fock queue synchronization to its participants

Status: proposed
Date: 2026-10-24

## Problem

Complete HF graph replay under Compute Sanitizer synccheck reports divergent
warp barriers in `generated_sm120_ssss_shell_class_fock_rhf_persistent_kernel`
on the integrated CUDA 12.9.1/sm120 build. The independent two-center RHF/UHF
fixture fails before completing its baseline RHF solve. A public water3
Direct-HF energy-and-force call reproduces the same kernel failure. Disabling
the optional validation GEMM does not remove it. Ordinary execution and the
complete-HF memory check pass; neither substitutes for synchronization checking.

The earlier density-bound repair in #2530 was qualified with generic/non-AOT
integral diagnostics. This campaign also includes generated packed Fock kernels.
Its source is `121b14ab59aa5ed94c2311de634e11e745e8923a`, with core SHA-256
`ea29370dcc36673a236dc9a82d72c2f87647b2820e0e88485b16c5b1f8de07ae`.
The [diagnosis](../../../benchmarks/results/packed-fock-warp-sync-20261024/README.md)
retains the original failures and execution identities.

## Candidate

Use a full-warp barrier for packed queues whose launch width is exactly 32.
All lanes participate in publication of the shared claim and in the barrier
after each task batch, including lanes without a valid tail task. Wider queues
retain CTA barriers. The task claim width, queue contents, screening, recurrence,
scatter, precision and launch geometry are unchanged.

This matches synchronization scope to the fixed participant set. The observation
alone does not establish a particular compiler or driver defect, and the failed
checks are not classified as false positives.

The selected generic-worker candidate explicitly reconverges each full warp before the CTA
barriers at the component-lane queue's publication and completion boundaries.
The failing `ddpp` PC maps to the publication `BAR.SYNC.DEFER_BLOCKING` in the
retained CUDA 12.9.1 binary. CTA barriers remain necessary for inter-warp shared
claims; a warp barrier alone cannot replace them.

An intermediate Rys-emitter edit did not cover the failing default `ddpp`
consumer: its generated generic queue was unchanged and synccheck still failed.
That edit was reverted. Verify the exact architecture-namespaced generated
consumer before compiling; a shell manifest's recurrence label alone does not
identify its selected value emitter. The retained failed cohort is
`2bfb223cbd84ab4557d5d6ddd5b7b72cbad34ab5`; the corrected candidate is
`f8edc52a6d7fe331be55eb7b35e106f9ea72f5fb`.

The complete provider regression also exposed a pre-existing assertion on the
overall schedule label (`l=2`, Cartesian, `md-j-primitive/retained-k`): automatic
MD-J replaces that label while generated K
remains selected. The test now checks the actual K selection gate and retains
all independent ERI/J/K comparisons, spin populations and AO representations.

## Validation and limits

Eighteen focused generator, launch-metadata and streaming-screening tests pass.
The source-matched repair passes the original complete-HF synccheck (including
the GEMM-disabled control), but public water3 now reaches a `ddpp` component-lane
queue and reports 7,584 synchronization errors. This is a partial repair; component-lane
queue reconvergence and independent DF/range/force regressions remain required. The first AOT compatibility test selected the newer small-system
native force owner; its explicit packaged-consumer rerun is retained separately.
The measured repair source is `c62e9d7dce87477738311f145f1838d4ad99d538`.
Its composition also includes #2530; this is not standalone-master GPU acceptance.
No speedup or numerical-tolerance change is proposed.

## Revisit when

Extend the synchronization policy only with compiled consumer coverage for the
new participant count, partial claims and graph replay. Keep independent complete
energy/force checks and preserve failed sanitizer logs when investigating a new
backend or toolchain.
