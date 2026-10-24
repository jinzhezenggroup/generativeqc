# Decision: compose explicit HF refinement with the newer KS defaults

Status: implemented
Date: 2026-10-24

## Problem

PR #2530 at `ab557d5b5485b4720a9aff5bba24cf9cc39f6ebf` predates
master `fc854745e14eb9a21734372a86403e2491a8e5d4`. Both touched the
incremental controller and convergence launch ABI. Master also promoted a
separate bounded KS proposal policy. Resolving conflicts by choosing either
side would lose HF energy-baseline reset, ordinary RHF owner qualification,
or the newer qualified KS behavior.

## Decision

Keep the PR's explicit screened fixed anchors and physical-stationarity-driven
full-energy transition. This supersedes master's density-only HF transition:
only the convergence controller retires deltas, clears the reconstructed energy
baseline, and requires two fresh full evaluations within the original budget.
The existing qualified-owner argument remains in the convergence launch ABI;
ordinary qualified RHF still uses its current stationarity limits, while
incremental HF and UHF keep their strict physical gates.

Keep automatic KS on its qualified rolling anchors and one-delta ordinary
cadence. Bounded Standard KS proposals also keep rolling anchors, their finite
proposal limit, and their independent full-density final-state proof. Only the
explicit controller outside those proposals uses the PR's fixed anchors and
requested cadence (including zero). Tests execute these distinct controller
routes, including actual anchor updates and refresh behavior. The two KS
policies are selected by their existing ownership/admission state, without
adding another allocation or a public selector.

Preserve newer native cold determinant provenance, fleet ownership, and work
accounting. The incremental warm shortcut still requires an entirely resident,
provenance-matched bucket and zero actual delta builds. External/moved seeds,
actual deltas, exported references and explicit final rebuilds retain canonical
finalization. Incremental HF remains experimental and off by default.

## Density-bound synchronization repair

The composed Release/sm120 binary reproduces the historical complete-HF
synccheck failure: 96 errors at `reduce_shell_pair_density_bounds_kernel<false>`
`+0x1250`. Its SASS identifies that PC as the first `BAR.SYNC.DEFER_BLOCKING`
after the ragged AO walk. Plain and ordinary captured standalone reductions
pass with the same source, so those tests alone do not qualify the complete
HF graph replay. Adding a warp join before the same CTA barrier still fails
at the shifted barrier PC `+0x1260`; do not repeat that unsuccessful repair.

The HF graph owner launches one complete warp. Use `__syncwarp()` for its
shared-memory handoff and reduction steps; keep `__syncthreads()` whenever
the block has a different width. Every live lane executes the fence outside
the lane-dependent update. The exact-one-warp condition is block-uniform.
This provides the required memory ordering without a redundant block barrier,
while preserving multi-warp consumers, launch geometry, arithmetic and the
screened task set. No generated scientific formula changes and no second
reduction implementation are introduced. The existing scientific ownership
classification is retained; its six added code lines are synchronization
selection, not additional integral mathematics.

New native controls compare actual captured reductions with independent scalar
shell-block maxima for both spin counts, 32/128/256 threads, tiny and through-g
shell blocks, partial/ragged walks, and active/inactive neighbors. The unchanged
complete incremental HF executable is the before/after failing-to-passing
sanitizer regression. This establishes the retained repair's behavior, not a
claim that the old diagnostic was a false positive or proof of a particular
CUDA compiler/driver defect.

## Evidence and limits

The original Slurm job 8706 failure log has been recovered byte-for-byte. It
reports 96 divergent-warp errors at the density-bound kernel offset `+0x1250`
in the candidate library's ordinary first full build. That invocation was not
a clean-master control. It must not be called a false positive based only on
uniform source-level barriers. The recovered log does not record the exact
command line or tool version; newly recorded invocations are separate evidence.

See the [integration receipt](../../../../benchmarks/results/incremental-hf-integration-20261024/README.md)
for the recovered log, source-matched validation and its limits. Historical
endpoint timings and failed cohorts remain unchanged. This integration makes
no new speedup claim and does not promote incremental HF defaults.
