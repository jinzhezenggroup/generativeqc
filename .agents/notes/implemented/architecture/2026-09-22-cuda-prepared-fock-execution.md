# Prepared CUDA Fock execution seam

Agent: ChatGPT (Agent A — Provider / DFT integration)
Model: GPT-5.6 Sol
Date: 2026-09-22

## Decision

CUDA method consumers must not borrow a concrete Direct-J/K handle merely to
execute the Fock model already owned by `PreparedFockPlan`. A new narrow
`PreparedCudaFockBinding` plus `enqueue_prepared_cuda_fock` seam exposes
device/stream/source identity and device-resident raw J/K execution while
leaving provider selection, scientific coefficients, screening, ownership and
resource admission with the prepared Fock owner.

## First consumer

`dft::CudaKsPlan` now uses only this prepared execution seam for its
ordinary-stream J/K binding and enqueue. It no longer includes the Direct-J/K
device header, stores `CudaDirectJkPlan*`, or calls Direct-J/K stream/device
helpers. The solver-region replay identity receives the provider's opaque
owner-local source identity through the same binding.

This slice deliberately preserves the existing CUDA KS scientific admission:
only the qualified conventional Coulomb-only semilocal route is accepted by
`CudaKsPlan`. The seam itself also admits an already-prepared full-range exact
J/K owner so a later hybrid slice can consume K without another DFT-specific
Direct-J/K dependency.

## Boundary and follow-up

The first implementation is the existing full-range exact CUDA provider.
Unsupported DF/mixed-provider compositions fail closed; no fallback or provider
substitution is added. A later #930 slice may teach the same seam additional
resident providers. CUDA PBE0/B3LYP promotion remains separately gated on
semilocal scaling, K/Fock/energy assembly, convergence/final-state validation,
and endpoint evidence; this refactor does not advertise that capability.

## Integration with public DF KS (2026-09-23)

The conventional-only admission above describes the initial slice. Public DF KS
landed independently in #1073 before this branch merged. Conflict resolution
retains its qualified resident DF adapter alongside the prepared exact binding;
it does not reinterpret a DF request as exact or make the exact-only facade
admit an unsupported provider. The two bindings are mutually exclusive. Mixed
precision and the direct-J solver-region prototype remain disabled for DF.
Extending the shared facade to fitted providers remains a separate follow-up.

The integration gate runs both the native conventional KS endpoint/state tests
and the public DF KS suite with independent PySCF references through Slurm.
## Shared fitted facade completion (2026-09-27)

The prepared execution seam now admits a single resident density-fitted owner in
addition to the exact Direct owner. `PreparedCudaFockBinding` supplies the same
opaque device/stream/source identity for either implementation, and
`enqueue_prepared_cuda_fock` dispatches strict-FP64 DF J/K behind that boundary.
Mixed exact/DF provider compositions still fail closed.

CUDA KS no longer stores or borrows a `CudaDensityFittingJkPlan*`, includes DF
device helpers, or branches to DF-specific J calls in its iteration loop. It
requests raw J/K from the prepared facade and keeps only the mathematical
`FockBuildSpec` distinction needed for capability gates such as mixed
precision, solver-region qualification, range correction and nonlocal XC.

This completes the fitted-provider follow-up left by the initial seam and the
temporary #1073 integration. Future HF/DF provider scheduling and resident
execution changes can therefore reach semilocal KS through the same prepared
Fock boundary instead of a DFT-specific adapter.

Agent: ChatGPT
Model: GPT-5.6 Sol
