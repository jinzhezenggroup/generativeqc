# Decision: canonical frozen-density native CPU semilocal curvature

Status: implemented
Date: 2026-11-12

## Problem

Actual nuclear S'/F'/D'/W' and mixed AO/grid directions still do not supply the
frozen semilocal energy's second nuclear source. Reusing a right-density feature
JVP in this component would silently count density relaxation again when the
stationary Hessian adds D'/W' contributions.

## Decision

`HostGgaNuclearCurvature` owns a bounded full-grid source over the authenticated
mixed geometry reader and arbitrary immutable symmetric total-RKS or spin-UKS
density. The shared compiler AO-feature mixed JVP now exposes its right-density
policy explicitly; frozen consumers disable that response, while existing
stationary consumers retain it. All repository callers migrate directly to the
canonical `feature_mixed_program`; there is no obsolete private alias.

The exact MethodIR/spin binding explicitly identifies a linked frozen-energy
mixed leaf. It composes unchanged production ingredient gradients/Hessians with
Cartesian ingredient mixed AD and the shared multiplication jet for quadrature
motion. Native code owns buffers, iteration, admission and compensated FP64
accumulation, never a second XC equation. All four scalar outputs stage until
final owner authentication; finite failures and reentry release ownership for
recovery. Source storage/map/point/AO/pair caps are mandatory and separate from
borrowed arrays, stack, allocator/RSS and complete endpoint timing.

## Rejected alternatives

- Including dD in a frozen-energy mixed source: double-counts stationary response.
- Native handwritten spin/gradient Hessian factors: duplicates the canonical
  ingredient/product AD and risks losing cross-spin terms.
- Numerical production differences or reference-engine delegation: oracle only.
- Calling scalar AOT packaging a Hessian capability: integral/ECP second sources,
  metric/density relaxation assembly, nuclear/D3 terms and endpoint admission
  remain separate.

## Evidence and consequences

Independent libcint/Libxc primal energy differences at multiple steps exercise
both spins, normalized Cartesian/spherical s/p/d/f, actual grid versions 1--4,
and complete default grids including signed atomic measures. Boundary controls
exercise explicit caps, identity/shape/density refusal, late owner/reentry,
transactional recovery, empty spin, mixed symmetry/scaling, translation, tile
reuse and a Python-free native executable. This is fixed-density source evidence,
not reconverged molecular Hessian, installed TS or representative Ni/I/Cs endpoint
qualification. See the signed-measure note for the full-grid admission diagnosis.

The initial empty-spin positive-control assumption was incorrect for the shared
full ingredient-Hessian contract: the existing host response explicitly refuses
nonvacuum empty-spin points. The new consumer preserves that refusal and keeps a
transactional negative control, alongside positive-density bilinear controls.
A separately derived restricted frozen-path limit could be finite, but is not
claimed or synthesized by multiplying divergent ingredient partials by zero.
