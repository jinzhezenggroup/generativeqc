# Decision: retire the unreferenced standalone XC gradient CUDA translation unit

Status: implemented
Date: 2026-10-06

## Problem

The build still generated and compiled `generated_xc_gradient.cu` from
`emit_native_geometry_cuda()`. Its geometry launch was the historical
`geometry_kernel<<<1, 32>>>` schedule. Repository-wide call-site inspection
found no declaration or caller of its internal `enqueue_gradient()`; current
public DFT forces use the stationary compiler/runtime owner instead.

Keeping the second implementation in every CUDA build added compilation surface
and made it possible for a future change to reconnect an obsolete single-block
force owner accidentally.

## Decision

Remove the generated translation unit from the production CMake target, remove
its generator entry point and source-identity input, and delete the private
native emitter. Keep `geometry_cuda.py` itself because the stationary owner
continues to consume its shared compiler-generated geometry mathematics.

Do not replace the dead `1 x 32` launch with a differently tuned launch. CUDA
schedule tuning belongs to the live stationary owner.

## Invariants

- Current stationary DFT force equations and runtime launches are unchanged.
- No grid, functional, precision, force, response, or acceptance policy changes.
- Shared compiler geometry/Becke mathematics used by the live stationary owner
  remains in `geometry_cuda.py`.
- Production build/source identity must not reintroduce the retired generator.

## Evidence

Repository search on master `8804b2382fbcfda1dc3daf1e0c4a9dc44c72e887`
found `enqueue_gradient()` only inside the retired emitter while CMake still
registered its generated source with `ADD_TO_TARGET`. A source-contract test
now guards all build/emitter anchors.

No CUDA build or runtime performance claim is made by this cleanup.

## References

Related CUDA schedule work: #2026, #2029, #2031.

Agent: ChatGPT
Model: GPT-5.6 Sol
