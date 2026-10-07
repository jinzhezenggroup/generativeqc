# Decision: use strict typed PTX lowering for the bounded CuMetal QC lane

Status: implemented
Date: 2026-10-06

## Problem

After aligning the runner with CuMetal's AIR deployment target, the first RHF
endpoint dispatched its primitive-pair-cache and initial-state kernels on the
Apple GPU. The next production one-electron `thread_pairs` kernel failed in the
same pinned provider's legacy PTX-to-LLVM path: `ld.param.b64` reported an unknown
parameter slot. This is a compiler ABI limitation, not a failed numerical oracle.

## Decision and precision boundary

Select the same pin's documented `CUMETAL_PTX_BACKEND=cumetal-ir` only for routine
and full QC endpoint execution. This routes PTX through the strict typed GPU IR
and MSL backend; unsupported input fails without retrying the legacy compiler.
Specify `CUMETAL_FP64_MODE=fast48` explicitly, retaining the runtime's previous
default. Give the JIT cache an explicit backend/precision suffix in addition to
the toolchain identity. The provider also keys compiled kernels by backend and
FP64 policy. Native build/toolchain caches are unchanged.

This is not a full-FP64 lane. Core arithmetic retains raw binary64 storage and
paired-FP32 fast48 software operations (approximately 48 significand bits with a
binary32 exponent envelope). The provider's double transcendental/libdevice
calls, including exp/log/erf, evaluate through binary32 under every FP64 mode.
That documented limitation already exists in the selected legacy compiler:
`lower_to_llvm.cpp` explicitly decodes double libdevice arguments through
`f64_slot_to_f32` and re-encodes the result. Typed MSL records the same limitation
as a semantic caveat. Selecting `ieee64` would not fix these transcendental calls.

Keep all scientific endpoint selectors, CPU oracles, convergence/energy/force
checks, per-case Apple-GPU provenance, and execution budgets unchanged. Passing
those gates can qualify this bounded reduced-precision provider lane only;
it does not establish general double-precision scientific coverage. Do not
loosen tolerances or substitute a CPU workload if the typed route still fails.

## Evidence and limitations

Run 37402174439 at 725d4aa81a086ef98cbda8b44be00167464e82c7 passed toolchain
preflight, native build, and the runtime CTest. Its retained QC XML recorded two
successful generic PTX Apple-GPU dispatches, followed by the one-electron
lowering failure at PTX line 112965. The kernel translation unit was 4,292,429
bytes. The typed route is a supported bounded compatibility attempt; local
configuration tests cannot establish that this real endpoint completes. Normal
CuMetal CI remains the required acceptance gate.

The pinned `lower_to_metal.cpp` returns immediately on typed compiler failure
and marks successful typed output as matched, so registration has no automatic
legacy retry. Typed f64 core operations call `cm_fp64_fast_*` helpers backed by
the VF64 Pair arithmetic module, rather than converting all arithmetic to f32.

## Revisit when

The pinned provider fixes legacy parameter-slot lowering, changes its precision
contract, or expands typed ABI support. A further provider gap should be reported
explicitly rather than hidden with endpoint exceptions or compiler rewrites.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/37402174439/artifacts/11385558387
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/docs/compiler-architecture.md
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/docs/fp64-policy.md
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/compiler/ptx/src/lower_to_llvm.cpp
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/compiler/ptx/src/lower_to_metal.cpp
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/compiler/metal/support/cumetal_fp64_inline_support.metal
