# RKS directional first-integral CUDA execution

Status: implemented candidate for #1905
Date: 2026-10-06

## Decision

Reuse the compiler-owned directional first-integral accumulator for the
semilocal RKS nuclear RHS. The RKS consumer declares the same direct
all-electron Cartesian source mathematics as the CPU path: overlap, kinetic,
nuclear attraction and the Hartree four-center density contraction. Direction
contraction, density weighting, Cartesian normalization and AO matrix
accumulation execute on CUDA; only the final H1(v) and S1(v) matrices are
published to the current response boundary.

No PBE-specific kernel or duplicate Hessian equation is introduced. The
existing CPU path remains the default. CUDA selection requires an explicit
CudaCompilerAdapter and never falls back to the CPU first-derivative evaluator.

## Boundary

This is one #1905 slice, not end-to-end CUDA RKS Hessian completion. Native
semilocal XC mixed geometry response, CPKS vector/reconstruction residency,
response-weight first-integral relaxation, nuclear/final HVP assembly and
public Calculator CUDA Hessian capability remain separately gated.

The real-device qualification compares the complete HVP with the existing CPU
scientific owner while monkeypatching the CPU directional first-integral path
to fail if it is invoked. The full-Hessian real-GPU gate additionally threads
both CUDA first-directional and second-integral selections.

Agent: ChatGPT
Model: GPT-5.6 Sol
