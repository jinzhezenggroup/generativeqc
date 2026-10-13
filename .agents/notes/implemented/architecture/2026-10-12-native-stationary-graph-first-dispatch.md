# MethodIR-first native CUDA stationary dispatch and build inventory

Status: implemented as a non-numerical, no-equation-change refactor
Date: 2026-10-12

## Problem

Native CUDA first derivatives already match the full compiler MethodIR
semantic key (point family/domain, spin, X/C coefficients, exact exchange,
Becke iterations), but packaging maintained a second handwritten list of ten
profile names and invoked the entire compiler/codegen stack separately for
each profile. The prepared C++ owner then converted its exact graph match
back to a diagnostic string and looked the graph up again.

These are independent sources of duplicated work and name-coupling. They are
not a justification for ten independent scientific CUDA runtimes.

## Change

- Derive the build's default stationary AOT inventory from the compiler
  discovery command. Users can still select a smaller explicit inventory;
  absence of a compiled graph remains an explicit unsupported context.
- Generate all selected native wrappers in **one compiler process**, reusing
  MethodIR lowering caches while preserving exactly the prior CUDA source
  for every selected graph. The shared s/p primitive inventory remains one
  object library.
- Keep native preparation, descriptor resource sizing, and controller
  construction on the **same immutable graph binding**. Do not round-trip
  through a method/profile name after admission.
- Assert exact byte identity against single-profile codegen, deterministic
  regeneration, no duplicate input profiles, and C++ graph-first dispatch.
- Retain the existing real-GPU E+F method matrix and static-link checks,
  but generate expected linked names from the compiler rather than another
  handwritten set of ten strings.

## Scientific and performance boundary

This refactor does **not** yet collapse distinct XC/geometry fused device
bodies into a single runtime-parametric CUDA kernel. LDA/GGA/meta-GGA have
different jet layouts; hybrid XC scaling and K terms alter reduction semantics.
Sharing these bodies needs new numeric capacity, independent forces/finite-
difference qualification, and cold/warm benchmarks across spin and basis.
No unvalidated generic execution path is promoted or silently substituted.

The resulting CLI remains standalone and Python-free at runtime. Python
remains a build-time compiler only. No release or merge is implied.
