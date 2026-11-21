# Decision: Stream canonical nuclear J/K sources when the direction cache is too large

Status: implemented
Date: 2026-11-21

## Problem

For real-spherical HF/cc-pVTZ (44 AO, 16 shells), the strict FP64 r²SCAN RKS
Direct SCF converged, but the F-z unit HVP did not publish within six minutes
on an RTX 5090 (Slurm 1704). A bounded GDB sample after SCF (1708) caught the
host launching kernels in `CudaWb97mvNuclearJk::apply`, not in CPKS. The complete
ordered group schedule fit, but its canonical last-direction pool would need
about 186 MiB of jets before alignment, above the persistent 64 MiB cap. The
planner therefore lost both eightfold source sharing and batched launches.

## Decision

Try the existing canonical-source tiled planner when requested last-direction
residency fails to fit, even if the complete ordered stream is already legal.
Replace that stream only after all windows are admitted and reduce the primitive
component work. Keep an admitted persistent direction so frozen and relaxed
actions can share it. If either the source/work/arena limits or the required
batch bindings preclude tiling, keep the legal ordered stream; if neither is
legal, fail before execution. Do not raise any budgets.

## Invariants

Only one canonical window occupies the same charged integral arena at a time.
The persistent direction cap is unchanged. Transient windows do not count as
direction hits and rebuild on a subsequent action. The full ordered closure,
signed spin-density contractions and Cartesian/center slot permutations are
unchanged. Ordered accumulation inside a window is deterministic; summing
windows can differ from global ordered streaming at FP64 roundoff. Independent
Libcint/Libxc and raw endpoint checks remain the acceptance authority.

## Evidence

- `r2scan-rks`, Direct, real spherical, Bohr, the same 16×6×12 explicit moving
  grid, RTX 5090 (SM120). Slurm 1710 published the cold HVP in **51.14 s** wall
  time (2.74 s SCF, 3.44 s response preparation, 44.42 s action). The prior
  measurement is censored at six minutes, not a completed timing sample.
- The two exact actions used 14 windows each: 9,316 canonical shell groups,
  8,103,728 primitive component evaluations and 35,794 integral launches
  measured in total. The same topology's complete ordered plan has 65,536
  groups and 26,873,856 primitive components **per action**, by shell-product
  enumeration. The optional persistent pool remained empty and the endpoint
  reported zero numerical host fallbacks, displaced SCFs or runtime compiles.
- Independent PySCF 2.14.0 / Libcint / Libxc 7.0.0 energies reconverged on the
  same moving grid. The HVP's maximum unit-column error is `4.33e-7 Ha/Bohr²`
  for the independent 0.002/0.001 Bohr stencil; the raw translation residual
  is `1.60e-14 Ha/Bohr²`. The center energy differs by `2.84e-14 Ha`.
- Slurm 1711 published all 36 entries in **141.35 s** wall time. The six
  frozen J/K columns used 24,311,184 primitive components and 107,382 integral
  launches; all six relaxed columns reused the frozen derivative panels.
  The raw symmetry residual was `5.83e-11 Ha/Bohr²`, the translation residual
  `1.12e-14 Ha/Bohr²` and the separate HVP difference `4.24e-13 Ha/Bohr²`.
  Three independently reconverged relative Cartesian energy stencils at 0.008,
  0.004 and 0.002 Bohr gave a maximum full-matrix error of `1.78e-7 Ha/Bohr²`.
- The default level-3 Treutler/Lebedev moving grid also passed on Slurm 1714:
  **51.63 s** for the unit HVP with the same exact J/K work, zero numerical host
  fallbacks and `5.55e-8 Ha/Bohr²` maximum error against the independent
  three-step energy reference.
- Slurm 1713 passed 18 native component regressions. They compare resident,
  canonical tiled and ordered streaming output against independently displaced
  values for RKS and UKS, and test source-cap refusal with an admitted ordered
  fallback. The host closure/arena checks, eight host CLI admission checks and
  Sphinx warnings-as-errors build also passed.

## Revisit when

Large f-shell endpoints are still limited by raw generated-integral work even
when their launches and symmetry reuse are bounded. Revisit partial adjoint
sources or native lowering when complete timings and independent references
show benefit, rather than increasing the persistent direction pool.

## References

- [Earlier f-shell qualification](../numerics/2026-11-21-cuda-r2scan-direct-f-hessian.md)
- `build/hessian-gpu-sm120/r2scan-orbital-f/` (local GPU logs, checked basis,
  inputs, native receipts, independent energy script and compiler-cache stats)
- Tested DSO SHA256:
  `e6b3c9f9bef5a0380ec812e0f4d1ab620053ad20a16f10bf74ffc813b8d242d3`
