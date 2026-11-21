# Decision: Keep the XC response finite at subnormal spin density

Status: implemented
Date: 2026-11-21

## Problem

Neutral doublet H3 at `[(0.13,-0.21,0.17), (0.43,0.19,2.1), (0.31,0.08,4.0)]`
Bohr, spherical STO-3G and PBE0 UKS converged the CUDA SCF but rejected dense
Hessians and HVPs on the `8/4/8` and `8/8/16` product grids. `12/4/8` and
`20/12/24` passed; disabling the dense CPKS cache did not fix the failing
grids. On Slurm 2101 a temporary trace isolated a moving-grid XC point with
positive spin densities of about `3.83e-316` and `4.34e-317`. The physical SCF
potential was valid, but the exchange response JVP formed `1/rho` before
multiplying by its finite direction. This overflowed although the derivative
of the point potential is representable.

## Decision

Generate the first-order cube-root JVP from the implicit residual
`input - value^3 = 0`, binding `value` to the same `cbrt` primal. The resulting
denominator `3*value^2` is normal and finite for every nonzero finite FP32/FP64
input. This preserves the physical model and avoids an overflowing density
reciprocal. Do not floor densities, skip tail points, change the quadrature,
or relax the finite-output or Hessian publication gates. Exact vacuum still
admits only its defined directions. The temporary trace was removed.

## Evidence

- The snapshot point fails on the old exchange-only host JVP and passes with
  the generated fix. The regression checks the exact UKS snapshot and an
  equal-spin analogue against 550-digit derivatives of the original PBE
  energy, for exchange, correlation, and their PBE0 sum. For the subnormal
  Cartesian gradient outputs the gate accounts for rounding in the existing
  high-reduced-gradient numerator before its final denominator scaling.
- Slurm 2102, RTX 5090: both formerly failing grids publish dense and HVP
  results with no numerical host fallback, displaced SCF, or intermediate panel
  downloads. The `8/4/8` FP32-proposal cache passes all nine FP64 residual checks
  and differs from the FP64-cache Hessian by at most `2.57e-14 Ha/Bohr^2`.
  Raw asymmetry and the largest translational residual are below `4e-16`.
- 44 scheduled GPU tests passed, including Python/CLI agreement for this
  H3 fixture and existing direct/DF RKS/UKS, AUTO, and FP32-proposal tests.
  122 host tests passed; the 25 device-only cases in that host run were skipped.
- The independent Libcint/Libxc UKS oracle reconverges total energies at 325
  geometries, including the center, with the same finite product-grid rule and
  molecular partition rebuilt after each displacement. Richardson extrapolation
  of central mixed and diagonal stencils at `0.003` and `0.0015` Bohr gives a
  maximum `8.83e-10 Ha/Bohr^2` Hessian error and `2.03e-10` HVP error (gate
  `3e-8` each). The ignored local receipts, replay script and sccache 0.16.0
  before/after statistics are under `build/hessian-gpu-sm120/h3-pbe0-*`.

## Revisit when

Revisit final-product scaling if a supported endpoint needs relative accuracy
for subnormal Cartesian gradient coefficients, rather than the absolute
roundoff bound used here. A positive finite density with a representable
direction must not be reclassified as vacuum to make that case pass.
