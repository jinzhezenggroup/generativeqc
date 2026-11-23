# Decision: Bind canonical CPU PBE UKS to the native stationary Hessian

Status: implemented
Date: 2026-11-23

## Decision

Link the exact bare PBE UKS program to the existing prepared CPU Hessian/HVP
owner. Keep separate physical alpha and beta XC fields and unit exchange and
correlation scales; the J-only Fock response contracts the sum of the spin
densities. The same point-domain binding selects fixed-grid JVP, moving-grid
RHS and mixed curvature. This shares the polarized physical point machinery
with PBE0 without inheriting its 0.75 semilocal exchange scale or Fock exchange.

The model gate checks the binding's actual spin, exact Coulomb and absent
exchange rather than assuming a J-only model is restricted. Packages without
the binding refuse rather than selecting the Python RKS controller. Restrict
this qualification to all-electron Direct FP64 with scalar columns; recycled
groups, UKS corrections, DF and ECPs do not inherit the new binding.

## Validation

From `7969f9bfa` plus this slice, independent PySCF UKS starts from its own
guess at each displaced geometry on the same discrete grid. The distorted
water cation tests compare energy, spin-resolved D/W, J and XC components,
forces and all nine raw columns. The coarse `1e-3` Bohr stencil differed by
`2.02e-6`/`2.50e-6` Hartree/Bohr² (Cartesian STO-3G/spherical def2-SVP).
Refinement to `3e-4` reduced both errors by about an order of magnitude;
`1e-4` gave `2.02e-8`/`1.65e-7` Hartree/Bohr² without changing the raw-column
tolerance. The native matrix is not symmetrized and agrees with the native
HVP and translational sum rule.

The SDK/CLI/Python installed comparison uses spherical HeH, including exact
and one-byte-short capacity, moved warm replay, stale geometry refusal and
recycled-schedule rejection. The native subprocesses have no Python or compiler
on their runtime path. The full CPU CTest suite passed; no CUDA claim is made.

Five post-warm-up paired complete endpoints with identical outputs and work
counters give HVP/dense changes of `+1.69%/+0.15%` for PBE RKS and
`+1.04%/+0.15%` for PBE0 UKS. Bare PBE UKS has no previous native timing
baseline. It took `0.452/2.488 s` with 16 SCF Fock builds, 26/222 induced Fock
actions and 56,160/479,520 response XC point visits. Source logs, benchmark
commands, hashes and work receipts are retained under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-pbe-uks`.
