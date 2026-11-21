# Decision: Compress only the optional repeated-column CPKS projection

Status: implemented
Date: 2026-11-21

## Decision

Offer `fp32-proposal` as an explicit dense CUDA Hessian option, separate
from SCF AUTO and off by default. Build the same projected operator by
physical FP64 unit rotations, but encode each resident column as FP32.
Reuse the not-yet-live weighted-density panel to stage one FP64 source
column. The matvec converts coefficients at point of use and accumulates
in FP64; nothing in the physical Fock or derivative features is narrowed.

Use the cached solve only to propose rotations. At most twelve iterations
and at most half the requested limit belong to this phase. If it does not
converge, use a zero seed and the remaining physical iteration budget.
Otherwise begin physical GMRES with a fresh residual for the proposed seed;
calculate tolerance from the original RHS, not the correction RHS. Audit
the residual again with the reconstructed derivative density, and do not
publish any column on failure. Both formats charge their actual cache bytes
to the same response and aggregate caps. An unavailable optional cache
uses uncached physical GMRES. Do not allocate it for one-shot HVPs.

Directly publishing an FP32 projected solution was rejected because its
algebraic residual does not bound the physical stationarity residual. Keeping
both a compressed and an exact matrix would negate the storage objective.

## Evidence

- Slurm 2097 (`node5`, RTX 5090): compared compressed, FP64-resident and
  uncached full WB97M-V Hessians for Direct RKS water and split-grid CUDA DF
  UKS with fitted long-range exchange. Projected storage was `400/800` and
  `64/128` bytes (FP32/FP64), respectively. Both mixed matrices used nine
  seeds and nine physical residual checks. Maximum matrix differences from
  both FP64 paths were `4.29e-12` and `2.61e-13 Ha/Bohr²`.
- Independent reconverged Libcint/Libxc energies at `0.004` and `0.002` Bohr
  gave complete Richardson stencil errors of `3.76e-8` and `5.36e-9 Ha/Bohr²`.
  Raw symmetry, translation, and agreement with one-shot HVPs also passed.
- Slurm 2098: 49 tests passed, including matching Python/CLI dense results,
  rejection of the dense-only option for CLI HVP, FP64 GMRES, AUTO SCF and
  r²SCAN+rVV10 regressions. Host frontend and budget validation: 128 passed;
  native C bridge input gate passed separately. Logs and raw stencils are
  under `build/hessian-gpu-sm120/hessian-fp32-cpks-*`.
- This is a storage tradeoff, not an endpoint speed claim. The same probes
  (one HVP plus the dense matrix) consumed `123/50/130` and `101/32/100`
  physical induced actions (compressed/resident/uncached), including all
  projected-operator setup. Do not enable it automatically based on cache
  bytes or projected matvec timings alone.

## Follow-up

Slurm 2099 also reproduced the existing dense CLI `invalid argument` for
neutral PBE0 UKS H3 (doublet, STO-3G, centers `[.13,-.21,.17]`,
`[.43,.19,2.1]`, `[.31,.08,4]` Bohr, product grid `8/4/8`) with both the
pre-change binary and the new default FP64 mode. Whether the rejection is
a scientific admission gate or an endpoint defect remains to be diagnosed.
