# Decision: fold the frozen semilocal Becke weight Hessian across columns

Status: implemented
Date: 2026-11-27

## Problem

The dense 96-atom FP64 endpoint repeatedly visits all point-pair switches for
each Cartesian right direction, in both Direct and fitted Hessians. The
unweighted semilocal XC energy density multiplying the second weight derivative
is independent of the response column. A dense point-by-coordinate-by-coordinate
cache is too large and unnecessary.

## Decision

At a fixed point write the unnormalized atom product as `P_i`, the sum as `S`,
the owner fraction as `f=P_owner/S`, and the moving weight as `w=measure*f`.
The derivative of the ratio gives

```
H(w) = measure/S * (H(P_owner) - f*sum_i H(P_i))
       - grad(w) (grad(S)/S)^T - (grad(S)/S) grad(w)^T
```

For a product with no zeros, the Hessian is `P_i` times the outer product of
its log gradient plus the sum of its pair-local log curvatures. An exact zero
factor can retain derivatives. One zero leaves the Hessian of that factor
and its two cross terms with the nonzero product gradient; two zeros leave
only the two cross terms between those factors; three zeros kill the whole
second jet. This avoids dividing by zero without dropping rounded-zero
derivatives. Transform the fixed-point gradients and both Hessian axes by
replacing the grid owner's row by minus the sum of the other center rows.

One bounded point tile carries a signed outer-product representation for FP64
GEMM and sparse pair curvatures. Curvatures reduce over one grid owner before
scattering the translated blocks. Only first derivatives and the contracted
dense matrix are retained across columns. The CPKS and XC cross terms still
use the original first weight derivatives and the original quadrature.

For wide FP64 r²SCAN CPKS, reuse the existing batched occupied–virtual
projection within an explicit work and tile budget, including pure semilocal
DF. The Coulomb complement and the independent physical residual checks still
run. The projector may retain rows that the generated XC value and derivative
evaluator proved are identically zero; no new density cutoff is introduced.
If the fixed-point correction census or projection refuses, use the existing
full physical operator build.

## Invariants and fallback

The actual local XC density and the geometry reference belong to the same
stationary CUDA lease. Recompute and verify the original partition. Reject
products outside the admitted binary range, invalid norm domains and nonfinite
staging/output. Do not publish a partial fold, change SCF, or subtract a new
numerical cutoff. Use the incumbent bounded per-column, shared-scaled CUDA
path on an unsupported shape/lease or failed fold. Charge both the retained
grid allocation and the temporary tensor-provider allowance before attempting
the optional path. Do not use this simple atom-major tile for ragged grids.

## Evidence

The unadjusted standalone GPU prototype agrees with signed contraction oracles
for 2, 6, and 10 atoms and folds a synthetic 82,944-point/96-atom quadrature in
4.02 s on RTX 5090. This is not an endpoint or a qualification of the normal
product domain. Developer evidence is in the ignored local scaling directory.
The independent displaced-value oracle also passed for 162 moving grid points,
including adjusted radii and signed measures. The native geometry suite
checks the successful folded contraction and numerical refusal.

Slurm 20912 paired fresh-process FP64 SCF + full Hessian on one RTX 5090:
96 H, 96 Cartesian STO-3G functions, 82,944 points on the same unpruned moving
grid, STO-3G fitting basis and second-order auxiliary-basis response for DF.

| Provider | Native | GPU4PySCF 1.9.0 | Max raw error (Ha/Bohr²) | Native peak / total |
| --- | ---: | ---: | ---: | ---: |
| Direct | 84.19 s | 96.46 s | 3.37e-9 | 8,453 / 32,607 MiB |
| DF | 82.31 s | 96.33 s | 4.88e-9 | 6,853 / 32,607 MiB |

Raw error, asymmetry, and translational residual were gated at `2e-8`, with
no publication-time symmetrization or numerical host fallback. Both endpoints
completed all 288 columns and 82,944 mixed grid contractions. The 2,880 induced
Fock actions include the projected build's Coulomb complement and physical
checks; their count is not the number of full XC actions. Direct executed
1,470,176,973 exact primitive components; DF executed 288 value and 288
contraction passes. The projected-work admission bound on this workload is
`5 * 82944 * 2304²` scalar products, separate from geometry pair work.

The per-column incident-edge specialization was withdrawn: it did not improve
the endpoint. Exact zero semilocal cotangent point elision alone improved the
same-card Direct/DF endpoints by only about 2%; it cannot close the gap.

## References

PR #3141; Slurm jobs 20898, 20901, 20909, and 20912;
`src/dft/cuda_becke_weight_fold.cuh`.
