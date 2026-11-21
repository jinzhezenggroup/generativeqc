# Decision: Share the rVV10 pair primal with the CUDA Hessian grid

Status: implemented (grid component)
Date: 2026-11-21

## Decision

Use the existing preconditioned rVV10 representation for the second-order
grid: each leg supplies `omega/kappa` and `kappa*sqrt(kappa)`, followed by
the same ordered-pair denominator used by first-order TensorIR. Scalar AD
differentiates both representations with respect to density/gradient fields
and differentiates the distance, attached AO jets, and weights where they
move. The local beta term contributes its weight/density mixed derivative
even though it has no fixed-grid density kernel. Damping, gradient scale and
method coefficient are runtime values, not properties of the generated artifact.

Do not substitute the original VV10 formula or freeze local scales when
taking the JVP. The existing one-half energy convention, fixed active set,
ordered reductions, borrowed-stream checks and workspace/pair admission
apply to both variants. The component takes an explicit parameter record;
it does not select a new model for an accepted molecular state.

## Evidence

Slurm 2081 on node5 (RTX 5090): 15 checks passed, including rVV10 RKS and
UKS with different damping, gradient scale and coefficient. The independent
real-space oracle checks reference energy and potential, induced Fock,
moving Fock, and nuclear mixed contraction from displaced values. A one-pair
budget shortfall refuses preparation without changing output. Existing VV10
component, grid split, graph replay and capacity regressions also passed.
The broader grid suites in Slurm 2082 passed 98 tests; 10 optional
molecular-probe or wide-arithmetic-oracle tests skipped. The targeted host
suites had 89 passed, 93 GPU-only skips.

Reproduce with `test_wb97mv_nuclear_grid.py -k rvv10_bilateral_moving_grid`,
setting `GENERATIVEQC_CUDA_WB97MV_GRID_PROBE` to the linked CUDA probe and
running the native cases inside a finite Slurm job.

## Remaining molecular boundary

The native SCF and Hessian family policies still reject a device-fused rVV10
molecular model. Exposing a composition such as r²SCAN+rVV10 requires a
matched point domain and nonlocal grid policy chosen before SCF, authentication
of the actual retained rVV10 source, and independent reconverged molecular
energies. A passing component test cannot certify that integration.
