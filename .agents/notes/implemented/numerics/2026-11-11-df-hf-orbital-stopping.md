# Decision: align compact DF RHF stopping with stock GPU4PySCF

Status: implemented; device and endpoint qualification pending
Date: 2026-11-11

## Problem

#3065 aligned Direct RHF only. Compact DF RHF still required density RMS and
an AO-maximum commutator, unlike the stock E/MO-gradient conjunction. The
aligned-RKS README cohort retained a DF HF96 cold loss (18.182064924389124 s
versus 17.741809528321028 s), with 11 target iterations / 12 physical Focks
versus 10 iterations / 11 effective potentials. This is not a causal timing
ablation: initial guesses and execution owners also differ.

## Decision

Reuse `GENERATIVEQC_CUDA_RHF_CONVERGENCE` for qualified all-electron FP64 compact
DF RHF with physical DIIS residual storage. After DIIS retains its unchanged
AO residual history, transform only the disposable stopping copy to
`X^T(FDS-SDF)X` using already reserved scratch. Its Frobenius norm divided by
sqrt(2) is the restricted physical MO-gradient norm. Default candidate admission
requires exact `abs(delta E) < energy_tolerance` and gradient norm below
`sqrt(energy_tolerance)`. Density change remains a finite diagnostic, not an
additional gate. Remove the same unintended density/energy-roundoff additions
from the shared Direct/DF orbital predicate so the contract has one meaning.

Explicit strict, tighter public E/D, ECP, mixed/application-budget, physical
reference and no-DIIS owners retain their existing strict/bounded behavior.
Explicit orbital requests on unsupported owners fail closed. Prepared Python
resource/checkpoint identities already bind the shared request; compact graph
reuse additionally binds the resolved policy. Final physical Fock, projector,
trace/idempotency, canonical proof and response-token checks are unchanged.
Neither the force accuracy gates nor GPU4PySCF defaults are modified.

## Validation contract

Execute actual Direct/DF reduction in ordinary and captured CUDA calls,
including large density change, localized/distributed gradients, nonfinite
entries and inactive neighbors. Independently compare complete energy/force
endpoints, cold/frozen warm and moved geometry for several restricted models,
including strict controls. Qualify large DF HF endpoints against a tightly
converged same-fitted-model oracle; retain the separate dense-fitting failures.
Measure strict/default endpoints in balanced same-card order, retain all work
and do not replace a complete README cohort with cherry-picked timing points.

## Rejected alternative

Do not treat the previous DF HF loss as irreducible merely because the owner
was outside #3065, and do not retune GPU4PySCF or relax independent accuracy
gates to manufacture a win.
