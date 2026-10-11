# Decision: bind physical response consumers without changing scientific owners

Status: implemented
Date: 2026-10-11

## Problem

The #932 audit found native execution consumers still bypassing the shared
response seam, despite an existing bound-plan contract. RHF frame execution also
has two scientifically distinct operators: a provisional screened action and an
exact physical action. Sharing only a dimension/resource estimate is not an
operator-identity proof. Python multi-RHS and stationary nuclear adapters rejected
empty requests and could discard imaginary components during FP64 conversion.

## Decision

Use separate bound problems/plans for screened and exact HF actions and move the
remaining CC Z solve through the existing shared response seam. Declare CC Z
symmetry only after its existing independent dense asymmetry/curvature gates.
Forward callback right preconditioning to the existing native GMRES algorithm;
do not introduce a replacement solver or change physical residual acceptance.
Validate direct native JVP/VJP dimensions before method callbacks.

Zero-column Python requests retain their response dimension and still validate
generation/provider/vector-owner/recycle lifetimes, request options and callback
contracts. They execute no workspace, rank analysis, transfer or scientific
action. Perturbation labels remain an exact request contract, so partial/empty
requests cannot silently discard labels. Real nuclear algebra rejects complex
data before conversion and avoids extra copies for already-FP64 inputs.

## Rejected shortcuts and remaining boundaries

The plan's logical token is not a scientific generation or borrowed-state lease.
Existing method/reference/provider owners remain responsible for those checks.
Likewise, moving native host-vector consumers behind `solve_response` does not
make their Krylov vectors device-resident. Conventional/DF native CC Lambda,
RHF frame response and installed native CPKS retain host-vector execution. The
existing resident qualification engines still need production integration with
right preconditioning, stream ownership and complete-endpoint A/B admission.
Native conventional MP2 also returns host MO blocks to its shared orbital and
weight algebra. UKS remains a tool-owned qualification adapter and cannot be
inventoried as the installed RKS owner merely because both use native J/XC.

The #934 production-path inventory records these exceptions and their retirement
conditions explicitly, as well as the unsupported complete CUDA RKS HVP binding
tracked by #1905. Local `fxc`, resident J/K storage and mixed integral stages are
not substitutes for complete molecular HVP source coverage or public support.
No feature capability or default resident schedule is promoted by this change.

## Validation

Native tests compare shared and raw GMRES solutions, residuals, action counts and
workspace accounting, including preconditioner failure isolation. Installed tests
cover empty/partial/ragged requests, stale owners, complex nuclear inputs and
preserved nuclear algebra. The real-GPU empty resident regression checks vector
leases and raw transfers/actions; it must execute under finite Slurm allocation.
