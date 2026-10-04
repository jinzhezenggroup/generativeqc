# Decision: amortize intermediate exact-response residual actions

Status: implemented; frozen complete endpoint qualified, newer parent integration pending
Date: 2026-10-05

## Diagnosis

#1901 targets fewer exact J/K actions in the remaining orbital-response phase.
The existing host GMRES defaults to a true candidate residual at every Arnoldi
iteration. Thus every iteration can consume two full exact J/K actions before
the RHF owner's separate final scalar audit. A more capable preconditioner
would still pay this avoidable duplication.

## Decision

Both native host and resident controllers now let the projected Hessenberg
residual request an early true residual when it predicts convergence. Only the
fresh unpreconditioned action can accept a solution. Restarts, breakdown,
iteration exhaustion and the configured maximum periodic interval also retain
mandatory exact checks. The general solver default remains one; exact RHF's
default interval becomes its restart size (30). The existing independent scalar
lowering and full frame stationarity gates remain unchanged.

No screening, approximate operator, Hessian construction or new workspace is
introduced. The explicit interval-one path remains available for comparison.
This is the first #1901 work reduction; it does not complete the stronger
preconditioner, strict-identity recycling and full endpoint acceptance by itself.

## Evidence

Slurm job2301 passed the full native GMRES contract test after the test script's
ccache shared-library path was repaired. New tests compare every-step versus
deferred actions, host versus resident controllers and an independent dense
residual. An intentionally inconsistent operator predicts convergence but fails
the true residual, proving the estimate cannot publish a result. The native
workspace bound is unchanged. The complete molecular results below supersede
the original pending measurement status.

### Complete endpoint evidence

Frozen production `95c71862f`, Slurm 2321 on one n2 RTX PRO 6000, compares
complete native ethane230 force calls with every-action and checkpoint residuals.
Both use 12 Arnoldi iterations; candidate residual actions fall from 12 to 1
and total exact J/K calls from 28 to 17. Orbital/nuclear time is
669.660284 → 472.719821 s and complete time 1333.307518 → 1162.576380 s.
These are single observations. Unrelated RHF variation is retained separately,
not assigned to checkpoint savings. The physical operator and final independent
scalar residual/stationarity audits are unchanged.

All five diagonal/stronger/cold/warm variants pass the force and residual gates;
the maximum force difference is 4.684e-9 Eh/Bohr. Independent small all-coordinate
FD and existing large two-coordinate/two-step FD re-audits pass. The stronger
inverse and recycling do not further reduce large actions. Retain checkpoints
as the default and the other accelerators as opt-in. Full records and source,
binary, GPU and input identities are in
`benchmarks/results/rhf-response-accelerators-1901/`.

The subsequently integrated prepared-provider branch passes build 2333,
response/independent small FD 2340, common host 2339 and shared Lambda 2348.
Its separate complete endpoint qualification does not inherit the old timings.

## Next bounded experiment

The following proposal is now implemented in
[bounded DF preconditioning and exact-identity recycling](2026-10-05-rhf-df-preconditioner-recycling.md).
The original rationale below is retained; complete large-endpoint qualification
is still separate from the implemented small-system gates.

Before changing the physical operator, derive an optional conventional
preconditioner from already retained same-frame DF factors. With x indexed ia,
the canonical RHF action is gap*x + 4(ia|jb)x - (ij|ab)x - (ib|ja)x.
A candidate is a diagonal exchange correction plus the complete low-rank
Coulomb term, inverted with a Woodbury/Cholesky solve. This must first be
independently compared to the existing frame TensorIR and rejected if its
diagonal/Cholesky/budget is unsafe. Setup and retained factors must be charged
before CC source owners are released. This paragraph is a proposed experiment,
not an implemented preconditioner or measured benefit.

Recycling must likewise bind exact normalized geometry/basis, the complete
reference/operator state and representation, rather than accepting matching
dimensions or approximate C equality. Existing `BasisGeometryIdentity` and the
shared Python response/recycling contracts are precedents. Reusing a subspace
across fresh RHF endpoints may be rejected even at identical geometry when
their canonical frames differ bitwise; no cold-endpoint benefit can be inferred
from a same-reference second-RHS microbenchmark.
