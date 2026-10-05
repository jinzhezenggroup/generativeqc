# Decision: amortize intermediate exact-response residual actions

Status: implemented; complete endpoint and retained-RHF integration qualified
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

## Retained-RHF integration qualification

After parent `c7486bf2e`, source `5a805b274` passes build 2349, expanded host
2351, response/independent small FD 2352, shared Lambda 2354, complete force 2355
and report 2359. Cold/warm ethane force times are 1120.394947 / 1118.673403 s,
with 17 / 16 exact J/K actions and 12 / 11 Arnoldi iterations. Both calls reject
recycling; the warm frame's one fewer iteration cannot be attributed to reuse.
There is no matched stronger/diagonal ablation of that frame. Maximum force
difference from the frozen reference is 3.615e-9 Eh/Bohr and the original gates,
including limited independent large FD, pass. Retained evidence distinguishes
each binary and GPU and makes no cross-version timing ratio.

## Post-merge integration qualification

Source `dc68eb0b2` composes master `12d709e46` through parent `33083727b`.
Build 2367, host 2369 (104 cases), response/independent small FD 2370, shared
Lambda/triples 2372, complete energy/force 2373 and report 2377 pass. Triples
has 25 passing cases and three explicit test-hook skips, not 28 passing cases.
The actual default diagonal/checkpoint force retains 17 exact J/K actions and
12 Z iterations; neither stronger preconditioning nor recycling is enabled.
Complete ethane force is 1221.661691 s, including RHF 234.671782 s and orbital
response 441.628298 s. Energy is 282.120948 s on the same allocation, but its
different RHF time prevents attributing their difference purely to force work.

Maximum force difference from the frozen reference is 2.954e-9 Eh/Bohr; all
original residual/stationarity and limited independent large FD gates pass.
The separate `post-merge-summary.json` retains source, binary, GPU and work
provenance. This verifies integration, not a new checkpoint timing ablation.
The public DF registration is energy-only; these are internal complete force
endpoints. Preserve the frozen selection evidence and unchanged acceptance
gates rather than relabeling old timings as measurements of the latest source.
