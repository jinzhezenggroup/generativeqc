# Decision: replace the semilocal RKS HVP size gate with explicit resource admission

Status: implemented
Date: 2026-09-27

## Problem

The tools-only semilocal RKS HVP path carried a historical validation limit of
12 Cartesian AOs / four atoms. That limit did not describe the actual memory
owners and rejected work even when every indivisible numeric allocation fit.

## Decision

Keep the existing scientific capability boundary (CPU, direct all-electron,
Cartesian closed-shell LDA/PBE RKS), but replace the historical size gate with
explicit integral/output workspace admission.

The caller supplies an integral budget. Before CPKS or HVP work begins, the
consumer checks the plan-weight workspace and the largest indivisible provider
output. The same budget is passed to the generated second-integral provider.
Provider/compiler metadata and live SCF/grid state remain outside this numeric
budget and are reported separately; this is not an end-to-end process-memory
claim.

## Preserved contracts

- No public Calculator Hessian/HVP endpoint is added.
- CUDA, DF, ECP, UKS, meta-GGA, hybrid/RSH and VV10 second-order support remain
  fail-closed.
- The seven molecular HVP sources and shared response equations are unchanged.
- An undersized integral budget fails before response work begins.
- Existing raw-Hessian output-budget semantics remain owned by the parent slice.

## Validation requirement

Retiring a validation-size gate requires a newly admitted domain to be checked
with a nontrivial independent numerical oracle. A zero translational HVP alone
is insufficient because missing/cancelling multicenter source terms can pass.
The accepted larger-domain regression should therefore include a >12-AO
multicenter nonzero direction and finite differences of independently
reconverged analytic gradients.

References: #178, #180, #1402, #1406, #1409.

Agent: ChatGPT
Model: GPT-5.6 Sol
