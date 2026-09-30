# Decision: move closed-shell stationary nuclear response to the installed runtime

Status: implemented
Date: 2026-09-29

## Problem

After #1412 moved method-neutral second-order orchestration into
`generativeqc.second_order`, the next public-Hessian dependency still lived
under repository-only `tools.generativeqc_hessian`: closed-shell nuclear RHS
construction, metric response, occupied-orbital/density reconstruction and the
single-/multi-RHS response consumer.

Letting a future public Calculator Hessian import that module would create an
installed-runtime dependency on `tools.*`. Copying the equations would create
two scientific owners.

## Decision

Make `generativeqc.stationary_nuclear` the canonical installed owner for the
closed-shell stationary nuclear-response algebra and reconstruction contract.

The production owner:
- contains no method-name dispatch beyond the already-qualified closed-shell
  `rhf` / `cpks` response contract;
- accepts the response solver as an injected callable and therefore does not
  duplicate #179 GMRES/Krylov implementation;
- preserves exact metric-density, RHS, occupied response, density and
  energy-weighted-density reconstruction semantics;
- preserves bounded multi-RHS order and resident-reconstruction callbacks;
- remains independent of repository `tools.*`.

`tools.generativeqc_hessian.response` is now a pure compatibility re-export.
`tools.generativeqc_hessian.perturbation` is only an adapter that injects the
existing #179 `solve` / `solve_many` implementation and retains strict RHF
compatibility wrappers.

## Preserved boundaries

- No public `Calculator` Hessian/HVP property is added.
- No C property ABI changes.
- No new DFT, CUDA, DF, ECP, UKS, hybrid, RSH, meta-GGA or VV10 capability.
- The method-specific RKS AO/grid/integral response providers remain separately
  owned and must migrate/qualify before a public Hessian endpoint is promoted.
- A solver result must still converge before response reconstruction is
  published; nonfinite/malformed matrices remain fail-closed.

## Evidence

Ownership regressions assert that tools-visible response dataclasses/algebra are
the same installed objects, production imports no `tools.*`, and the old
response shim defines no duplicate scientific functions. Existing Hessian
response tests continue to exercise the compatibility solver adapter and the
RHF/CPKS induced-Fock contract.

Refs #180, #932, #1412.

Agent: ChatGPT
Model: GPT-5.6 Sol
