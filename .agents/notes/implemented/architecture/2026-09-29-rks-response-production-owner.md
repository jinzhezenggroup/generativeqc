# Decision: install the shared Krylov solver and live RKS response adapter

Status: implemented
Date: 2026-09-29

## Problem

After #1582 the closed-shell response problem, RHF/CPKS equations and semilocal
XC response kernel were installed owners, but the actual LDA/PBE RKS response
endpoint still depended on repository-only components:

- `tools.generativeqc_response.krylov` for scalar/multi-RHS GMRES;
- `tools.generativeqc_posthf.ReferenceSnapshot` for the closed-shell reference;
- `NativeSource` / `NativeJKBackend` / `CudaSpinJKBackend` for exact J;
- the tools `NativeRKSResponse` live-state adapter.

A public Hessian path cannot depend on those repository tools, and copying the
solver or response equations would create parallel scientific owners.

## Decision

1. Make `generativeqc.response_solver` the canonical owner of the existing
   bounded true-residual GMRES, blocked/recycled multi-RHS execution and Krylov
   recycle semantics. `tools.generativeqc_response.krylov` is a module alias
   to the installed implementation so historical private imports resolve to
   the same objects.
2. Add `generativeqc.rks_response.NativeRKSResponse` as the installed live
   all-electron LDA/PBE RKS adapter.
3. Build the response reference directly from the current
   `StationaryKsState` and an exact J-only `FockPlan`.
4. Reconstruct hcore by evaluating that same exact FockPlan at zero density.
   Native Fock semantics are `F = hcore + J/K`, so this is an exact provider
   result, not an inferred or external value.
5. Use `FockPlan` with Coulomb present, exchange absent, derivative order zero
   and zero screening for both CPU and CUDA response J actions.
6. Keep the exact native SCF-domain XC response: CPU consumes the snapshot point
   response and CUDA consumes the snapshot-owned CUDA response plan.
7. Make the tools `NativeRKSResponse` symbol an alias of the installed class.
   UKS remains under its existing spin-response owner and is not inferred.

## Preserved boundaries

- No SCF rerun or synthetic convergence proof.
- The native snapshot token remains authoritative; replay, failure or owner
  closure revokes the response.
- RKS admission remains direct all-electron, unscaled FP64 LDA/PBE only.
- No r2SCAN/meta-GGA, hybrid/RSH, VV10, DF or ECP response capability is added.
- No public `Calculator.hessian` property or C property ABI is added.
- No RKS Hessian performance claim is made by this ownership cutover.
- UKS response stays on its existing dedicated spin contract.

## Compatibility

The installed reference retains the fields used by response/Hessian consumers:
overlap, hcore, Fock, canonical coefficients/energies/occupations, electron
count, energy, residual and exact geometry/basis/functional/grid identities.
It remains a frozen dataclass so identity-negative tests can use
`dataclasses.replace`.

The response backend retains an explicit `_plan`, mutable identity for stale
provider tests, resource statistics and `validate_reference`. CUDA XC
diagnostics continue to come from the same snapshot CUDA owner.

## Evidence

Existing CPU RKS response tests remain the primary independent libcint/Libxc and
reconverged-density oracle. CUDA response tests continue to block CPU fallback
and validate exact owner lifetime/resource accounting. Hessian RKS tests consume
the same top-level `NativeRKSResponse` symbol, now resolving to the installed
adapter.

Ownership tests additionally require:
- tools Krylov module identity equals `generativeqc.response_solver`;
- tools/top-level NativeRKSResponse identity equals the installed class;
- installed response modules import no `tools.*`;
- the tools native-KS module no longer defines a second NativeRKSResponse class.

Refs #179, #180, #932, #1412, #1572, #1582.

Agent: ChatGPT
Model: GPT-5.6 Sol
