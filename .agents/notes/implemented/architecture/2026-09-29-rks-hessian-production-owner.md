# Decision: install the semilocal RKS Hessian consumer chain

Status: implemented
Date: 2026-09-29

## Problem

After #1594, the live LDA/PBE RKS response state, exact J provider, semilocal
XC response and shared Krylov solver were installed owners, but the actual
second-order molecular consumer still depended on repository-only Hessian
modules and on the old post-HF NativeSource topology seam.

The remaining tools-owned pieces were:

- nuclear-direction construction and shared CPKS response assembly;
- generated first-integral response contractions;
- generated fixed-weight second-integral HVP contractions;
- the seven-source stationary HVP assembly;
- bounded raw full-Hessian assembly.

A public Calculator Hessian cannot depend on those repository tools, and
retaining NativeSource solely for Hessian topology would reintroduce the
post-HF ownership seam removed from NativeRKSResponse.

## Decision

Install three production modules:

- `generativeqc.rks_hessian_integrals`
  - borrows the live `NativeAO` owner through `RKSIntegralTopology`;
  - owns the generated direct Cartesian first-integral nuclear direction;
  - owns plan-weighted first-integral contractions;
  - owns bounded generated second-integral HVP contractions;
  - owns the closed-form all-electron nuclear HVP.
- `generativeqc.rks_hessian_directional`
  - owns one-/multi-direction RKS nuclear response;
  - composes generated integral geometry response with the exact native
    SCF-domain XC/grid/Becke response;
  - injects the installed response solver into the installed stationary nuclear
    reconstruction owner.
- `generativeqc.rks_hessian`
  - owns the MethodIR-derived source inventory;
  - owns complete seven-source RKS HVP assembly;
  - owns shared multi-RHS HVP blocks;
  - owns bounded raw full-Hessian publication and symmetry diagnostics.

The old tools `rks_directional` and `rks_molecular` modules are module
aliases to the installed owners.

## Integral topology

The production integral provider no longer consumes
`tools.generativeqc_posthf.NativeSource`.

`RKSIntegralTopology` is constructed directly from the borrowed live
`NativeAO` basis. It validates:

- Cartesian representation;
- nonempty AO/atom topology;
- shell component counts matching `NativeAO.nao`;
- current NativeAO lifetime;
- through-f second-derivative capability.

Shell order, atom ownership, primitive exponents/coefficients and Cartesian
normalization therefore come from the same NativeAO object already bound to the
live RKS response.

## Scientific preservation

The installed consumer retains the previous qualified CPU direct all-electron
LDA/PBE RKS decomposition:

1. one-electron;
2. Coulomb;
3. XC AO motion;
4. XC grid-point motion;
5. XC partition-weight motion;
6. overlap/Pulay;
7. nuclear repulsion.

The same MethodIR `StationaryHVPPlan` supplies fixed and response weights.
The same compiler-owned first/second integral DAGs are used. The same native
SCF-domain XC point response, analytic Becke JVP/mixed response and CPKS density
response are reused.

The full Hessian remains raw: no post-hoc symmetrization is performed.

## Preserved boundaries

- CPU only for the complete molecular RKS HVP/full-Hessian consumer.
- Direct, all-electron, Cartesian, closed-shell LDA/PBE only.
- No DF, ECP, UKS, meta-GGA/r2SCAN, hybrid/RSH or VV10 Hessian promotion.
- No public `Calculator.hessian` or C property ABI yet.
- No new performance claim.
- The public-resource contract remains a separate final promotion step.

## Compatibility

The historical tools RKS modules resolve to the exact installed module objects.
Top-level `tools.generativeqc_hessian` RKS exports therefore preserve object
identity.

The directional module preserves the named
`solve_stationary_nuclear_perturbation[s]` seam so existing multi-RHS tests
can continue to monkeypatch/count one shared solve.

The pure resource-admission helper continues to accept a metadata-only fixture;
physical execution still requires the NativeAO-owned topology.

## Evidence

Ownership tests reject any `tools.*` import from the installed RKS Hessian
modules, assert tools module alias identity, assert top-level RKS API object
identity, and reject NativeSource/NativeRHFState ownership in the production
integral topology.

Existing independent finite-difference, raw-symmetry, larger-domain and
resource-admission Hessian tests remain the numerical qualification gates.

Refs #178, #179, #180, #932, #1412, #1572, #1582, #1594.

Agent: ChatGPT
Model: GPT-5.6 Sol
