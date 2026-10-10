# DFT Hessian and HVP execution contract

This page describes second-order MethodIR source rules and the
stationary executor, including the bounded public molecular
endpoint. An available functional or first derivative never implies
a qualified second-order method/backend combination.
The [RHF term map](hessian.md) remains the conventional
Hartree–Fock derivation; [Reference](../reference/capabilities.md)
is the user-facing source for capability admission.

## Stationary DFT plan and molecular execution

DFT Hessian source planning comes from resolved `MethodIR` primitive
capabilities, not functional-name branches. `StationaryHVPPlan` admits
compiler-only direct all-electron FP64 LDA/GGA RKS/UKS with the native SCF
point model and optionally composes full-range exact exchange. LDA/GGA share
one-electron, Coulomb, XC AO/grid/partition, overlap/Pulay and nuclear
sources, while their `rho`/`sigma` features come from the method graph.
Global hybrids add one exchange source without PBE0/B3LYP-specific code.

The planner binds the shared #179 CPKS contract, #161/#236 XC feature-Hessian
action and #178 weighted second-integral HVP contract. Its integral blocks
execute only the bounded source algebra; native CPKS and XC/grid consumers
remain separate owners. Second-order support is now admitted through an exact
primitive-type rule registry rather than a fixed "one semilocal primitive"
condition. Each rule declares its required derivative capabilities, supported
ingredients, directional sources and any response inputs needed by bounded
integral HVP lowering. The plan derives its source inventory by composing those
rules with the stationary mean-field envelope. It fails closed when an active
primitive has no registered second-order rule. The full-range exchange rule reuses the existing
ordered-quartet stationary-gradient weight and its TensorIR JVP with both
same-spin density response inputs. It requires first-integral directions
and generated weighted second-integral HVPs. Range-separated exchange,
`tau`, nonlocal correlation, DF and ECP remain unsupported at this planning
boundary. A compiled source rule does not qualify a molecular endpoint.

Molecular execution is a separate method-neutral layer. Its canonical
installed owner is now `generativeqc.second_order`; the former
`tools.generativeqc_hessian.stationary_executor` module is a compatibility re-export
and contains no second implementation. `StationarySecondOrderExecutor` accepts
only a plan identity and complete ordered source inventory, one perturbation
provider, one opaque stationary response driver and exactly one contributor per
declared source. It performs one response solve, passes the same response object
to every contribution and publishes a result only after every source returns a
finite Cartesian HVP. There is no HF/RKS/UKS or functional-name dispatch in
this executor. A MethodIR-derived DFT plan and an HF second-order plan can
therefore share the same orchestration while retaining their own perturbation,
response and primitive providers. Missing or extra contributors fail before
execution.

The plan now exposes bounded `integral_block` programs for the one-electron,
Coulomb, overlap/Pulay and optional full-range exchange sources. Each block
reuses the stationary-gradient
source energy and derives its fixed integral weights, then generates an exact
TensorIR JVP for a supplied
density or weighted-density response, and contracts that response with the
first-integral directional tile plus #178's fixed-weight second-integral HVP
vector. This is an executable source-level algebra slice for both RKS and UKS;
native shell/center recovery, shared CPKS execution, XC/grid/partition motion
and molecular assembly remain owned by their qualified consumers. In particular,
compiled PBE0/B3LYP exchange HVP weights do **not** qualify hybrid molecular
Hessians: the mixed K/XC CPKS response, metric/Pulay relaxation and complete
molecular source inventory still need independent numerical acceptance.

The closed-shell nuclear-perturbation consumer follows the same rule. Its
canonical installed owner is now `generativeqc.stationary_nuclear`: metric-density
RHS construction, occupied-orbital/density reconstruction and the single-/multi-
RHS consumer contract live there without importing repository `tools.*`.
Method-specific Fock physics remains injected through the response operator's
`induced_fock(delta_density)` contract. The shared #179 GMRES implementation
is now installed as `generativeqc.response_solver`; tools response/Hessian
imports are compatibility aliases or wrappers around the same production
objects. Existing `solve_rhf_nuclear_perturbation[s]` names remain strict RHF
compatibility wrappers.

The CPU direct all-electron Cartesian LDA/PBE RKS path now has installed
production owners for the live CPKS state/J/XC binding, nuclear-direction
response, NativeAO-owned generated first/second integral response, complete
seven-source molecular HVP, shared multi-RHS CPKS and raw bounded full-Hessian
assembly. The existing finite-difference, raw-symmetry and resource-admission
tests continue to qualify this same scientific slice. Repository
`tools.generativeqc_hessian.rks_directional` and `rks_molecular` are module
aliases to those installed consumers. The same owner is now exposed through
`Calculator.hessian_vector_product()` and `Calculator.hessian()` for CPU direct
all-electron Cartesian strict-FP64 closed-shell LDA/PBE RKS. Public calls retain
explicit integral/output budgets and report the executed public endpoint in
result diagnostics. Wider method/backend support remains separate.

## Qualified execution boundary

The documented public molecular endpoint is CPU direct all-electron
Cartesian strict-FP64 closed-shell LDA/PBE RKS. Additional functional
families, density fitting, ECP, spin/backend domains, and resource
scopes must independently pass their full second-order gates.
Generated CUDA components are not, by themselves, a public CUDA
molecular Hessian/HVP endpoint.
