# Decision: prepared native ECP force defaults retain derivative resource identity

Status: implemented
Date: 2026-11-11

## Problem

The packaged through-f CPU first-derivative provider added for #2844–#2848
could answer an explicit prepared force request, but the Python method-level
metadata still described the narrower s/p/d ECP compatibility consumer.
Consequently a default public f-shell ECP singlepoint converged successfully
while returning no forces. Broadening the compatibility predicate would have
admitted a different, unqualified derivative owner.

A second ownership mismatch accompanied promotion: non-output-aware resource
requests discarded explicitly requested forces when method metadata lacked
them. Replaying native forces with an energy-only request would reserve the
wrong transient workspace. A successfully enlarged plan also needs to retain
its observable identity across subsequent energy-only replays.

## Decision

Use the immutable native prepared item's capability to extend ECP default
outputs. Preserve the deliberate energy-only all-electron and automatic-method
defaults, singlepoint-versus-batch policy, explicit energy-only requests and
the conservative public method registry. Missing native capability never
widens the Python diagnostic's angular or resource domain.

Resource requests retain the union of requested and conservatively required
outputs, except for already output-aware contexts. Revalidate a newly requested
output when the prepared resource identity does not include it. Reject an
insufficient budget before native execution, and publish an enlarged plan only
after successful execution. Track the actual admitted request's observables,
not merely the last requested subset, so an energy-only replay cannot erase a
retained derivative reservation. Honor the same identity in caller-supplied
force plans.

## Evidence

- Twelve native-only compact NaH complete endpoints cover Cartesian/spherical
  LDA/PBE/B3LYP and RKS/UKS, with f orbitals on both centers. They prohibit the
  Python compatibility force bridge. Maximum independent moving-grid analytic
  force error is `2.4877613858e-11 Eh/Bohr`; maximum multistep energy-difference
  directional error is `5.9541581734e-11 Eh/Bohr`. Gates remain `1e-7` and
  `2e-7`, respectively. State-density and independent orbital-gradient gates
  are `2e-8` and `2e-9`.
- Tests exercise missing native discovery, explicit energy-only selection,
  default singlepoint and batch outputs, force-budget failure before execution,
  recovery, caller-supplied force plans and retained promoted reservations.
- The compact fixture keeps complete-endpoint regression work bounded; existing
  raw f-shell tests retain signed multiexponent contraction coverage.

## Boundaries

No scientific kernel, quadrature Hamiltonian, precision gate or compatibility
consumer is widened here. Compact NaH force qualification does not qualify
canonical mixed Ni/I/Cs ECP derivatives, Ni UKS curvature, total UKS/ECP D3
curvature or a molecular transition-state task. Those issue-level acceptance
requirements remain independent.

## Revisit when

Additional prepared derivative providers or resource owners are qualified.
Keep default-output policy, actual prepared capability and resource identity
separate rather than restoring a duplicated Python angular whitelist.
