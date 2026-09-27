# Decision: move method-neutral second-order orchestration to the installed runtime

Status: implemented
Date: 2026-09-27

## Problem

The method-neutral `StationarySecondOrderExecutor` lived under the repository-only
`tools.vibeqc_hessian` package even though installed runtime consumers need the
same orchestration contract. Keeping a second production copy would duplicate
source ordering, transactional publication, response reuse, and identity policy.

## Decision

Make `vibeqc.second_order` the single canonical implementation owner. Keep
`tools.vibeqc_hessian.stationary_executor` as a compatibility re-export of the
same Python objects; it must not retain a second implementation.

The owner remains method-neutral. Plans provide only identity and an ordered
source inventory. Method-specific perturbation providers, response drivers, and
source contributors remain outside this module. The executor still performs one
response solve, gives the same response object to every contributor, validates
finite Cartesian outputs, and publishes only after complete source coverage.

## Preserved contracts

- No public `Calculator` Hessian/HVP capability is added.
- No method name, functional family, grid, integral provider, or CPKS layout is
  introduced into the generic executor.
- Existing tools imports resolve to the canonical installed class objects.
- Missing/extra/duplicate sources and nonfinite or malformed Cartesian values
  remain fail-closed.
- Scientific RKS equations and resource qualification remain owned by the
  existing Hessian/response adapters and their parent issues.

## Rejected alternatives

- Copying the tools implementation into production would create two scientific
  orchestration owners.
- Making production import repository `tools.*` would break installed-package
  ownership.
- Promoting a public Hessian property in the same change would conflate an
  ownership migration with method-specific capability qualification.

## Evidence

The PR includes object-identity tests for the compatibility shim, a static
production-to-tools dependency rejection, the existing executor behavioral
suite, and repository CI. Parent numerical/resource acceptance remains in
#1402, #1406, and #1409; this ownership change does not replace those gates.

References: #932, #180, #1409, #1406, #1402.

Agent: ChatGPT
Model: GPT-5.6 Sol
