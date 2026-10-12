# Decision: select stationary worker normalization from verified spin metadata

Status: implemented
Date: 2026-10-12

## Problem

The restricted native SCF descriptor fixed total-density normalization, but
`NativeKsSnapshot.evaluate_xc_points` still called the polarized v2/v3 batch
bridge. A real converged CPU RKS snapshot therefore evaluated its stationary
features with a different worker than SCF in the active total-density tail.
At total rho=1.1e-15, the complete point energy differed from the independent
PySCF 2.14.0 / Libxc 7.0.0 unpolarized reference by about 22% for VWN4,
Chachiyo and modified PZ. Ordinary molecular interior checks hid this error.

## Decision

Add a separate private layout-aware v4 bridge, preserving v2/v3's polarized
semantics and layouts. The live snapshot supplies its verified functional spin;
standalone helpers retain the polarized default. Restricted feature transport
must contain two equal half-feature channels. The bridge restores total rho,
gradient and tau before invoking the exact registered U1 descriptor and
duplicates its coefficients into the existing 11-value transport. Contraction
against both half-feature directions then gives the total-feature derivative;
kinetic retains the native vtau/2 convention.

The density-work descriptor must match SCF's domain version and ingredient
mask. Missing v4 or a matching restricted descriptor fails explicitly. Legacy
work domains retain their SCF spin-fold continuation rather than acquiring an
unqualified new scientific contract.

## Rejected alternatives

- Infer RKS from equal rho: an equal-spin UKS producer still has P2 semantics.
- Change v2/v3 globally: silently reinterprets existing polarized callers.
- Fix the oracle or average polarized tail derivatives: leaves the wrong worker.
- Fall back to P2 when v4 is missing: a stale library would appear current.

## Evidence

The real live-snapshot regression failed numerically for all three LDAs before
the fix, with tail energy errors of 1.28e-21 to 1.69e-21. The permanent tests
bind the same live source to independent pinned unpolarized E/vrho references
in vacuum, the active tail and the ordinary interior. Additional executable
tests cover equal-spin UKS, direct native layout rejection, stale-library
failure, and unchanged legacy GGA/tau-MGGA behavior. Raw qualification receipts
are retained under the ignored owned `xc-point-layout-phase-pr-20261012` cache.

The preceding Ready PR's CI also exposed a stale density-only public selector
expectation and an obsolete snapshot constructor digest in the existing frozen
DFT-MP-v1 preflight audit. The selector now checks each family's exact domain,
not a permissive version set. Only the already existing constructor baseline is
rebased to the audited domain-selection change; decode, fitted reserve, grid,
capacity and mutation-rejection gates remain intact. No new AST/body-hash tests
or numerical tolerance changes are added.
The existing extracted-constructor fixture also lacked the new domain constants.
It now executes the real constructor against native ABI stand-ins, retaining
copy/cleanup assertions and adding cross-domain stale and unknown-domain cases
without asserting import spellings or AST layout.

## Invariants and remaining scope

No runtime Libxc/PySCF oracle dependency, new scientific algebra, descriptor
layout change or capability promotion is introduced. Point/source correctness
does not clear physical particle-birth, complete molecular force, precision,
automatic CUDA RKS provider, or installed/AOT/cache closure gates.

## References

- #1118, #1122, #1124.
- Extends `2026-10-12-restricted-density-native-abi.md` to its stationary consumer.
- `docs/developer/libxc_bulk_capabilities.md`.
