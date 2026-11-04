# Decision: compose native policy ownership with open-shell and scalar-ECP CBS

Status: implemented on the policy draft; endpoint qualification remains incomplete
Date: 2026-11-04

## Problem

The native-policy draft and master independently changed CBS composition and
precision ownership. Selecting either complete side of the merge would lose
native extrapolation coefficients or the already-integrated unrestricted and
scalar-ECP endpoint contracts. Master also removed the former Python SCF-default
selector, which the draft still imported.

## Decision

Keep the selected library's copied native CBS plan as the single coefficient
owner. Select RHF/MP2/CCSD(T), unrestricted counterparts, and fitted restricted
correction owners independently of those coefficients. Retain master's ECP
Hamiltonian identity, valence-electron admission, analytic-force, auxiliary-basis
and bounded resource behavior. This composition introduces no new mathematics.

Require the selected native SCF-default query alongside the draft's native
precision execution, candidate classification and observable comparison. Do not
restore Python selection or missing-query fallbacks to resolve the conflict.
Facade fixtures may inject explicit metadata receipts, but those fixtures are
not scientific or endpoint qualification.

## Invariants and validation

- Selected-plan propagation tests cover restricted and unrestricted component
  methods, energy/full-force weights, and the existing convergence-budget split.
- Independent Decimal extrapolation and native ABI tests protect the plan;
  existing independent PySCF/libcint endpoint gates protect CBS and scalar ECP.
- Validate the composed tree with a rebuilt matching CPU library, not a binary
  from either parent. Historical benchmark source identities remain historical.
- Preserve the Draft hold: the original eight-block CBS campaign and wider
  resource, real-GPU and complete-endpoint acceptance are not supplied by this
  conflict resolution or by CPU host regressions.

## Revisit when

The complete composed native-policy endpoint is independently qualified. Until
then, resolving integration conflicts does not authorize default promotion or
closing the parent runtime-retirement scope.

## Evidence storage composition

The actual merged evidence tree exceeded the unchanged 80 MiB aggregate cap by
33,347 bytes. Recompressing the existing historical CBS `records.json.xz` with
XZ preset 9 extreme saves 36,964 stored bytes without changing its path or a
single decompressed byte. Its decompressed SHA256 remains
`e989ee8b95a15761e6b19bfd77768a800a0e351c24d8e2168e7783e942772062`.
Only the compressed storage checksum/size in its publication and review records
change. Historical observations, failed candidates, independent oracles, sources,
decision fields and replay code remain unchanged. No payload is deleted, no cap
is raised, and no external archive or Release is created.
