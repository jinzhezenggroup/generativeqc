# Decision: explicitly audit independent README reference environments

Status: implemented
Date: 2026-10-30

## Problem

The native and GPU4PySCF benchmark engines use separate installations. A strict
package-map equality check rejected valid independent numerical references
because the production installation intentionally did not contain GPU4PySCF.
The A100 allocation is managed by LSF, while the local devices use Slurm;
requiring a Slurm-only identifier also discarded genuine scheduler receipts.
Neither problem justifies altering archived references or inventing job IDs.

## Decision

Retain the existing strict same-environment reuse contract. An explicit v2
reference-reuse manifest additionally records the exact reference package map,
a list of exact accepted measurement package maps, and a nonempty scientific
input audit. Every entry is keyed by the SHA256 identity of the complete,
unmodified original oracle receipt and still binds its original source/driver,
method, basis/representation, charge and spin. Unknown package maps, missing
review or changed scientific inputs fail validation.

The benchmark drivers retain real Slurm or LSF allocation fields and the
scheduler-provided CUDA visibility. An LSF measurement never receives a fake
Slurm job identifier. This is provenance support, not permission to bypass a
scheduler or to run tests on an unallocated GPU.

## Invariants and evidence

No numerical threshold or production scientific setting changes. Qualification
still recomputes every sample's energy/force errors, reference grid stability,
and matching CPU/GPU reference spot checks. Missing, unstable or disagreeing
oracles remain unverified; inaccurate production calls remain failed. A stored
`passed` field never overrides a recomputed gate. Tests explicitly cover these
failures under the new environment contract, as well as receipt immutability,
unreviewed package maps and missing scheduler identity.

The current campaign audits the retained `readme-refresh-20261026` references
against their existing `readme-final-20261024/reference-reuse.json` contracts.
The original driver at revision `3928aea6a` has SHA256
`3559d86aaa55e5c07231a1268bfb4f20a9945f4720daf0e160368472548dabdf`.
The method/basis recipe, scaling geometries, r²SCAN-3c basis and independent
D4/gCP corrections remain unchanged. This audit allows reuse of those exact
receipts; it does not assert that every package version or future driver is
scientifically interchangeable.

## Consequences

Native/reference package isolation is reviewable without falsifying provenance.
Raw reference receipts retain their original identities. A new environment or
scientific recipe requires a new explicit audit or new reference calculations.
