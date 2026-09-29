# Decision: GenerativeQC is the sole active project identity

Status: implemented
Date: 2026-09-29

## Problem

The former VibeQC name collided with an independent quantum-chemistry project.
The collision affected package discovery, commands, native symbols, generated
identities, documentation, and repository links. Issue #1548 requests a hard
rename without compatibility aliases.

## Decision

Use GenerativeQC and `generativeqc` throughout active source, build interfaces,
packages, commands, tests, CI, and current documentation. The old Python import,
CLI, C ABI, CMake options and targets are retired together. Repository and
Codecov URLs point to the intended new repository name; the GitHub repository
setting must be changed when the code rename is merged.

Keep immutable historical evidence in `benchmarks/results/` and historical
engineering rationale in `.agents/notes/` with their original bytes. Preserve
the original environment paths in `tests/reference_data/accuracy/*.json`, since
they are part of hashed reference provenance. These are the documented
exceptions to the stale-name audit: past run metadata, file hashes, snapshots,
and prior decisions describe the identity at the time they were recorded. New
active code must never consume an old name as a compatibility fallback.

## Rejected alternatives

Aliases, duplicate packages, and a deprecation period would preserve the
branding collision. Rewriting past measurement records would invalidate their
provenance and published hashes.

## Invariants

- Active source and current documentation use only the new identity.
- Historical evidence and notes remain traceable to their original commits.
- CMake, package, native symbol, and command names agree at each build boundary.

## Evidence

The rename PR records the active-tree case-insensitive name audit and CPU,
Python, and compiler validation results. Source-registry digest pins were
refreshed for 44 repository-owned inputs and outputs whose bytes changed. The renamed
reference schema also required new outer record hashes for 14 fixtures;
numeric data and hashed original environment provenance remain intact.

## CI follow-up: retained evidence and active identities

The first full Python CI run exposed a boundary missed by the focused tests.
Active fixtures whose schema spelling changed need their enclosing canonical
hash recalculated from their actual new bytes. The CC gradient, explicit grid,
XC integration, basis and local-space numeric arrays did not change. Their
active method bindings must agree with those new identities.

Published benchmark bundles and evidence archives are different: their original
schema and nested hashes are part of the measurement. Their readers validate
the original `vibeqc.*` schema as recorded, including reconstructing the old
explicit-grid schema when checking density workload hashes. This data-reading
boundary is not a Python package, CLI, native ABI or CMake compatibility alias.
Historical result files remain byte-for-byte unchanged. Source migration
receipts record the exact current exporter bytes, including the deliberate
identifier and schema rename; no new scientific reference execution is claimed.

A synthetic test that pinned a literal hash of a hand-built cache payload now
checks the canonical-hash relationship and mutation behavior. Tests that
verify reference arrays, published attachments, native source provenance or
generated source reproducibility retain exact hashes.

## References

- Issue #1548.
