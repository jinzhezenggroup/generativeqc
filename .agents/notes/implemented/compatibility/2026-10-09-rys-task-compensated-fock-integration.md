# Decision: compose Rys-task dispatch with compensated Fock output

Status: implemented; source and host integration validated
Date: 2026-10-09

## Problem

PR #2153 at `bf39f790553a8c4dd285ff206eaee417ff07c973` adds Rys-task
and task-work launchers. Master `ec71ef7fea6f705a3623b45c7bfb51105c43337a`
integrates #2161's runtime-owned `CompensatedOutput` Fock sink. A textual merge
upgrades the shared generated workers but leaves the new native declarations,
disabled stubs and generated extern-C declarations accepting `double*`.
The host selector then combines incompatible function-pointer types; accepting
the stale extern-C ABI would also misdescribe the generated kernel launchers.

## Decision and invariants

Carry `CompensatedOutput` through both new routes, including the single-profile
registry declarations. The existing shared emitter already upgrades task
workers, so no integral recurrence or task-queue change is necessary. Ordinary
K-only pointer callers retain the runtime constructor's null correction plane;
callers supplying both planes forward both pointers unchanged. Force output
remains a plain pointer.

Keep #2153's electronic-energy high/low state, canonical pair reuse, rolled
DDDS loops and Fock-specific `mixed_pair_products_fp64` choice. Preserve both
parents' test dimensions: Rys task counts 1/33/129 and ordinary/compensated
output are a cross product, with unchanged independent contraction tolerances.
The emitted host queue census retains task/work cases and the new sink ABI.

## Artifact audit

Regenerating each exact parent independently reproduces its ten full-bundle
hashes. The #2153 parent also reproduces its six retained-source hashes; master
has no separate retained-source fixture. The combined manifest and catalog
are byte-identical to #2153.

Relative to #2153, nine combined bundle hashes change: the four empty portable
shards gain the runtime compensation header, the four SM120 shards gain the
compensated Fock ABI, and registry source gains its corresponding declarations
and dispatch type. The generated registry header remains byte-identical.
Four retained-source hashes change (SM120 incumbent/component-Rys/K-block and
portable incumbent); the two empty portable alternatives remain identical.
The fixture records every final hash and preserves both correction histories.

All ten combined bundles and six retained sources recover the complete #2153
bytes after reversing only the compensation include, `Output` template,
`CompensatedOutput` signature, Fock dispatch alias and resulting line wrapping.
This checks the entire generated source, including queues and numerical bodies.
Relative to the initial textual merge, the final registry corrects 24 task and
task-work extern-C declarations; every mathematical shard is unchanged.

Canonical scientific identities from `generativeqc.autotune.source_identity`:

- #2153 parent: `6d517c95ad467049382bee31d10630150fd706a6e594988423f7d48564c09b8e`
- master parent: `a0fcb18029f89d056d3fb0cdddbc768e403c72c8bdb3c5c53d3364df1ac7512e`
- combined source: `afa64bf70ddb196e6450ed3f5f8260c69cee2a6137c26e38d55a93eb9a022163`

## Validation and limits

Native executables compile the real selector, generated registry, actual
emitted wrapper signatures and disabled stubs in separate translation units.
They cover all six routes, RHF/UHF block fallback, explicit correction and
pointer-only calls, checking pointer forwarding and selected output writes.
Emitted task-source checks and the existing mixed-product Decimal regressions
preserve precision and schedule semantics. These are ordinary host regression
checks, not GPU execution or performance qualification.

The frozen #2153 candidate remains
`d087c507d3e02b69b224a01d3dbb6bac478f152670452fc7a3710e38d152f9aa`.
Its GPU receipts and the separate master campaign retain their measured source
and binary identities. Neither campaign establishes timings or GPU acceptance
for this rebuilt combined source. No frozen receipt is rewritten or relabeled.
