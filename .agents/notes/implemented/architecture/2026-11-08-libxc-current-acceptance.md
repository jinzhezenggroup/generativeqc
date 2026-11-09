# Decision: Separate Libxc archival consistency from current-source acceptance

Status: implemented
Date: 2026-11-08

## Problem

Issue #1118's implementation lanes were already delivered. Its remaining gate
requires current native energy/force, spin/domain and lifecycle evidence. The
offline catalog correctly demotes stale measurements but still exits successfully
when its manifest, demotions and rendered report agree. Such an exit status is
an archival-integrity result, not current-source scientific acceptance.

## Decision

Keep the existing default checker as the truthful archival gate. Add the opt-in
`--require-current` acceptance gate to the same checker, after authenticated
receipt ingestion and regression checks and before writing any updated artifacts.
It requires fresh checked source/native identities plus admitted paired energy
and force products for each ordinary LDA/GGA/tau-MGGA family, CPU/CUDA backend and
RKS/UKS layout. Pair products within an identical full context, including the
registration, binary, geometry, domain and receipt identity. Observed-but-blocked
contexts do not count.
Each family must have at least one common representative across both backends
and spin layouts; unrelated registrations cannot fill the cross-backend matrix.

## Rejected alternatives

- Requiring a new molecular campaign for every imported registration would turn
  representative generic-path qualification into an unnecessary named allow-list.
- Requiring `public-method` receipts would confuse optional stronger evidence
  with structural public selection.
- Treating explicit stale-evidence acknowledgements as fresh measurements would
  manufacture qualification without an independently measured current binary.
- Making freshness mandatory in ordinary archival CI would prevent retaining
  truthful historical campaigns after unrelated source identities change.

## Invariants

- Existing numerical gates, source/Graph/native identities and independent
  PySCF/Libxc references remain authoritative; this change relaxes none of them.
- Default-allow public selection is unchanged. No second-order, empty-spin,
  higher-angular-momentum, DF, ECP or mixed-precision capability is inferred.
- A paired context cannot be assembled from separate registrations, geometries,
  binaries or lifecycle receipts.
- Build logs and binaries remain ignored local artifacts, not release assets.

## Evidence

Build features must be checked independently of source identity. In particular,
disabling `GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT` also makes the initial default
of `GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA` false. A bounded package without
Python-consumer plug-ins must explicitly enable the native option and select
matching automatic stationary profiles. Otherwise CPU measurements can pass
while the current binary correctly advertises no CUDA force provider. Such
measurements must not be relabeled as receipts for a subsequently rebuilt ELF.

Public API tests now exercise both present and absent automatic package models
instead of assuming all native builds lack automatic CUDA forces. The absent
case still proves that a curated carrier cannot manufacture force admission.

Retained authenticated campaigns exercise the executable acceptance policy in
`tests/python/test_libxc_native_catalog.py`. Tests retain its authenticated
historical-source seam and reject stale evidence, missing families/backends/spins/
products, merely observed endpoints and mismatched paired contexts. A representative
three-registration subset suffices without `public-method` qualification.

A pre-integration local trial measured source
`7bebadd4ae9f0368c63eee4b217534d69402b83d` and force-capable ELF
`c901ac502139724348d5918162f0b4e48b64dd48fd905a938ecf8feb6acd689b`.
Its 11 registrations passed 44 CPU/CUDA spin contexts, 88 paired energy/force
products and four separate resource tests. These local results are not relabeled
as evidence for the subsequently integrated source or rebuilt binary. Only a
fresh integration campaign may populate
`benchmarks/results/libxc-native-catalog-20261109/`. The preceding published
campaign remains immutable historical evidence; neither measurements from an
earlier energy-only binary nor skipped diagnostics are promoted.

### Integration qualification (2026-11-09)

The retained campaign freshly measures integration source
`fbf57c111c7c885efbc0cd1457e6a7732c9e3d95`, inventory
`ba63873931ccffae7d9b2cea01831028f9bcf04a9903b818da76bd1539f64d64`,
and exact ELF
`c2d1b725fb754e72d35ca5829ba9a067974703c6a98bc4e4cefb44ea6abc256c`.
All 11 registrations pass, yielding 44 admitted CPU/CUDA spin contexts and
88 paired E/F products with `stale=False`; four separately source/binary-bound
resource tests pass without skips. Normal Release compilation keeps
`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, verifies sccache 0.16.0 on all 607 compiler
commands and discovers all 22 selected native force providers. The README
records finite Slurm allocations, independent oracles, exact merge selection
from complete checkpoints, unchanged gates and explicitly unmeasured domains.

## Consequences

An acceptance run now has an explicit failing status when only stale reporting
is available. Future source changes can still deliberately demote retained
measurements without pretending to satisfy the stronger gate.

## Revisit when

A separately qualified domain needs a different acceptance matrix. Extend that
domain explicitly rather than inferring it from an ordinary molecular receipt.

## References

- Issue #1118 and its source-bound catalog gate from #2861.
- `tools/check_libxc_native_catalog.py`.
- `docs/developer/libxc_bulk_capabilities.md`.
