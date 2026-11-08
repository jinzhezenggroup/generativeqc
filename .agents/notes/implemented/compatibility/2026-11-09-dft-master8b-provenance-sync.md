# Decision: synchronize DFT with explicit offline CUDA provenance

Status: implemented; metadata integration, not a new GPU measurement
Date: 2026-11-09

## Problem and decision

After publishing DFT evidence head5424c9881, master advanced from2bf3fd98d to
8b9983820 (#3050). Merge that actual upstream commit without rewriting either
branch. Preserve its explicit installed-CLI compiler/image provenance, refusal
of schema-v1 profiles and serialized bounded historical fetches. Resolve only
the generated Libxc catalog conflict using the official writer/checker.

The accepted master8b summary exactly equals the existing previous-summary
payload. Rebind its source/scope to8b while retaining its original checksum,
campaign, measured source/ELF, historical bindings, empty acknowledgements and
all capability/stage sets. Both catalog writer and normal checker pass. No
stale native claim is promoted.

## Scientific and measurement boundary

The merged source identity is
`8b1b284f5198dda7bb167f8c3980db36dda5164462f3774a745a5d62553eab93`.
It is not the qualified native image75d94b32 or rejected image9d11d755.
Retain their original complete endpoint measurements under their actual image
identities; do not call them timings of a rebuilt merged image. No new GPU
campaign occurs solely for this integration.

CPU-only audit loads the historical and current profile parsers over the same
actual schema-v2 manifest and compares every resolved dataclass field for
explicit targets80/90/100/120. All selections match exactly (21/21/21/44).
This establishes plan parity, not linked-ELF, real-device or latency parity.
The upstream CMake addition writes installed CLI build metadata; it introduces
no new integral/derivative algebra, screening or precision policy. Future
performance claims for a newly compiled image must retain that image's identity.

## Evidence and references

- #2976 and PR #3051; master8b9983820 and prior evidence head5424c9881.
- `benchmarks/results/dft-adjusted-becke-20261109/master8b-plan-audit.json`.
- Ignored reproduction: `.artifacts/dft-adjusted-becke-20261109/audit-master8b-plans.py`.
- The rejecting Becke replay and original source-bound streaming qualification
  remain unchanged. This integration does not establish all-README leadership.
