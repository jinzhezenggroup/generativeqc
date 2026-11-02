# Decision: recover the historical public force-policy payload from merged Git

Status: implemented
Date: 2026-11-02

Current A100 stage evidence and the WB97M-V follow-up must fit the unchanged
80 MiB evidence limit. The October 5 public force-policy campaign is historical;
all remaining production, maintenance and note references target its README.
No runtime or regression test reads its raw payloads. The offline verifier uses
paths relative to its own restored directory.

The manifest in `benchmarks/results/pbe0-public-force-policy-20261005/` pins all
63 original files (696,731 bytes) to existing merged revision
`195bdc95cbd2b255d37a338bdca1244b7a15f467`. Every blob was compared against the
checkout, restored through the existing authenticated Git recovery tool, and
replayed before removing duplicates. The two offline reports are byte-identical.
The original 864 endpoint and 144 profile comparisons, cold regressions, source
patches, full vectors and profiling CSVs remain recoverable without a Release,
external backup or raised storage limit. The original README is also recoverable;
the retained README adds restoration instructions and keeps the historical
numerical and timing conclusions unchanged.

Active scientific inputs remain in the checkout. This storage decision does not
renew historical device/default qualification. Revisit if a live consumer needs
the original payloads directly; restore verified bytes before using them.
