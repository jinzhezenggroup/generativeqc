# Decision: retain frozen A100 receipts through authenticated local Git

The DFT96 and DF96 branches independently exceed the unchanged 80 MiB evidence
checkout budget after H100 guard delivery. Retain the original 868,556-byte A100
molecular receipt stream in already-merged Git history rather than raising that
budget, deleting unfavorable observations, rounding vectors, or publishing a new
external archive.

The existing offline snapshot loader authenticates its original size, SHA-256
and Git blob. The shared exact-FP64 decoder accepts this only through an explicit
manifest supplied by the source-bound A100 reader. Ordinary bundle readers do
not discover snapshots, fetch history, or import recovery machinery implicitly.
Existing stream and reconstructed-text digests remain authoritative, including
when a changed local file is present. Missing Git objects fail with the existing
explicit recovery error rather than silently skipping scientific acceptance.

Before migration, normal and optimized CPU report output was identical. Both
must remain byte-identical to that baseline after recovery. This changes storage
only: the original source identities, complete vectors, native/molecular failure
records, independent RSH stencils and acceptance gates are not requalified or
relabeled as current-source GPU evidence.

## Composed merge-queue retention

The H100 final cohort and long-range WB97M-V integration share one historical
Nsight deletion, so their storage savings cannot be counted twice. Their combined
checkout plus the pending DFT96/DF96 descendants needs additional headroom.
Extend the same explicit per-part A100 recovery to failed-molecular and native
receipts. The existing offline host-LR, host-full and disk-ERI readers already
accept explicit snapshots; retain their original compressed packets through
that path too. All seven packets total 1,999,824 bytes, now authenticated at
already-merged revision `8c6c6208bb1d68c96e62ed2d15ee561fcc784081`.

Each original Git blob, size and SHA-256 is checked against the present packet
before removal. Normal CPU reports must remain byte-identical for all four
families, as must optimized A100 replay. The host/disk readers intentionally
require assertions and are not qualified under `-O`. This is storage migration,
not new performance evidence, and does not raise the 80 MiB aggregate limit.
