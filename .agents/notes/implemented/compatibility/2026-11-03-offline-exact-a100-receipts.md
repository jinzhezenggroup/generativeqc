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
