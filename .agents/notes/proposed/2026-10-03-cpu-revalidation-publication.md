# CPU revalidation publication boundary

Status: proposed; numerical and descriptive endpoint evidence retained
Date: 2026-10-03

## Source identity

The clean measurements use baseline commit `9a5871dca1f371e5f6e9fcf9d6b076c83b894c14`
and local candidate commit `a7c118aa3076dbfdc13e47ed2d1056c3b0fe6e80`, whose tree is
`0680717ea6cda4e6abb2d3c478defba37a02b755`. The publication integration contains
later reviewed upstream work and the approved lossless post-HF evidence cleanup.
It was not the timed binary. All sixteen original candidate files remain
byte-identical to the measured snapshot.

The GitHub connector may create different commit author/time metadata. A remote
snapshot is an equivalent source reference only after its complete Git tree is
verified equal to the measured tree. Preserve the original measurement IDs in
the scalar records and distinguish that remote alias from a separately timed
commit. New builds must record their own binary hashes.

## Storage and acceptance

The five-file scientific capsule under
`benchmarks/results/cpu-revalidation-20261003/` retains the clean cohorts,
invalid pilot, failed final-validation case and exact runner sources. It uses
deterministic gzip of an indexed scientific JSON graph with hash-checked,
readable expansion. It contains no density matrices, build products, logs,
profiler dumps or raw run archives. The directly readable summary includes
the slightly slower LDA control and all qualification limits.

The current indexed retention policy, rather than stale documentation, is
authoritative: 1 MiB per file, 2 MiB incoming evidence review, 64 MiB aggregate.
No limit or exception was changed for this campaign. The integration includes
the separately approved post-HF lossless cleanup; do not assume this change is
admissible against a different upstream tree without checking it again.

Public-HF telemetry does not satisfy the stricter compiler/default-promotion
envelope. Numerical and descriptive endpoint evidence is retained, while the
formal performance-promotion decision remains inconclusive. The six-pair clean
comparisons, not the contaminated HF96 pilot, support the reported medians.
The PBE96 failure is a shared baseline/candidate final energy gate failure after
25 iterations and two final builds; its configured 150-step limit was not
exhausted and no failed endpoint is a performance sample.
