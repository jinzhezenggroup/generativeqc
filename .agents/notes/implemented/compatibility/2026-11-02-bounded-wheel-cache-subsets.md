# Decision: publish bounded wheel compiler-cache subsets

Status: implemented
Date: 2026-11-02

## Problem

Master wheel run 41368495181 saved an Actions cache of 2,526,344,480 bytes,
then its optional trusted snapshot exceeded the 2 GiB-minus-1 MiB encoded
limit after approximately 55 seconds of packaging. The live compiler cache
allows 3G; already compressed entries need not compress below the snapshot cap.
Actions cache transport size and ccache's statistics are not archive inventory
sizes and must not replace actual filesystem accounting.

## Decision

After the existing complete-source decoded-size and file-count checks, select
recent wheel cache records using actual file sizes, USTAR framing and
conservative gzip budget headroom. Retain the existing encoded LimitedWriter
as the final size gate. Record only the selected inventory in the manifest.
Do not modify the live cache. Missing records cause ordinary strict compiler
cache misses. CPU snapshot behavior is unchanged.

The complete source inventory still fails closed above 3 GiB or 100000 files.
This change addresses encoded overflow inside those bounds; it does not promise
a snapshot for an arbitrarily large source cache or an individually oversized
record. Recency selection is a bounded heuristic, not an optimal hit-yield claim.

## Invariants and rejected alternatives

Keep successful-master provenance, pinned toolchain and architecture identity,
checksums, no-config imports, hostile-archive checks and atomic activation.
Do not increase security bounds, clean the consumer cache, relax compiler
validation, or import PR-owned data into trusted merge-queue snapshots.
Keep all scientific, full-XC, architecture, build and linkage gates unchanged.

## Evidence and limits

The focused CPU/wheel snapshot suite passes 146 tests under ordinary Python
and optimized Python. Added tests cover incompressible subsets through real
package/verify/activate, source byte/mtime preservation, oversized-record
fallback and deterministic equal-recency ordering.

A local 32 MiB incompressible fixture under a reduced 16 MiB archive limit
failed with the previous implementation and produced a 14,685,486-byte subset
in three candidate trials. A separate host ccache 4.12.2 probe retained 19/40
records; all 20 rebuilt object hashes matched, with five hits and 15 ordinary
misses. CI uses 4.14; these are bounded local integrity probes, not production
CUDA or end-to-end speedup qualification. Measure retained hit yield and total
restore/build time on real wheel caches before claiming a CI improvement.

## Reference

https://github.com/jinzhezenggroup/generativeqc/actions/runs/41368495181
