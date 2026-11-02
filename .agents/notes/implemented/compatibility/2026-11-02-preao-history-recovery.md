# Decision: recover the historical pre-AO portfolio from merged Git

Status: implemented
Date: 2026-11-02

The tracked benchmark corpus was within 1,114 bytes of its fixed 80 MiB limit.
The October 6 pre-AO portfolio is historical evidence with no direct runtime or
test consumer; the two external references point to its retained README. Its
public and mechanism verifiers already accept an explicit restored directory.

All 86 original files (1,933,552 bytes), including source patches and the original
README, are pinned to merged revision
`8804b2382fbcfda1dc3daf1e0c4a9dc44c72e887` in
`benchmarks/results/preao-force-portfolio-20261006/historical.manifest.json`.
The existing restoration tool was run and every recovered byte was compared with
the checkout before removing duplicates. Both original offline numerical and
performance replays still complete. The README retains the publication identity
and local restoration commands; the original publication descriptor is recovered
with its payload to keep active publication tooling from following retired paths. In particular the 48/96-atom native
losses of 2.95%/4.80% against sampled execution remain visible.

This uses existing Git history, without an external publication, a Release or a
larger storage exception. Active scientific fixtures remain in the checkout.
Restoration authenticates the original lengths and hashes; it neither fetches
nor executes historical code. A shallow clone must first obtain the exact pinned
revision. New WB97M-V receipts retain complete endpoints and failed logs; optional
intrusive streams remain in their original ignored local archives with hashes
and explicitly labeled component aggregates. They are not clean timing gates.
