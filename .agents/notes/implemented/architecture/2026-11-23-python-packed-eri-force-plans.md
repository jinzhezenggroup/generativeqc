# Decision: Reserve packed CPU force capacity before Python replay

Status: implemented
Date: 2026-11-23

## Decision

The packed values stay owned by the existing native prepared Fock plan. The
new native fleet query charges their exact sum and the maximum serialized
workspace across preparation, SCF and stationary forces. The force envelope
includes the reverified spin frame, the integral owner's existing 64 MiB
publication cap, and the AO and Becke buffers that overlap during the XC
pullback. Native force results can survive across fleet items, so charge their
vectors for every item. Explicit corrections retain their additive resource
owners; do not hide their storage in the electronic envelope.

This is a bound on declared numeric owners, not a complete allocator census.
Integral recurrence temporaries, BLAS/runtime internals and caller-owned
storage are excluded. Keep partial native capacity observations, including
the Becke allocation while the last XC tile remains live. Do not price the
Python compatibility derivative path into this native plan or read a force
capability from its byte count. Every item must pass the prepared native
capability check; an energy-only packed batch cannot upgrade to forces.
Budgeted scalar force requests use the same explicit force plan as the batch.

## Evidence

On base `cae2959e1`, 10 CPU CTests and 256 resource/API/source-identity pytest
cases passed (14 skipped); four separate installed SDK/CLI/Python force
comparisons against PySCF passed. The native tests verify packed cap and
transactional failure, serialized lifetime across a ragged fleet, and a
large AO tile exceeding the integral cap. Python tests check exact host and
packed caps, one-byte refusals before native work, rejection of disabled
export, scalar entry, and cold, warm and moved geometry replay without
compatibility derivatives. Energy-only replay does not retire the reservation.

Do not reconstruct a validated force request at every replay: the model,
topology, native library and planning inputs already fix its byte envelope.
Keep it local to the prepared owner and revalidate if inputs change. The test
checks zero repeated queries across ordinary replays and one on environment
change. This reduced the measured per-sequence admission overhead on the tiny
PBE0 and corrected B3LYP UKS controls. It does not eliminate initial admission
and external-plan validation overhead.

Seven interleaved pairs per case on one EPYC 7K62 core used the same native
library and returned identical energies, forces, iterations, Fock builds and
warm-state selections for cold, warm and moved-geometry replay. From
calculator construction through publication, plan admission increased the
median by 5.7% on ragged PBE0, 14.2% on tiny corrected B3LYP UKS, and 0.36%
on corrected CsH B3LYP RKS with scalar ECP, compared to unbudgeted packed
execution. Ordinary budgeted PBE and corrected B3LYP energy controls changed
by -0.13% and +0.23% in same-library adapter comparisons. CsH admitted at
121,456,948 modeled host bytes with 758,640 packed bytes; sampled replay
capacities peaked at 4,676,190 bytes. Its three replays used 16, 4 and 11 Fock
builds. Raw data are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/python-packed-force-bench`.

## References

- [Packed energy-only host inventory](2026-11-22-python-packed-eri-host-inventory.md)
- [Native force CLI](2026-11-22-cli-packed-eri-forces.md)
