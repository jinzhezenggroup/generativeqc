# Decision: detach the exact CUDA RHF source for conventional CC

Status: implemented
Date: 2026-10-04

## Problem

CUDA RHF already uploads normalized public-AO metadata, but conventional CC
previously constructed a second `PreparedFockPlan` after RHF. That repeated
one-electron integrals, host basis packing and device setup. DF correlation
also constructed a `RawSource` just to recover its orbital `System`.

## Decision

The single-system reference adapter optionally detaches immutable metadata from
its fresh, converged RHF bucket. A compact device allocation reuses the existing
public-AO raw ERI producer through a source-only Direct plan. The adapter moves
its existing system snapshot. One bounded D2D compaction releases the much larger
SCF/DIIS arena; it does not recompute integrals or repack/upload the source.

Conventional CC retains that source through its force Hamiltonian consumer.
DF correlation instead passes its original System directly into the DF builder;
it does not construct an unused conventional source. The CPU prepared path keeps
its existing owner. Retire the obsolete second-plan capacity planner.

## Invariants and fallback

- Source reads are unscreened FP64 public-AO ERIs, on the requested same-device
  stream; the handoff exports no density, screening state or mutable SCF arrays.
- The reference owns one-electron matrices; the source advertises device ERIs only.
- Admission charges the reference peak plus compact retained source, including
  simultaneous allocations during compaction. Capacity is not a measured peak.
- Budget/allocation failure leaves the source empty and preserves the successful
  RHF reference. Conventional CC retains an explicit RawSource fallback. Later
  provider/CC admission may retire the optional source before retrying.
- Failed arithmetic/driver calls do not trigger this resource fallback. Consumers
  fence their stream before releasing their last shared source reference.

## Evidence

RTX 5090, Slurm jobs 12244/12247/12249: independent CPU ERIs for s/p/d/f,
Cartesian and spherical bases after the RHF arena is destroyed; changed geometry
without mutating the old source; wrong-device rejection; exact and one-byte-short
handoff budgets. Compute Sanitizer memcheck: zero errors. Public RCCSD(T) and
DF complete-force suites: 38 passed, including independent analytic gradients,
energy finite differences, changed-geometry/batch behavior and failure publication.
Five host owner-lifetime/admission tests passed, including retained-source overflow
and optional provider failures. These establish correctness, not a large-endpoint
speedup claim. Reproduce the direct source probe with
`GENERATIVEQC_RHF_SOURCE_HANDOFF_TEST=1` and `GENERATIVEQC_LIBRARY` inside Slurm.

## Rejected alternatives and revisit conditions

Retaining the full RHF bucket unnecessarily retains iteration storage. A second
prepared Fock plan retains duplicated setup. Separating immutable RHF metadata
into a shared allocation could remove the remaining D2D compaction, but requires
changing the general bucket arena lifetime and admission contract. Revisit that
when other consumers need the same shared metadata owner.

References: #1500 source reuse; #1503 warm-state lifetime; stacked on #1818.
