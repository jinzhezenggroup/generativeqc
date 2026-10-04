# Decision: prune unreachable stationary primitive JIT roots

Status: implemented; complete endpoint qualification pending
Date: 2026-10-04

## Problem

The ordinary stationary CUDA endpoint requires complete native integral sources
when atoms exceed 32, AOs exceed 128, or basis primitives exceed 4096. It rejects
an unavailable native producer instead of executing the AO-task fallback. Yet
both prepared and unprepared JIT paths compiled the entire primitive derivative
inventory before calling that producer. The composite owner already compiles
only the nuclear primitive for its native-integral path.

## Decision

The method compiler projects the primitive roots to nuclear repulsion when the
complete native producer is mandatory and the artifact is JIT. Runtime passes
the same demand to source emission and the existing geometry-only source owner.
The prepared schedule identity includes primitive demand. Scientific source
coverage, native admission/failure checks, and final reduction remain unchanged.

Ordinary basis validation still runs through the original full layout first:
this change does not inherit the composite geometry owner's f-shell capability.
Small domains retain the complete bounded AO fallback, even when a native
provider usually succeeds. Packaged artifacts keep their exact existing request
numbering and never discover a compiler or emit source.

## Work and memory

For a Cartesian domain of size C, the existing metadata inventory still examines
3 C^2 + C^4 tuples before symmetry canonicalization. This patch does not claim
to eliminate that metadata work. For mandatory-native JIT, primitive graph
emission and compilation change from R canonical derivative roots in ceil(R/16)
SPD units to one nuclear root/unit. Native integral, AO/grid and geometry work
are unchanged. No new device buffers are allocated; the original resource
bounds are retained, rather than treating smaller code as a device-memory win.

The host-only census retained during #1830 on node2 found 362 SPD requests:
313 ERI, 16 each overlap/kinetic/nuclear-attraction, and one nuclear. Source
size was 23,002,437 bytes in 23 units versus 1,550 bytes for nuclear-only.
These are emitted-source counts, not executed integrals or FLOPs. Full emission
took 39.893 seconds on node2; nuclear-only emission followed in the same process
with warm caches. Those times are not a causal comparison, an RTX 5090
calibration, or an endpoint speedup claim.

## Correctness and qualification

- Host tests cover both primitive ABIs, mandatory/optional producer boundaries,
  packaged numbering, prepared emission, nuclear retention, failed-provider
  rejection without publication, and different prepared schedule identities.
- Existing AOT no-compiler, d-shell, geometry reset, resource and lowering gates
  remain required.
- Before promoting performance claims, use independent complete E/F oracles,
  cold/warm/moved/moved-warm endpoints, and a source-matched library. Verify the
  actual owner is nuclear-only, retains its identity on geometry refresh, and
  never enters CPU scientific fallback. Keep failed attempts and cold negatives.

## Rejected alternatives

Do not prune merely because a provider is callable: small domains can legally
fall back when resource admission fails. Do not silently repack an AOT library
or broaden basis qualification. Do not call this a warm-gap solution: #1830's
48-atom warm endpoint remains about 2.76 times GPU4PySCF, and this change does
not reduce executed integral or grid loops. That evidence is a different frozen
source snapshot, not a measurement of this patch.

## Revisit when

A compiler-owned native coverage proof becomes available for small domains or
partial producer coverage. Extend demand by proven source coverage, not by
optimistic availability, while preserving nuclear and failure semantics.
