# Decision: prune unreachable stationary primitive JIT roots

Status: implemented; 24-atom complete E/F correctness qualified, no speedup claim
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

## Source-matched real-device qualification

Finite Slurm job2216 on node2/PRO6000 passes all 12 full-grid PBE0/def2-SVP
energy/analytic-force calls: cold, five warm, moved, five moved-warm. The
independent GPU4PySCF reference is generated in the same allocation. Maximum
energy error is 4.434e-12 Eh and force error is 2.167e-11 Eh/Bohr, against
unchanged 1e-8/1e-7 gates. The complete endpoint observer sees exactly one
nuclear-only emission, one geometry-only source owner and one geometry rebind;
forbidden AO-integral source emission/execution and CPU scientific consumers
are never entered. The source-matched library was built with ccache on n5.

- Tested production commit: `53ecd28d8`, based on master `8125e8e55`.
- Source: `06f9ef5a10321383be52f9edaa61edcc9a8f8814571a521540d4fe0cc72c790f`.
- Library: `b1a73ff4e99d2bfa75f8cf15885c0efa681e0677386fe5e276256d015211404f`.
- Reference JSON: `f785c189b4f2deaa8d6d50f643d35a9cd3073e4b70420cdb5f8fe9571edf9c70`.
- Native JSON: `edff20507c0abcf96545fc5fb5ece836c3ad8956457cb13220c836489b3ddd9a`.
- Root-demand receipt: `809c501556fd2737dd9faee4b9b9ea93d475f45cbabd87e1f5db48ace49ebeb2`.

Full raw arrays, scripts, identities and failure receipts remain local under
`.artifacts/qualification-2216/` in the isolated root-demand checkout; they are
not represented as a published replay bundle. Node2 timings are not RTX 5090
calibration, and there is no same-GPU old/new endpoint speedup measurement.
The preceding job2213 failed before forces with missing GCC-12 `cc1plus`, not
a numerical force mismatch; its incomplete endpoint is retained separately.
The repaired compiler bundle explicitly uses ccache and installed g++-11, with
a successful host-only NVCC preflight before2216. No n3 jobs were submitted.

Adjacent host gates: 133 pass. Compiler structure: 422 modules, zero errors.
Changed Python files pass repository-pinned Ruff 0.16.9 checks and formatting.

## Revisit when

A compiler-owned native coverage proof becomes available for small domains or
partial producer coverage. Extend demand by proven source coverage, not by
optimistic availability, while preserving nuclear and failure semantics.
