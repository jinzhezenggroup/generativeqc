# Decision: preserve held-density factors and allow explicit fitted range caching

Status: implemented; range cache remains explicit
Date: 2026-10-23

## Problem

WB97M-V physical energy confirmation holds D unchanged but runs an eigensolver
that overwrites the proposal scratch. The previous implementation correctly
refused that unaccepted proposal as a factor, but also discarded the retained
AO frame that generated the unchanged D. The next occupied K became dense and
the final force lost its authenticated U projection. At 96 atoms the resulting
exchange response took about 65 seconds.

The independent long-range Direct correction also failed optional-cache
admission whenever the full-range primary used DF, even though its source and
resource contract were independent. Reusing a range cache then exposed another
unnecessary dense primary rebuild during warm-energy rebasing.

## Decision and invariants

Restore only a retained restricted AO frame consumed by the preceding occupied
K when the density is deliberately unchanged. Clear readiness before copying;
unsupported frames keep their previous invalidation behavior. This preserves
the original D rather than accepting a nearby canonicalized proposal. The same
final density, stationarity, source, scratch-generation and projection gates
remain mandatory.

Allow explicitly forced, strict range residency to use its independent Direct
provider while the primary is fitted. Automatic DF admission remains unchanged.
Include both providers' host inventory, respect actual free memory and the
complete numeric allowance, and retain the existing exact fallback for every
omitted angular bucket. No grid, metric, precision or convergence control changes.

At warm rebasing, a range-only cache did not change the primary's value schedule.
Retain its current J/K only when the fitted final projection and warm factor are
present (pure J needs no K projection). Rebuild the independent range term and
record only the operators actually executed. Optional range storage can retire
at a pressured force handoff without invalidating the final fitted U lease.

## Diagnostic evidence and rejected alternatives

All initial runs used the frozen experimental source manifest based on
`8b68f626c82e7bf0f33bd6a993f91ec64b5c790b`, before subsequent integration.
These are traced, single-process complete E+force observations, not clean repeated
performance claims. The held-factor repair changed WB97M-V DF96 cold/warm from
694.984/105.959 s to 625.208/45.144 s, with 28 cold iterations. Exchange response
fell to about 3.17 s, borrowing 3,554,672,640 bytes of final U and using
1,920,039,424 response scratch bytes. Projection work fell from roughly
69.5 trillion to 142.2 billion logical FLOPs.

Explicit capacity-selected range buckets gave 501.434/45.398 s. Mask `0x3fcf`
gave 453.668/45.503 s and `0x3f` gave 476.016/46.744 s. All samples passed the
retained independent dense reference: energy error per atom about 3.009e-6 Eh,
force maximum about 1.995e-5 and RMS about 9.392e-6 Eh/Bohr. These masks are
diagnostic controls; no molecule-specific mask is promoted. The `0x7f` request
exceeded its actual free-memory allowance and correctly fell back, taking
637.805/45.817 s. GPU4's retained cold observation is about 392.171 s: the goal
remains unmet.

The focused native `--df-wb-composition` entry point checks restricted and
unrestricted fitted full-range plus exact long-range exchange and VV10 against
the independent CPU implementation. The broader synthetic PBE range fixture
fails final token model validation on both the original stage-two core and the
experimental core; that baseline failure is retained and the original default
suite remains unchanged. Host tests execute held-frame restoration, refused
proposals, dense/UKS/missing/orthonormal frames and copy failure. The reviewed
begin-failure test fix from stage one is preserved during integration.

## Revisit when

Promote automatic fitted-range caching only after a general resource/work policy
passes source-matched clean repeated cold/warm/moved endpoints and independent
numerical gates. Historical traced masks do not establish a universal selector.
The full integral and force consumer must remain bounded under actual memory
pressure, and unsuccessful optional preparation must leave the exact path usable.

The later [automatic fitted-range candidate](2026-10-23-automatic-fitted-range-candidate.md)
extends admission after a compiled stack-headroom pilot. That candidate retains
its own default-qualification requirement; the explicit pilot populations above
remain historical evidence for their original source and controls.
