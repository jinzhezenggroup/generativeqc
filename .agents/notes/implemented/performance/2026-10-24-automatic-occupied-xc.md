# Decision: automatically request authenticated occupied-XC features

Status: implemented; current-source endpoint qualification retained with this change
Date: 2026-10-24

## Problem and decision

Master `72e0a7e5a011592428b5625acdfa6f19168645ac` already contains the
independently tested occupied-XC implementation from #2703, but only explicit
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` requests it. Ordinary strict-FP64 meta-GGA
cold solves therefore keep performing AO-density matrix products even when
an authenticated smaller canonical occupied frame is already resident.

Make unset select the existing request. Explicit `0`, and other unsupported
explicit values, retain matrix contraction; `1` keeps its previous meaning.
No mathematical emitter, reduction, compiler schedule, resource reserve,
quadrature, screening, convergence criterion or reference-engine option changes.
A tile uses the factor only below its actual active AO count. For four jets,
its projection work is proportional to `4 * points * active * occupied`
instead of `4 * points * active * active`; Vxc still uses the AO basis.
This algebra describes the selected domain, not an executed hardware FLOP count.

## Lifetime and fallback invariants

Only the canonical frame that formed the accepted D can lend its scratch.
Every completion retires the lease; accepting a new proposal may republish it
only if final probing did not overwrite that frame. Density pointer, generation,
shape, occupations, spin and alias checks remain. Imported/held densities,
rejected proposals, unsupported/mixed/nonlocal/response consumers and tiles
without a smaller factor retain the existing bounded matrix route. No CPU
oracle or additional allocation enters production.

The first frozen warm call consumes a held density and has no live proposal
frame. A warm UKS solve can require further SCF iterations; a newly accepted
proposal can legitimately supply a later frame. The new public-default test
checks the first held frame rather than incorrectly requiring every warm
iteration to decline. The original overstrict assertion and its corrected
rerun are retained in the evidence, with no production change to satisfy it.

## Alternatives and evidence boundaries

The [follow-up experiment note](../../rejected/2026-10-24-direct-dft-cold-followups.md)
retains the below-floor sampled-map chunk schedule and an older, superseded
combined candidate. Their observations do not contribute to this promotion.
The [original integration note](../../proposed/2026-10-23-occupied-xc-current-integration.md)
remains a historical opt-in decision on its own source.

Current-source comparisons use separate pristine-master and candidate libraries,
three rotating fresh-process rounds, and stock GPU4PySCF. Cold covers setup,
SCF, analytic forces and host export; one warm call uses each owner's own frozen
cold density. Timed calls do not run an oracle. Independent dense-grid references
and CPU/GPU spot checks qualify every static sample. The displaced-geometry
control uses a fresh, tighter matched-grid reference after native timing.

The predeclared gain/nonregression floor is the larger of 2% and twice the sum
of relative MADs. A smaller measured kernel or host scope cannot override the
complete endpoint gate. Default-grid sampled workloads do not establish a
universal fastest configuration, UKS molecular speedup, or whole-method ranking.

## Current-source result

The [full measured report](../../../../benchmarks/results/occupied-xc-default-20261024/README.md)
retains all 166 static and three original/warm/moved E/F calls, source/binary
bindings and the offline rechecker. Three-process Direct Standard/96 cold
medians are 42.376693 → 40.730322 seconds for r2SCAN-3c (3.89%, above its 2%
floor), versus GPU4PySCF 32.448150 seconds. PBE/PBE0 and warm changes remain
below the floor. This closes only part of the remaining 25.52% 3c gap.

A separate same-image profile records eight occupied generations and density
product time 4.415185 → 2.748385 GPU seconds across cold+warm. Potential and
integral-force work stay unchanged. Force map/work and numeric bounds match.
The 433 host checks, 23 GPU feature/owner/force/default-policy tests and all
three eight-case sanitizer selections pass. No scientific gate is relaxed.

A small-system three-pair cohort flags a 3.81% cold slowdown. Both subsequent
five-pair same-image isolation and different-image confirmation fail to reproduce
a policy regression; all cohorts and their predeclared floors remain separate.
The latter gives 1.492828 → 1.460484 seconds below its 4.04% gain floor. Small
cold speedup and universal nonregression are not claimed. Warm remains stable.

## Revisit when

Change admission or frame lifetime only after independent feature, alias,
failure/recovery and full cold/warm/displaced E/F validation. Retain the explicit
rollback. Further improvement should target the still-dominant integral-force
and AO-potential contractions; this promotion only changes density features.
