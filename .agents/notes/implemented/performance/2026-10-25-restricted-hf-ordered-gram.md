# Decision: reuse ordered DIIS Gram rows for large restricted HF

Status: implemented
Date: 2026-10-25

## Problem and decision

Direct HF96 cold spends avoidable time repeating old residual dots. The existing
ordered physical-slot Gram cache already preserves each dot's scalar FP64 order,
and its staged pending-row/full-CTA update is qualified for large KS owners.
Restricted HF now automatically admits it at 512 AOs and history capacity two.
Resource estimation, cached-plan invalidation and execution share the same guard.
Smaller and unrestricted owners retain serial defaults; explicit disable and
cooperative/ordered diagnostic selection keep their existing meaning.

This adds no scientific kernel, cache allocation, approximation, convergence
tolerance, preliminary solve or oracle dependency. Only the newest chronological
row is evaluated; live old-old entries keep their physical-slot identities.
The first insertion evaluates no dot; the second establishes the earlier diagonal.
Count/head resets and route changes must continue invalidating cached history.
The public resource/checkpoint identity also records both DIIS controls.

## Evidence and boundaries

The [source-bound capsule](../../../../benchmarks/results/hf-ordered-gram-20261025/README.md)
retains three alternating complete endpoint rounds. Standard HF96 cold medians
change 19.366 to 18.471 s (4.62%); Fast changes 19.413 to 18.493 s (4.74%).
Both keep eleven Focks. HF48, below admission, remains effectively unchanged;
its Standard/Fast controls are retained instead of selectively reporting wins.
HF96 warm remains approximately 2.93 s. This does not close the GPU4 cold gap.

Independent original/moved/property/rollback checks cover 66, 96 and 120 atoms
in both profiles. Native large-owner tests cover 512/513 AOs, one/two spins,
histories 3/8, normalized/raw metrics, resets and poisoned inactive storage,
with independent long-double dots and bitwise output checks. Host selector
tests exercise automatic, explicit and inactive-invalid cases.

## Rejected alternatives

Partial canonical ERI caches do not address this endpoint profitably. On the
integrated control, HF96 cold is 19.318 s; angular bucket masks 144, 1456 and
32512 take 97.316, 64.554 and 83.060 s. Their approximately 2.95 s warm calls
do not repay cold construction. All complete receipts and executed controls
remain in the capsule. Earlier full 16-GiB residency, split J/K and incremental
HF negatives remain in the preceding response-capacity and Direct-Huckel notes.

## Revisit when

The remaining cold gap is dominated by repeated full-Fock recurrence. Investigate
source/consumer work and reuse that pays back within an ordinary cold solve,
with complete setup included. Requalify unrestricted owners or a lower AO floor
before widening this guard. Do not infer a universal crossover from water96.
