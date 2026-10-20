# Proposal: prioritize Lambda work for further CBS Fast acceleration

Status: proposed; screened, not promoted
Date: 2026-10-21

## Problem

The user reports that Fast is not much faster. The initial seven-point homepage
campaign observes C7 1788.44 → 1484.93 seconds (1.204x); C1 is nearly unchanged.
CCSD FP32 warmup executes, but covers only early auxiliary iterations. C7's Fast
CCSD solver is 144.62 seconds, including 66.18 seconds of independent replay;
triples still take 342.80 seconds. Those receipts do not isolate Lambda, so
the remaining time cannot simply be assigned to it.

## Measured diagnosis

A separate C4 profile on the frozen native source resolves the missing phases:
Lambda is 31.70 seconds, including 20.25 seconds of GMRES, versus 8.24 seconds
for CCSD, in an 86.90-second complete endpoint. The TZ MP2/HF group takes 28.66
seconds. CCSD-only tuning cannot remove those costs.

The existing experimental Lambda auxiliary FP32 warmup is a promising next
consumer, but its default Q-cap halving matters. Turning it on alone changes
the selected Q32 tile to Q16: the FP64 independent audit grows from 4.83 to
6.33 seconds. Requesting twice the cap preserves Q32 on this shape and recovers
that loss, while retaining the existing bounded admission and fallback.

A tighter warmup handoff is counterproductive in this screen. It reaches the
16-iteration bound, adds residual replays, and removes only one FP64 action.
Do not repeat that experiment assuming more FP32 iterations must be faster.

Clean complete C4 observations are Standard 99.3862 s, current Fast 86.9800 s,
Q32 Lambda candidate 80.3389 s. C2 observes 11.3434 → 11.0425 s. Thus even this
next candidate offers a modest total improvement on these inputs; broad claims
of a several-fold Fast speedup are unsupported. Subsequent work must address
triples, independent replay/audits and higher-basis response as well as the
iterative sections. Those are separate numerical/algorithmic changes.

## Decision and boundaries

Retain the candidate patches and complete evidence, restore production defaults,
and keep the homepage chart tied to its original measured implementation.
This proposal is not authorization to skip independent equations, silently
lower final arithmetic, or replace analytic forces with finite differences.

The candidate passes C2's independent two-step directional and energy gates,
and C4's all-constituent/composite Standard controls. C4 Standard is not an
independent oracle. These single clean observations and one instrumented shape
do not qualify a production default across larger shapes, memory limits,
changed geometries or devices. The existing public telemetry only reports CCSD
FP32; promoting Lambda must also expose its actual arithmetic and refusal.

## Revisit when

Qualify the Q32 candidate on larger and constrained-memory shapes, add complete
Lambda precision telemetry, retain independent energy/full-force controls, and
measure repeated complete endpoints before changing Fast's default. Profile
triples and TZ response separately before selecting additional mixed consumers.

## References

- [Nine endpoints, candidate patches, profiles and verifier](../../../benchmarks/results/cbs-fast-diagnosis-20261021/README.md)
- [Original Fast decision](../implemented/numerics/2026-10-20-ccsdt-cbs-fast-convergence.md)
- [Existing Lambda mixed warmup](../implemented/performance/2026-10-11-df-lambda-mixed-warmup.md)
- [PR #2648](https://github.com/jinzhezenggroup/generativeqc/pull/2648)
