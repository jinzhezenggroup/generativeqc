# Decision: bounded Lambda FP32 warmup in CBS Fast

Status: implemented
Date: 2026-10-21

## Problem

CCSD preconvergence covers a small fraction of the complete CBS energy-and-force
endpoint. The [preceding screen](../../proposed/2026-10-21-cbs-fast-lambda-followup.md)
identified Lambda as a larger iterative consumer, but also found that enabling
its existing mixed option naively doubled later auxiliary tile work.

## Decision

Grouped CUDA DF-CBS Fast force endpoints request the existing Lambda auxiliary
FP32 warm start, alongside CCSD warmup. Standard, energy-only, standalone and
supplied-problem defaults remain unchanged. The Lambda owner alone admits the
actual storage/provider path and records request, refusal and executed work.

Keep the existing absolute handoff tolerance of `1e-6` and 16 warm iterations.
Only a finite proposal enters a new FP64 GMRES solve with empty Krylov history.
The selected physical residual, independent expanded transpose audit,
parameter/factor VJPs, triples and final force response remain FP64. Nonfinite
failure propagates; it is not hidden by a strict retry.

The experimental owner halves its requested auxiliary Q cap to share scratch
with FP32 inputs. CBS Fast requests 64, preserving the ordinary Q32 ceiling
rather than forcing Q16 for the strict solve/audit. This is a ceiling, not a
promise: the bounded planner can select a smaller batch or refuse mixed work.
No new owned device allocation is added to the Lambda algorithm.

A private versioned native getter copies request/execution/refusal flags,
mixed/strict iterations and shared-operator actions, FP32 GEMMs/casts, actual
Q batch, numeric/device capacities and phase times before owner destruction.
The frozen public C performance structure is unchanged. Missing telemetry is
unknown, not measured zero; an admitted CBS Fast force request requires the
new admission telemetry. Overall mixed arithmetic means CCSD or Lambda actually
executed FP32 GEMMs. Energy-only calls report Lambda as not requested.

## Rejected alternatives

- Simply enable the existing option with cap 32: the observed Q16 strict work
  consumes part of the warmup saving. Preserve Q32 subject to real admission.
- Tighten the warmup handoff to the final Lambda tolerance: the retained screen
  exhausted the warmup bound and added work while saving only one FP64 action.
- Loosen the final scientific allowance or omit replay/force consumers to make
  Fast look faster: neither follows the user's requested complete method.

## Invariants

The existing CBS coefficient-L1 error allocation and tighter-caller precedence
remain authoritative. FP32 proposals never certify the solution. All published
precision claims require measured native work; requests alone are insufficient.
Action counters exclude the separate audit, nested phase times must not be
summed twice, and numeric capacity is not measured peak VRAM.

## Evidence and limits

The [qualification bundle](../../../../benchmarks/results/cbs-fast-lambda-20261021/README.md)
retains complete endpoint receipts, exact source/library identities, repeated
matched C4 comparisons with prior Fast, C1–C7 Standard/Fast pairs, independent
C2 directional/energy checks and full-vector water finite differences for both
reference models at original/displaced geometries. It includes cold/process-warm
replays, a successful 256 MiB cohort and the failed 64 MiB complete-endpoint
admission. Native tests separately cover optional-storage refusal, finite warmup
exhaustion and sticky nonfinite failure. No batch is supported by this CBS API.

The larger alkane controls are against Standard, not independent full-vector
oracles. The homepage has one fresh observation per policy/size, not a universal
speedup guarantee. RTX 5090 evidence does not establish other-device performance.

## Revisit when

Further endpoint gains require measuring triples, independent replay/audit and
TZ MP2/reference response separately. Changes to those consumers need their own
scientific qualification. Revisit the Q compensation if the Lambda owner's
scratch policy changes; preserve explicit bounded fallback and work accounting.

The [GMRES workspace follow-up](2026-10-22-cbs-lambda-workspace.md) addresses
C7's Q4 admission cliff by removing unused host reservations while preserving
this warmup policy and all final convergence gates.
