# Decision: admit bounded physical replay batching for CUDA DF forces

Status: implemented
Date: 2026-10-12

## Problem

#2221 admitted the existing expanded physical replay's optional Q batching only
for energy owners without retained response. The final-amplitude replay does
not consume retained adjoint intermediates, so excluding analytic-force owners
unnecessarily retained many one-Q launches.

## Decision

Enable the existing option for native CUDA DF owners irrespective of retained
response. Do not change iteration policies, Lambda/orbital-response equations,
precision, Q order, provider ownership, scratch ceilings or resource fallbacks.
Every expanded residual remains fresh. Internal supplied-problem defaults stay
false. The optional batch must fit already admitted device scratch; finite
checks still run after each original Q addition.

## Evidence and source boundary

Frozen clean baseline `34dd51ed3ebc5b03b17f095de2cd7b39903923c8`, plus the
retained one-file dispatch overlay. This experiment is independent of #2298;
incoming-residual mixing remains disabled for forces in both measured arms.
All generated scientific kernels and both endpoint/probe binaries are identical.

On node2's Slurm-assigned RTX PRO 6000, four cold complete ethane230 force
processes run in ABBA order. Complete median wall decreases from
275.611584 to 274.532346 seconds, **0.3916%**. Both candidate samples are faster
than both controls. This is a small scoped observation, not statistical/global
performance qualification or latest-master whole-library timing.

Replay median decreases 9.668271 -> 8.414243 seconds; complete CCSD decreases
61.066811 -> 59.878997 seconds. Actual admitted replay batch is **four**, not
the energy predecessor's sixteen: replay Q tiles decrease 488 -> 122.
CCSD evaluations remain 38, Lambda actions 22 and orbital actions 13.
CCSD GEMMs decrease 54,862 -> 45,468; semantic contraction summands remain
26,799,066,988,432. Physical GEMM summands increase slightly because the batched
lowering has its own layout/contraction inventory. No arithmetic-work win is
claimed. Packing and accumulation counters are logical traffic, not measured
DRAM bytes.

Complete logical numeric capacity remains 19,509,839,897 bytes; CCSD capacity
increases 4,392,162,937 -> 4,392,219,807 bytes for bounded host bindings. Physical
peak VRAM is not measured. The complete maximum hides this small CC-stage
increase, so both capacities must be retained.

The 38 independent complete-force/public tests, large same-Hamiltonian energy,
fresh residual, Lambda/Z, and independent directional-FD gates pass.
Maximum force difference is 7.64e-14 Eh/bohr; maximum independent large
direction error is 7.11e-9 Eh/bohr. Memcheck and actual per-graph work receipts
are retained alongside process RSS and exact source reconstruction. Baseline
independent gates reuse the already-qualified, byte-identical baseline rather
than rerunning unchanged scientific work.

## Alternatives and consequences

Do not grow the replay scratch ceiling to obtain the energy predecessor's Q16.
That would be a different resource/ownership change requiring separate evidence.
Do not combine this timing with #2211 or energy-only receipts. Retaining
independent PRs makes the small launch-reduction gain attributable and reviewable.

## Revisit when

Requalify broader workloads or altered scratch ownership with independent forces
and complete endpoints. Do not advertise the descriptive 0.39% as a universal
speedup or relax fallback/numerical gates to increase it.

## References

- #2221; independent incoming-residual force extension #2298.
- `benchmarks/results/df-cc-force-batched-replay-20261012/publication.json`.
- `tests/python/test_df_cc_batched_replay.py` and its force evidence regressions.
