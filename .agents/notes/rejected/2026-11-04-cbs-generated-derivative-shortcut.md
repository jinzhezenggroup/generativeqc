# Decision: do not promote a generated-derivative probe as CBS qualification

Status: rejected
Date: 2026-11-04

## Problem

The required-native-policy retirement in PR #2972 leaves native science
unchanged, but its complete physical H2 DZ/TZ CPU CBS endpoint campaign cannot
finish the first CBS block within either the original 600-second or revised
3600-second finite ceiling. Four Precision case/order comparisons pass, but no
durable CBS publication is available. A slow baseline is not a passing
candidate non-regression result.

An unmodified baseline GDB sample reached `mp2::reference_force` through
`conventional_derivative_cpu`, the single-channel weighted ERI derivative
wrapper and primitive reference recurrences. The existing multi-channel owner
already supports `WeightedEriDerivativeExecution::PackagedCpu`. It was plausible
that simply routing this call through that owner would unblock qualification.

## Decision

Reject **promoting the diagnostic interposer or treating kernel availability as
endpoint success**. This does not reject the existing generated derivative
implementation or prove that it is unsuitable for a properly qualified change.

The ignored diagnostic interposer passes the original weights unchanged to the
existing packaged owner; it introduces no second derivative equation. Even so,
its full 600-second endpoint probe timed out without a completed high-basis MP2
group. A separate bounded GDB/phase trace sampled the streamed orbital-response
RHS in `NativeBlockProvider::get_many_impl`, called by
`rotation_gradient_streamed`. One completed provider invocation reported
38,416 reads, 614,656 values and 2,065,244,160 transform FMAs. These counters
belong to that invocation only; neither timeout supplied whole-endpoint work.

The source traverses four provider orientations per orbital in the streamed
rotation gradient. This explains why derivative acceleration alone need not
bound endpoint work, but a sampled stack is not proof of the entire bottleneck
or a substitute for complete timings. Current workspace admission protects
memory; it does not certify inexpensive integral transformations.

## Invariants

- Preserve the full physical workload, independent scientific gates, first-call
  record, 3 warmups, 21 timed samples, ABBA/BAAB and 1.05 limit.
- Keep failed campaigns and unknown counts explicit; never infer publications
  or numerical results from elapsed time, a sampled stack or a small group.
- Do not change native science or measured source under the existing policy
  retirement evidence. Any native optimization needs its own source/binary
  provenance, independent numerical oracle and complete endpoint/work gates.
- Preserve bounded resource/provider fallbacks and admission when adding reuse;
  do not install unconditional dense storage just to make a small case fast.
- No CPU diagnostic, host metadata, package build or skipped CUDA case is
  real-GPU execution evidence.

## Evidence

`benchmarks/results/required-native-policy-20261104/` retains exact measurement
drivers, raw completed records/journals, source/binary provenance and an offline
receipt replay that exits 2 for the incomplete campaign. Protocol v2 completed
400 Precision publications with maximum ratio `1.002917157468209`; all eight
CBS process blocks remain missing. The first attempted CBS block genuinely
terminated at 3600 seconds, not merely at a tool observation timeout.

The 180-second baseline stack, 600-second packaged-owner probe, and separate
180-second packaged-owner GDB/trace probe are diagnostic-only. Their instrumented
small-group times are deliberately excluded from qualification results. Full
transient logs, loader setup failure and compiler-cache observations remain in
the ignored campaign artifacts. No production source was changed by these probes.

## Revisit when

A separately reviewed native change bounds the actual repeated provider/response
work and derivative work together, retains correct lifetime/capacity refusal,
passes independent full component/composite E/F oracles, and completes the
unchanged endpoint protocol. First establish feasibility with a bounded full
endpoint diagnostic; do not blindly increase every block timeout or substitute
an easier molecule/basis to turn the campaign green.

## References

- #2227 R4/R6 and PR #2972; neither is completed by this negative packet.
- `src/posthf/mp2_gradient.cpp`: streamed rotation-gradient provider traversal.
- `src/posthf/mp2_force_workspace.hpp`: explicit live-block workspace admission.
- `src/posthf/mp2_derivative_cpu.cpp`: current reference derivative consumer.
- `src/integrals/s_integrals.hpp`: existing generated derivative execution owner.

## Later diagnostic, not a production change

Additional same-day probes isolated repeated AO generation after a force-only
tile change. A diagnostic resident source plus the earlier derivative probe
completed one unchanged physical endpoint and passed its component/composite
oracle checks. This does not qualify or promote the derivative-only shortcut or
repair the failed campaign. The separately proposed bounded source/lifetime
direction is recorded in
`.agents/notes/proposed/2026-11-04-cpu-mp2-force-source-residency.md`.
