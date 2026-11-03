# Proposal: bounded CPU MP2 force source residency

Status: proposed
Date: 2026-11-04

## Problem

The policy-retirement CBS campaign remains incomplete. Subsequent diagnostic
work distinguishes three costs that a generated derivative alone did not fix:
reference derivative evaluation, small-tile MO transformations, and repeated
regeneration of the same AO integrals during native MP2 response/force work.

`Mp2Prepared::execute_with_reference_seed` already owns a CPU exact prepared Fock
plan and exposes it through `PreparedFockInteractionSourceView` during energy
execution. Before forces, it deliberately retires that source/plan to preserve
the legacy CPU force reservation. The conventional CPU force then receives a
`RawSource`, unlike the conventional CUDA route's retained qualified source.
Simply changing the last source argument is therefore unsafe: the current CPU
lifetime and memory-admission policy must change together.

## Bounded diagnostic observations

These observations are not clean-product qualification or performance claims.
All use the unchanged physical H2 DZ/TZ Standard full E/F endpoint, a source-bound
CPU library and ignored LD_PRELOAD probes, without GPU execution:

1. A constructor-wide source-tile override was rejected by the existing
   compiler-schedule/work guard. Preserve that guard; do not alter energy
   schedules behind the compiler's back.
2. Restricting tile selection to the CPU force owner preserved the guard. A
   completed provider invocation used 1,248,940 transform FMAs and four stages
   instead of the earlier billions of FMAs, but still took approximately
   fourteen seconds. The full 180-second probe timed out. Smaller transform
   work alone did not remove repeated AO generation.
3. A further diagnostic cached exactly one original full AO read for the
   28-function force source, advertised its actual capacity through
   `retained_numeric_bytes`, and retained ordinary native force admission. With
   the existing packaged derivative probe and force-only cross-shell tile, the
   complete endpoint returned within its 180-second limit. The source held
   4,917,248 AO-value bytes and serviced 1,149 successful cached reads; these
   counters describe that source, not the entire endpoint's semantic work.

The completed diagnostic's five components and composite agree with the pinned
PySCF reference under the existing 1e-9 energy and 2e-6 force gates. Composite
absolute errors are 4.5062265030537674e-10 in energy and
1.1071001147083814e-9 in maximum force. This is one instrumented feasibility
observation, not ABBA/BAAB, a 1.05 result, independent fallback qualification, or
permission to replace the original failed campaign.

## Proposed direction

Prefer borrowing the existing CPU prepared integral owner over adding another
unconditionally allocated dense cache. Keep the owner alive only when checked
complete-force admission includes its actual retained capacity, provider staging,
response workspace and derivative work. If residency cannot be admitted or is
unavailable, retire it and preserve the bounded streaming route. Validate both
paths and their source/reference lifetimes; do not subtract shared storage twice
or omit it because it is immutable.

Use the existing generated source planner to select admitted CPU cross-shell
tiles for every force consumer. Do not change compiler-selected energy work or
weaken schedule assertions. Selecting the packaged derivative owner requires its
own supported-domain, independent numerical and fallback qualification. No part
of the diagnostic interposer is a production implementation.

## Evidence and next gates

The local campaign retains `provider_tile_probe_v1.cpp`,
`provider_force_tile_probe_v2.cpp`, `provider_tile_probe.cpp`, all three terminal
probe outcomes, compiler-cache observations and `feasibility-audit.json` under
`.artifacts/required-native-policy-20261104/`. The audit checks full observable
shapes, all five component/composite oracles, coefficient reconstruction, and
balanced completed trace scopes. Its report explicitly sets qualification and
production-path flags false and leaves whole-endpoint semantic work unknown.
These transient feasibility artifacts are not substituted for retained
qualification receipts.

Any implementation needs fresh source/binary provenance, exact/below-capacity
and allocation-failure/lifetime tests, independent full physical component and
composite E/F oracles, and the unchanged complete endpoint/work protocol. The
earlier negative packet and rejected-shortcut rationale remain valid history:
`.agents/notes/rejected/2026-11-04-cbs-generated-derivative-shortcut.md`.
