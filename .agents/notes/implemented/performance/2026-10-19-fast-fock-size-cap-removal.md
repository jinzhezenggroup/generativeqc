# Decision: remove measured-size ceilings from Fast Fock admission

Status: implemented
Date: 2026-10-19

## Problem

The user explicitly requested removal of the upper size limit after reviewing
the 192–768-AO default. Removing only the anchor policy's ceiling would be
ineffective: shared budget-Fock admission also rejected more than 768 AOs or
96 atoms. Those limits originated in the original qualification window, not
fixed-size recurrence workspaces or anchor storage.

This supersedes the upper size limits in the
[module-selected Fock decision](2026-10-15-module-selected-fast-fock.md) and
[anchor default decision](2026-10-18-fast-anchor-force-handoff-and-size-gate.md),
not their numerical gates, resource fallbacks or historical measurements.

## Decision and invariants

- Remove the AO and atom-count ceilings from shared application budget-Fock
  admission. Retain at least 24 AOs/two atoms, restricted all-electron execution,
  supported Direct precision capabilities, shells through d and all allowance
  floors. This affects shared Fast Fock consumers, not just WB97M-V.
- Default Fast WB97M-V anchors at 192 or more AOs, intersected with the remaining
  scientific/provider/precision/spin/nonlocal/chunk/resource-ledger conditions.
  Preserve explicit rollback and smaller eligible opt-in.
- Native integer/dimension checks and overflow-checked storage planning remain.
  The optional anchor costs `40 * nao**2 + 8` bytes, with no fixed-size buffer.
  Device OOM preserves ordinary Fast; pressured force handoffs retire idle
  anchors before optional ERIs. No memory reservation or arbitrary-size fit is
  promised, and host/non-allocation failures remain explicit.
- Keep fixed FP64 anchors, total current-minus-anchor density, bounded mixed
  proposals, fresh strict physical targets, canonical/stationarity gates and
  FP64 analytic forces. Do not change the separately qualified mixed-gradient
  shape limits, derivative arithmetic or explicit observable audits.

## Evidence scope and consequences

Host executable admission tests cover 769, 1024, 1536, 2048, 4096 and 8192 AOs,
atom counts beyond 96, the preserved lower floors and the scientific exclusions
at larger shapes. These are policy tests, not large-molecule numerical or
performance qualification. Independent native CPU-composition and retained
GPU4PySCF endpoint gates remain necessary regression coverage.

Matching-source verification passes under finite n1 Slurm job 10261: native
`--fast-anchor-delta` retains its independent small-system CPU composition and
GPU lifecycle/error controls, and the selected Python suite reports 399 passed.
All configured pre-commit hooks pass. The authenticated scientific source is
`337d4014e5b46776b769d0ac3268f917f10150f55cc72b706c6b3253f3927e2e`;
production library SHA256 is
`6c6be697b0c3f595ce07532b4ff47e47a1693679316605b92e39528b071c30dc`.
Verified sccache 0.16.0 rebuilds changed-header dependencies and provenance,
without changing the retained earlier builds. Raw build/test logs remain in
ignored `.artifacts/fast-anchor-unbounded-20261019/` and on n1 under
`/data/jzzeng/pr2602-unbounded-20261019/`.

Historical source-bound packets, including adverse 96-AO timings, qualified
192/768-AO endpoints and incomplete 384-AO oracle attempts, remain unchanged.
They do not become measurements of the new source or of sizes above 768 AOs.
The existing Fast qualification is not a per-call observable audit, and resource
safety alone does not establish a size-independent absolute energy/force error.
This user-directed admission expansion makes no new large-size accuracy or
speed claim; report any subsequently measured independent large-size gates
separately rather than relabeling the frozen evidence.

Revisit admission using independent complete energy/analytic-force endpoints,
their actual work and resource failures. Do not replace 768 with another
arbitrary upper benchmark dimension or remove numerical/lifecycle fallbacks.
