# Decision: default to producer-proven restricted PBE0 force points

Status: implemented in the worktree; master-aligned endpoint qualification pending
Date: 2026-10-10

## Decision

Default the private stationary point request to on. The fast kernel is still
selected only by exact mathematical composition, an owned same-generation
identical-spin rho/gradient witness, precomputed phased storage, sufficient atom
scratch, and absence of external seeds. Explicit `off` preserves the general
route for ablation and diagnosis. Older producers/consumers keep their v1
fallback; the public task-view ABI does not change.

This promotes the algorithmic result rather than shipping a permanently disabled
experiment. General/response/HVP algebra, grid, precision, SCF stopping rules,
direct/DF choice and complete force assembly stay unchanged. First derivatives
use four common channels and one exchange evaluation. Spin-antisymmetric second
derivatives cannot be recovered from this restriction and remain outside scope.

Remove the prototype's unused guarded generic `evaluate_restricted` entry.
Only the composition-qualified PBE0 bound entry remains. Its numerical body and
the eight-direction general/response body are unchanged from the qualified
header; the compile-time channel invariant remains intact. The PBE0-only
subnormal exchange ratio avoids division by quantized rho^(4/3), without a
cutoff, changed acceptance gate or independent production formula.

## Integration and evidence boundaries

Merge master `4444d0376` into the owned branch, preserving fitted reserve/tile
policy and stationary residency assets. The PR diff contains this optimization,
tests, documentation and evidence, not unrelated reversions from the old branch.

Prior snapshot evidence remains in
`../../proposed/2026-10-10-rks-point-producer-binding.md`: routing/parity,
memcheck/initcheck, all 87 independent points, and official complete warm/
moved-warm E+F comparisons at 48/96 atoms. Those measurements are the #2185
snapshot, not current master. Independent arms' frozen seeds differ; actual
SCF/AO/force work matches. Bootstrap timings are not clean cold/reconvergence
performance evidence.

Master-aligned promotion passes 583 focused CPU tests, with three real-GPU
opt-in tests skipped outside Slurm. This includes all emitted point references
under the unchanged strict gate, binding, launch-failure, ABI and fallbacks.
No failing general PBE/PBE0 oracle row is removed. General PBE/UKS/HVP numerical
qualification is not implied.

Build authenticated native/grid/PBE0 AOT artifacts with the verified compiler
cache, then require actual default-route restricted counts and small matched
off/on 48/96 complete endpoint measurements before publication. Retain 1e-8 Eh
and 1e-7 Eh/Bohr independent gates, host return and actual work counts. Unrelated
master commits do not justify repeating older GPU matrices.

## Revisit when

The mathematical domain, producer ownership/generation, planner or consumer
changes; a producer loses the proof; or complete endpoints regress instead of
using the bounded general fallback. Register counts alone cannot justify this
promotion.
