# Decision: prove iteration invariants in common TensorIR

Status: implemented
Date: 2026-10-06

## Problem

HF/KS already share fixed-point control and prepared integral providers. Replacing
those with a second SCF engine would duplicate working ownership without proving
that repeated numerical graph work disappeared. Conventional RCCSD still
executes reference-only nodes during every current/trial amplitude evaluation.
The user requested reusable infrastructure before CPU/GPU lowering, with CPU
correctness qualification first.

## Decision

Add a small dependency/effect proof over the existing immutable TensorIR. Its
consumer declares invariant inputs; transitive dependence on any dynamic input
or opaque/effectful operation excludes reuse. The analysis retains scientific
arithmetic and reduction order, owns no arrays, and contains no method names.

The first native consumer is the explicit internal, default-off conventional
dense CPU RCCSD `iteration_invariant_reuse` candidate. One synchronous
`solve_cpu(const Problem&, ...)` supplies the immutable reference lifetime; its
private current/trial amplitudes remain dynamic. A new solve always creates a
new preparation epoch. This is deliberately narrower than a cross-geometry cache.
The generator consumes the identical proof for CPU and CUDA prepare/replay
entry points. CUDA owner activation remains unqualified and is not enabled.

Retained nodes receive exclusive arena slots before dynamic coloring. Simply
extending their last-use intervals is wrong: early dynamic operations can reuse
a slot before its invariant's old position on a later replay. Complete solver
admission includes the retained layout. The existing full evaluator stays the
bounded fallback; the separate expanded physical-residual replay stays uncached.

## Rejected alternatives

- A new universal SCF class: existing self-consistent control, DIIS and prepared
  source owners already cover this responsibility; CC amplitude equations must
  not masquerade as HF/KS density iteration.
- A memoization cache keyed by dimensions or pointers: neither proves unchanged
  geometry, basis, parameters, precision or numerical reference contents.
- A Python-only reuse wrapper: it would not remove repeated work in the native
  consumer. The generic proof is connected to generated native evaluators.
- Reuse every input in DF CC: externally supplied corrections can depend on the
  current amplitudes. Existing specialized DF ownership is preserved.
- Promote CUDA execution from source parity: asynchronous ownership, failure and
  device execution require separate qualification.

## Invariants

Every retained value is derived entirely from declared immutable inputs and
proven-pure ancestors. Density/amplitude changes recompute all their descendants.
Failed preparation cannot publish ready state. Independent scientific validation
and budgets retain their previous gates. No approximation, tolerance, method or
backend changes are allowed as a fallback.

## Evidence

The common CPU tests exercise transitive dependencies, density updates, fresh
preparation after geometry/basis/parameter changes, effectful ancestry, strict
input validation, actual restricted/unrestricted HF/KS composition, actual CC
amplitude dependencies and equal CPU/CUDA schedule identities. These are
correctness/schedule tests, not a complete-endpoint performance benchmark.

At representative `o=2, v=3`, the common conventional graph contains 9 invariant
operations (8 einsum permutations/copies and 1 addition), 169 dynamic operations, and 2,080
logical invariant bytes. Runtime symbolic dimensions determine actual storage;
logical retained bytes alone are not complete solver memory admission. Native
owner tests and their independently expanded/determinant oracle are the numerical
acceptance, with the exact command/result reported in the associated PR.

## Controlled CPU result and limits

A preliminary comparison had noisy mixed signs. The final retained comparison
uses one binary, the same ample budget, the actual default-off baseline versus
explicit candidate, DIIS history 6, and nine alternating pairs. At `o=8,v=16`,
median full-solve time is 440.682 ms baseline and 429.415 ms candidate; the median
paired change is -2.24%. The initial slower sign did not reproduce. Final states
and operation/iteration counts match exactly; these checks do not record every
intermediate solver iterate.

A separate nine-evaluation fixed-work control at the same shape measures median
294.751 ms for the original traversal, 294.309 ms for the pinned layout with
preparation repeated every time, and 280.205 ms for actual reuse. It checks every
evaluation output and rotates all three positions. This does not establish a
pinned-layout penalty or hardware cache-miss cause. Full-to-reprepare compares
combined layout, traversal, code-generation and function-splitting effects;
reprepare-to-reuse holds pinned addresses fixed and removes repeated preparation.
The control is diagnostic and excludes DIIS and final physical replay.

## Consequences and revisit conditions

Retained intermediates trade memory for reduced repeated graph work. Budget
fallback preserves the original schedule. The candidate remains default-off:
the bounded synthetic native-solve evidence does not qualify complete molecular
SCF/MO/CC endpoints, gradients, broad shape/device regimes or GPU activation.
A generic profitability policy should compare work/traffic saved over expected
replays against retained lifetime/peak cost and measured evidence, rather than
hard-code a shape cutoff from this small probe. Further HF/KS/DF/SCC consumers
must supply their own immutable epochs and preserve physical validation.

## References

- Shared infrastructure/lifecycle: #926 and #933
- Provider-neutral backend boundary: #1886
- Structured solver execution: #835
- `docs/developer/iteration_reuse.md`
