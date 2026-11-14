# Decision: one-source moving-grid curvature columns

Status: implemented
Date: 2026-11-12

## Problem

The accepted mixed AO/Becke reader is bounded for one pair of directions, but
replaying it for every left coordinate repeats AO evaluation and the whole
partition source. Integral and ECP column sources alone cannot remove that
remaining frozen-XC scheduling cliff.

## Decision

Add `HostGridColumnGeometry` beside the scalar-pair reader. Evaluate ordinary
order-three normalized spatial AO jets once per tile. Record the Becke source
once per point and reverse its right JVP once to all physical coordinate leaves.
The grid-point leaf scatters back to its physical owner. Fixed-arity unit left
seeds of incumbent compiler scalar jets supply the local Jacobian and its right
JVP; native code schedules these coefficients, not new derivative formulas.
Nuclear pair distances are shared across the point tile. The output weight
column is value/right/full gradient/HVP, with signed atomic measures unchanged.

Native forward product normalization and reverse adjoint normalization have
separate exact binary exponents. An exactly zero product may have a nonzero
gradient or HVP, even when the chosen right leg is zero. Never reset a forward
product's accumulated exponent just because its value/right fields are zero.
Never materialize a possibly enormous normalization coefficient on the reverse
pass: carry its exponent until adjoints combine or reach the final coordinate.
Normalize local reverse coefficients before multiplying their mantissas so a
subnormal coefficient does not disappear before the carried scale can recover it.

The existing dimensionless retained-partition gate, norm coincidence refusals,
version 1--4 prescriptions, radius adjustments and iteration counts are unchanged.
All outputs stage until final immutable-owner and right-content authentication.
Explicit mandatory caps include per-tile AO and source-pair visits and forward
plus reverse tape-node visits. The latter are semantic schedule counts, not
scalar FLOPs; local jets use at most three fixed seeds, not 3N molecular seeds.
Conservative owned capacity includes tape metadata and both readers' scratch;
borrowed arrays, allocator overhead and RSS/time remain separate.

## Rejected alternatives

- Replaying the full mixed geometry source once per left coordinate hides
  invariant AO/source work behind bounded memory.
- A coordinate-squared AO or weight tensor stores unnecessary molecular indices.
- Handwritten norm/Becke Hessians create another scientific derivative authority.
- Log-product differentiation or division by individual switching factors loses
  exact-zero products and their nonzero derivative fields.
- Dropping zero-primal products before the reverse pass changes the derivative
  contract, even when ordinary primal partitions appear unaffected.

## Validation and remaining composition

Tests use independent normalized libcint order-three spatial AO jets and
70-digit displaced primal direct-product partitions for every physical gradient
and HVP component. They also test raw matrix symmetry without repair, right-leg
linearity, exact rigid right zeros, partial tiles, iteration policies, signed
measures, cap refusals, stale/final owners, reentry, borrowed/null/output aliases,
actual right mutation and recovery. Synthetic 3/25/40-center controls retain
one AO visit per point/AO and one pair traversal per center/point source, not
per left coordinate. The standalone native executable has no Python/compiler
at execution time. These controls are not complete electronic endpoints.

The frozen-XC energy consumer still uses its scalar-pair schedule. Its next
column schedule should evaluate primal/right density features once, obtain
weighted field adjoints and their right directions from the exact linked
mixed-energy closure, then use canonical TensorIR AO-feature pullbacks and JVPs
to scatter through the retained spatial jets. Fixed ingredient-basis work must
be explicit; do not substitute per-molecular-coordinate source replay.
Stationary actual D/W response, other energy sources, installed native CPU TS
and the requested electronic/frequency qualification remain separate work.

## Observed qualification

The host-only cohort is retained in
`build/qualification/host-grid-column-20261112/`. The initial column suite
passes 18 tests; the source-matched final related regression passes 144 with
no skips (101.92 s). All nine optional-target CMake tests pass (217.49 s),
including a new actual static-target-linked grid consumer running with Python
absent from PATH/PYTHONPATH. Six selected CPU CTests, compiler structure
(610 modules, zero dependency errors), strict Sphinx, type checks, changed-file
hooks and the root rebuild/no-op also pass. The earlier 144-test run before
test-probe symbol disambiguation is retained separately, not substituted for
the source-matched final run.

Eight independent all-coordinate controls cover versions 1--4 and both AO
representations through f. At h=3e-3/1e-3/3e-4, maximum final gradient and HVP
errors are 3.989491776101861e-7 and 1.7107052852072258e-7. Acceptance uses the
unchanged combined atol3e-7/rtol2e-6 and twelvefold-convergence/2e-9-floor
gates; the maximum gradient error does not meet the absolute term alone.
The independent partition arithmetic uses 70 digits; spatial AO jets are
direct independent libcint derivatives rather than differentiated native AOs.

Synthetic three-point controls at 3/25/40 centers visit 9/75/120 AO lanes and
12/1200/3120 source pairs, without a factor of 3N. Forward/reverse node visits
are 189/231, 10815/13365 and 27495/33975. Conservative capacities are
19600/909896/2287256 bytes; observed source times are approximately
0.000060/0.00198/0.00496 s. These are geometry-only schedule controls, NOT
25--40-atom electronic endpoint timing or memory qualification. No GPU work
or installed TS/electronic endpoint runs in this cohort.

The initial smoke invocation omitted PYTHONPATH and failed collection before
any native work. Type-check diagnostics first exposed the test-module search
path and pytest's statically opaque fixture wrapper; the accepted invocation
sets the search path and explicitly casts the wrapper to Any. Ruff rewrites a
constant getattr back to attribute access, so getattr alone does not resolve
the type-check contract after formatting. These tooling diagnostics remain
retained; none changes a numerical acceptance gate.
