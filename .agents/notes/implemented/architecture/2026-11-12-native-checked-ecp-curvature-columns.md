# Decision: checked general residual-ECP second sources with whole-column reuse

Status: implemented (frozen source; stationary electronic endpoint remains separate)
Date: 2026-11-12

## Problem

The existing residual-ECP first direction covers the checked general operator,
but the frozen mixed source covers only its power-two local/s/p/d analytic
specialization. A full native stationary Hessian cannot silently replace other
legal powers/projectors with that specialization or hide numerical quadrature
failures behind zero density weights or a projected zero direction.

## Decision

Add `ecp_curvature_plan` and `checked_ecp_curvature` with explicit structural
Empty/Gaussian/CheckedGrid dispatch and mandatory resource limits. Reuse the
analytic order-two pair/frontier evaluator with column retention rather than
forming a coordinate Hessian. For the general backend, differentiate the
existing Gaussian AO Graph to symmetric second jets, project them once per
radial layer and reuse the projections for every AO pair. The canonical
bilinear product Graph supplies the same 43-field relative integral-jet
contract as the analytic provider.

One shared compiler column map and compensated native output owner consume
each relative A/C,B/C jet once into value/right/gradient/HVP fields. Arbitrary
ordered weights have no occupancy factor. A common ECP-site origin preserves
exact right-translation zeros. There is no production displacement, reference
fallback, runtime graph/JIT, AO-fourth-power tensor or left-coordinate source
replay.

## Raw physical acceptance

Preserve incumbent primal 2e-9 and gradient 2e-8 coarse/fine gates. Require
second derivatives to meet 2e-8 per raw physical component on the same 160/32
and 224/44 grids. This is an explicit empirical discretization gate, not an
adaptive convergence guarantee. Check raw components before arbitrary weights
or right-leg contraction, including zero and rigid requests.

For each AO pair, only its physical A/B coordinates receive contributions
from every ECP site. Reduce their value, gradients and symmetric Hessian
blocks globally. Any other physical coordinate belongs to exactly one ECP
site, so its C gradient, A/C, B/C and C/C components are checked at that site's
completed reduction and discarded. Mixed derivatives between two other sites
are identically zero. Compiler incidence contractions combine coincident
physical A/B/C identities before checks; do not gate independent slot fields
as if they were different physical atoms.

Keep 28-field symmetric relative and physical-slot panels for two grids and
two residual parts: 224 matrix-sized FP64 fields, independent of atom count.
Ten-field AO/projection scratch and O(atoms) compensated output complete the
numeric staging. Conservative source capacity also covers metadata and grids;
borrowed state, scalar stack, allocator overhead and RSS/time are distinct.
The query starts from the existing complete-grid work inventory and replaces
its first-order retention envelope explicitly. Contraction counts are semantic
point/projector pairs, each carrying 43 second-order fields, not FLOP totals.

## Rejected alternatives

- Projecting to weighted scalar mixed energies before gates can turn a bad
  quadrature into an apparent success at zero density or rigid motion.
- Materializing coordinate-squared AO derivatives would replace a bounded
  physical-incidence problem with unnecessary atom-count storage growth.
- Replaying the entire quadrature for every left coordinate repeats invariant
  AO values/projections and creates the same work cliff as scalar ERI replay.
- Reimplementing Gaussian derivatives or spin/occupation factors natively would
  create another scientific authority rather than complete the existing one.

## Independent validation and finite-difference noise

Complete column controls use normalized independent libcint primal residual
operators, both representations and s/p/d/f shells, with both analytic and
general powers/projectors. A separate executable test constructs a genuinely
first-order-qualified but second-order-unqualified quadrature and requires
refusal even with zero weights and motion; no mocked predicate suffices.
Resource, transactional owner failure/recovery and 3-to-32 spectator-coordinate
source-work controls are separate from scientific oracle controls.

The initial Cartesian f general-grid control exposed a validation-oracle
roundoff cliff, not an accepted weakened gate. Its mixed error decreases from
2.7123e-6 at h=3e-2 to 3.0122e-7/2.6971e-8/2.9372e-9 at
h=1e-2/3e-3/1e-3, then grows to 2.9071e-8/4.8734e-7 at h=3e-4/1e-4.
The retained six-step diagnostic contains every primal corner. Use
1e-2/3e-3/1e-3 for final column controls, retaining atol3e-8/rtol2e-6 and the
original eightfold-convergence/5e-9-floor gate. Do not force smaller steps into
the oracle noise floor or relax the acceptance threshold to make them pass.

## Consequences and remaining scope

The general source retains more finite-slot AO gate storage than a first
direction; resource refusal is explicit. Synthetic residuals and spectator
atoms are not actual Ni/I/Cs electronic endpoints or 25--40-atom performance
qualification. Stationary composition with actual D/W response, complete
moving-grid XC scheduling, installed native CPU TS and final unscaled harmonic
frequencies remain separate work. This decision extends, not retroactively
rewrites, the accepted analytic mixed-source evidence.

## Observed qualification

The 2026-11-12 host-only cohort is retained under
`build/qualification/host-ecp-column-20261112/`. The final related regression
passes 145 tests without skips; the component cohort passes 52. Six selected
CPU CTests, strict Sphinx, compiler structure (610 modules, no dependency
errors), type checks, changed-file hooks and the root rebuild/no-op pass.
The native direction/column executable also passes with Python absent from
PATH and PYTHONPATH. No GPU work or installed electronic endpoint ran here.

All 16 independent normalized libcint primal controls pass the unchanged
primal and derivative gates. Maximum final gradient and mixed errors are
1.7759198009739685e-9 and 2.937226388976741e-9. The initial failed convergence
test and six-step roundoff diagnostic remain alongside the accepted run.
The real second-only refusal uses exponent scale 10 and coefficient scale
1.3769090238204313; its raw first-order gaps are approximately
[7.1993e-10, 2.1927e-15, 1.5e-8, 5.8935e-14], so the existing first-order
gates pass while the raw order-two gate refuses even zero weights/motion.

For both the 3-atom and 32-atom spectator controls, actual source work is
identical: the analytic provider visits 1,152 primitive pairs; checked-grid
visits 768 layers, 8,538,112 AO point terms and 74,818,208 contractions.
Observed source times are about 0.0008/0.0007 s and 0.4673/0.4683 s,
respectively. These synthetic source measurements are not complete electronic
endpoint performance. Plans and raw telemetry are retained in the two
`ecp-column-source-reuse.npz` files, discovered recursively rather than by
assuming pytest's truncated directory names.
