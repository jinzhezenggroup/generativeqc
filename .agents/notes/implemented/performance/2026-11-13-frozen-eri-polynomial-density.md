# Decision: contract frozen ERI curvature in a fixed polynomial density space

Status: implemented; complete molecular endpoint qualification remains separate
Date: 2026-11-13

## Problem

The physical-center work planner removed a worst-case overestimate but did not
remove primitive evaluations. For the unchanged 81-public-AO ICH3I stationary
fixture, ordinary frozen curvature still requires 1084965064 primitive visits.
Adding the unchanged 237738954-visit first source exceeds the finite one-billion
cap. Another unchanged endpoint run or a raised cap cannot address that problem.
See [the stationary orbit decision](2026-11-13-native-stationary-eri-orbits.md)
for the separately qualified stationary state/gradient and its limits.

## Decision

Add an optional internal `ProjectedMatrix` frozen-curvature schedule. Adjacent
public records with equal physical owner, normalized radial range and angular
degree share a complete Cartesian monomial block. The lab-frame polynomial map
is constant with respect to nuclear coordinates. Pull each spin density back as
`D_poly = T^T D_public T`, then feed it to the existing generated quartet weights
and existing primitive value/mixed-jet authorities. Do not reconstruct exchange
algebra in the native consumer. Public packed coefficients include Cartesian
component normalization, so the internal monomials must use unit coefficients.

Only frozen ERI value/gradient/raw curvature changes representation. S/T/V, W,
SCF, first Fock sources, moving-D/W response, ECP, XC and dispersion retain their
incumbent public-space ownership. No Cartesian SCF or auxiliary reference engine
becomes part of production.

Congruence uses compensated FP64 accumulation. An owned snapshot authenticates
the original density and complete packed basis throughout the cache lifetime.
Both projected density panels and both snapshots are mandatory numeric storage.
Allocation/authentication happens at admission; congruence and primitive work
wait for complete endpoint work checks. Density magnitudes above `1e50`, packed
polynomial scales above 16 or unsupported homogeneous layouts decline this
optional representation. Ordinary matrix and streamed children remain bounded
fallbacks. Only explicit preconditions/capacity and allocation failure may select
a fallback; scientific/numerical failures must propagate.

## Invariants

- Keep public AO, physical-center identity, model, grid, basis and ECP unchanged.
- Do not screen weights or redefine what one primitive evaluation means.
- Preserve raw symmetry/translation gates without repairing a raw matrix.
- Price generated quartet maps in the new, possibly larger polynomial extent.
- Count congruence contributions separately from primitive/map evaluations.
- Charge the final actually selected schedule before numerical preparation;
  optimistic projected admission never covers a more expensive fallback.
- Do not admit public ECP saddles from component tests or metadata budgets.

## Evidence and limits

The metadata-only native census in
`build/qualification/host-projected-curvature-20261113/plan.json` constructs the
unchanged 81-AO, 73-radial-primitive, six-atom system. Its independent shell and
canonical-quartet enumeration agrees with both native plans. The internal space
has 86 monomials and the projected frozen plan has 645136023 primitive visits;
with the unchanged first source it totals 882874977, below the unchanged cap.
This is an executable work-budget audit, not an executed endpoint or timing.
Generated-map work increases from 516613140 to 656462280; optional curvature
numeric storage increases from 5592 to 188096 bytes. Congruence prices 9216
scalar contributions for this single-spin fixture.

For producer `624c4692219c3c055051cd04edec2edc1105ab8b573f16777678acb7984d939b`,
the complete host-integral module passes 231 cases without skips. Another 63
public-CPU/source-identity cases and all ten native/API/CLI groups pass. The
native API group now includes actual converged spherical-d RKS/UKS batches:
projected/public frozen ERI work is 3276/6758 and 9759/17294 respectively.
Complete electronic Hessians agree within the retained `2e-8` gate. Optional
storage forces the ordinary matrix; its larger work refuses the projected-only
cap transactionally and succeeds when independently budgeted. These compact
endpoint results do not qualify the molecular ICH3I saddle. Documentation links
and the full Sphinx warnings-as-errors build also pass.

The subsequent bounded unchanged-physics ICH3I attempt completes the native
stationary checkpoint at 546.038 seconds, then refuses the **different**
`CPU Hessian complete XC map-work cap exceeded` gate. Process wall time is
546.08 seconds and peak RSS is 377348 KiB. Independent saved-state validation
again accepts energy/density/root/gradient without a reference SCF or displaced
state. No Hessian source preparation or polynomial congruence executes for this
molecule, so neither the 81-AO selected schedule nor raw H/modes is qualified.

The new metadata-only `xc-plan.json` audit retains the actual default level-3
version-4 grid: 87458 points, 81 AOs and CPU tiles of 256. Per direction, first
XC maps require 40403082290 terms and curvature maps require 69026978216; all 18
directions require 1969741089108, above the unchanged `2^40 = 1099511627776` cap.
`xc-independent-audit.json` independently sums the emitted map work and confirms
that this is the real scheduled envelope, including the two curvature
pullbacks, not another whole-registry or direction-count overestimate. This
one-off audit is not a permanent source-spelling CI test. Do not rerun the same
expensive SCF/gradient or raise this cap. Reduce actual XC contraction work,
then qualify its components and all complete-request budgets before considering
another materially changed endpoint attempt.

Permanent component tests compare full projected frozen value/gradient/raw H
with the public matrix and independent libcint scalar stencils, for s through f,
both representations/spins, and contracted/multiple shells on three distinct
physical owners. They retain unchanged numerical gates and exercise caps,
scaling fallback and snapshot invalidation. Source-bound terminal results, not
this test description, determine acceptance. This decision alone does not
qualify native molecular raw H/modes, a practical 25–40-atom endpoint or the
requested-level nickel transition-state optimization.

## Rejected alternatives

- Tightening planning further cannot eliminate the ordinary schedule's actual
  over-cap work. Keep accurate planning, but reduce the physical computation.
- Increasing the primitive cap or weakening symmetry/SCF gates only hides the
  problem and does not establish practical complete-consumer behavior.
- Do not change the public basis or stationary variational space to Cartesian.
- Do not project one-electron/response owners or introduce a second recurrence
  merely because the frozen ERI contraction permits an algebraic pullback.

## Revisit when

A measured complete-consumer profile justifies a different contraction schedule,
or a qualified basis extends beyond the bounded s–f polynomial domain. Preserve
the independent raw-matrix oracles, finite work/storage accounting and explicit
fallback contracts when revisiting the representation.
