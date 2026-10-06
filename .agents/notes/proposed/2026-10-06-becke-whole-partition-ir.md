# Decision: authenticate the complete Becke normalized-product AD graph

Status: proposed; compiler/host gates implemented, native integration pending
Date: 2026-10-06

## Problem

PR #1996's original primitive checks reachable canonical scalar companions,
not their full composition. A ratio/log/switch hash cannot prove that the
selected objective has the correct complementary factors, all atom products,
normalization denominator or owner semantics. #1894 requires that proof before
native promotion, in addition to fixed-grid and complete endpoint gates.

## Decision and ownership

Expose the existing scalar primal constructor so both scalar and composed IR
use one scientific algebra implementation. Build the full normalized-product
objective in the same Graph and differentiate the entire objective once.
Distance leaves bind the existing point/center norm owners; there is no second
force formula, runtime reference dependency or coordinate-by-grid Jacobian.

Every pair ratio and Becke polynomial is actually present in the reachable
graph. Mu saturation includes equality. Factor clipping kills derivatives only
strictly outside [0,1]; equality keeps the raw tangent so a rounded-zero factor
can have a nonzero derivative. Products use complementary oriented factors;
the denominator includes every atom product, and the owner is a valid integer
leaf without a tangent. The symbolic product is the mathematical definition,
not a replacement for the existing stable log/zero execution prescription.

Recognition authenticates reachable primal AND JVP roots against the canonical
composition, ignoring supplied identities. Changed objectives/derivatives,
cross-graph roots, unsupported partition or wrong atom domain fail closed.
Matched operations carry the composition identity and atom domain. Both
experimental emitters revalidate them; planners reject a mismatched domain.
Legacy scalar-only operation identities remain stable for frozen experiments,
but do not meet full-composition production acceptance.

## Evidence and limitations

- 294 composition/coefficient/indexed/phased/grid-response host gates pass.
  Whole graph tests cover iterations 1/3/5, dimensions 1/2/3/8, stale supplied
  metadata, changed primal/JVP, swapped roots, foreign roots, unsupported radius
  adjustment and domain mismatch. Complete generated JVPs bind point motion and
  every center-distance tangent and match generic and coefficient contractions.
- All 20 scalar and 20 mixed AD identities equal the committed measured base.
- Final broader host cohort: 619 gates pass, including all Becke schedules,
  grid/native response, mixed quadrature response and existing stationary
  phased-resource admission. Ruff and compiler dependency checks pass.
- Concrete 48/96-atom graph witnesses contain 79,493 / 320,261 nodes. The first
  construction plus recognition is observed at about 2.25 / 11.32 CPU seconds
  in a fresh interpreter. This compiler cost is not a kernel or endpoint win.
  Do not rebuild these witnesses per point, tile, seed or geometry. A native
  dynamic-domain/template integration must account for cold compiler work and
  preserve the actual domain contract instead of trusting a representative
  small graph or silently applying one fixed-domain operation to another size.
- Source generation remains independent of the public runtime and real GPUs.
- Native ordinary/composite owners, concurrent admission and cache invalidation,
  RKS/UKS forces, final matched-source GPU gates, per-phase attribution and
  paired 48/96-atom PBE0 E+F endpoints remain pending. No production dispatch is
  enabled and #1894 is not complete.

## References

#1894; #1830; #1950; PR #1996;
`2026-10-06-becke-dense-coefficients.md` preserves the second losing schedule;
`2026-10-05-becke-partition-primitive.md` preserves the first losing schedule.
# Follow-up: bounded native domain

The exact-domain limitation recorded here is addressed in the draft branch by
`2026-10-06-becke-native-domain.md`. That follow-up authenticates an actual
active-prefix primal/JVP family rather than reusing an exact witness at other
dimensions. It does not promote the losing schedule or close the issue.
