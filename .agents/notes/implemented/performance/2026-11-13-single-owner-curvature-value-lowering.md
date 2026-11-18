# Exact scalar lowering for single-physical-owner ERI curvature

## Context

The preceding exact first-order lowering retains independent mathematical-
center gradients even for a quartet whose four AOs all belong to one physical
atom. Those gradients are not individually zero, but every physical nuclear
scatter is zero: all four slots share the same origin and both relative motion
legs vanish. Heavy atoms with many AOs can repeat this unnecessary derivative
work in every molecular Hessian column. The value and its moving-density scalar
contraction still participate in source consistency checks and cannot be dropped.

## Decision

Select a canonical value-only operation for a symmetry-unique quartet only when
all four immutable AO atom-owner indices agree. Reuse the existing factorized
Graph algebra with explicit `output="value"` lowering, compiled in a separate
translation unit. Do not implement another recurrence or infer output packing
from a numeric zero direction. The scalar ABI writes exactly one value even
when the generic caller supplies nonzero mathematical-center motion.

Preserve quartet enumeration, every compiler weight incidence, scalar value and
moving-D contractions, validation cadence and conservative work/heap admission.
Track actual value-only, first-only and mixed primitive visits; all three count
against the same unchanged primitive cap. Ordered execution remains the complete
full-mixed bounded comparison path. No cache or resident resource assumption is
introduced.

## Rejected alternatives

- Common coordinates or right motion do not identify physical atoms. Distinct
  owners retain their independent nuclear derivatives even at coincident sites.
- Dropping the whole quartet loses scalar source and moving-density fields.
- Reusing the first-only ABI and pretending its computed gradients are free
  hides real numerical work rather than reducing it.
- Relaxing tolerances, chemistry, resource caps or saddle gates is unnecessary.

## Invariants

All spin, normalization, spherical expansion and s--f contracts retain their
compiler owners. Arbitrarily small distinct motions still reach the required
mixed operation. Numerical output is transactional and owner/resource failures
remain visible. Public ECP saddle admission stays closed. Historical native Ni,
Cs and IHH evidence must retain its original production identity.

## Evidence

The predecessor is frozen in
`build/qualification/host-curvature-first-lowering-20261113`; its public Cs RKS
and UKS, source tests and failed IHH reference discovery are historical inputs,
not acceptance of this changed source. New source/build, independent scalar ABI,
all-spin/representation source and complete-endpoint receipts belong under
`build/qualification/host-curvature-value-lowering-20261113`.

The new production source is
`92ce37caf5b4880cf15c57d22bc878312bfcc91cb4a5172f2bf941934f3234bd`;
frozen core SHA256 is
`4bffe15b5f1dd8fb32fbacd075bec1e54af09fd81a907f58375ab90ac20a803a`.
The independent source suite passes 120 tests, including coincident distinct-
owner curvature, arbitrary small motion, common-translation moving D/W, both
spins, both representations and s--f independent primal controls. Scalar/full
factorized primitive coverage passes 23 tests; the separate factorized-host
suite passes 12. Seven native/API/CLI groups and 123 targeted Python regressions
pass (five opt-in heavy cases deselected). Formatting, lint, compiler structure,
diff and Sphinx-W checks pass. A retained one-off comparison confirms full-jet
emission is byte-identical to the predecessor; this is not a permanent source-
fingerprint test.

At 32 single-s-orbital source sites, actual unique dense visits partition as
32 value / 0 first / 139,624 mixed, and sparse visits as
32 value / 123,225 first / 16,399 mixed. Both still total 139,656 primitive
visits; the ordered comparison retains 1,048,576 full-mixed visits. These are
source-only measurements, not electronic or molecular-scale qualification.

Both canonical Cs/H public RKS and cation-doublet UKS complete raw H and two
HVPs with unchanged native/independent states, all three reference matrices,
root and numerical gates. Their finest matrix errors are respectively
`9.014619412051061e-9` and `5.2988538001841334e-9`; raw asymmetries are
`5.601803743093825e-13` and `1.9162449405030202e-13`. Public H endpoint times
are `134.09772000927478` and `233.4077392788604` seconds, with matrix-plus-two-
HVP intervals `184.56941487826407` and `318.80661191325635` seconds. Complete
qualification, including references, takes `4:13.91` / `6:45.15`, peak process
RSS 570,580 / 574,592 KiB. Those process peaks include oracle work and are not
isolated production peaks or derivative-owned reservations.

These timings are close to the predecessor's 135.36 / 234.50-second public
H endpoints: they do **not** establish material endpoint acceleration. Each
H still reports six response columns, 2,002,470 first-source and 4,398,324
curvature-source primitive visits; response iterations/Fock actions remain
76/164 for RKS and70/152 for UKS. Do not confuse fewer derivative fields with
fewer quartet visits or resolved molecular-scale work. Further performance
work must target measured complete consumers, not repeat small order-lowering
changes and extrapolate a 25--40-atom result.

## Consequences

Single-owner quartets avoid independent first and mixed derivatives without
discarding value work. Multi-owner enumeration, maps, SCF, response and other
sources still execute. This is not an asymptotic or 25--40-atom endpoint claim.

## Revisit when

Source-driven pair/radial or multi-column reuse has separate independent and
complete-electronic evidence. Preserve the explicit exhaustive fallback and
truthful derivative-order accounting.

## References

- [Predecessor first-only lowering](2026-11-13-exact-curvature-first-order-lowering.md).
- [Current integral curvature contract](../../../../docs/developer/dft_hessian.md).
- Issues #2844--#2848.
