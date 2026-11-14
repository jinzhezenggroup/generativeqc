# Decision: align independent SP-shell coordinates before matching density

Status: implemented
Date: 2026-11-12

## Problem

The native spherical STO-3G Si/H records retain their input SP-shell ordering.
The independent PySCF molecule used by the light-element gradient oracle sorts
each atom's shells by angular momentum, stably within an angular class. For
silicon this moves the outer s function ahead of the inner p block. Matching
the two raw density arrays without the coordinate map therefore compares
different AO slots, not different electronic states.

The retained native SiH3 run reached a freshly verified first-order saddle in
three accepted attempts and satisfied the five original convergence controls.
Its subsequent test failed a raw density comparison near 0.568 even though
independent total energy and nuclear gradient agreed. That failed pytest receipt
must remain failed; a successful optimizer stage does not imply independent
final-curvature acceptance.

## Decision and invariants

The validation-only `_independent_ao_order` helper in
`tests/python/test_molecular_saddle_sih3_reference.py` expands the native
real-spherical shell blocks and applies the independent stable atom/angular
shell order. Both density axes must use this permutation. Before using it,
the scientific test compares actual native and independent AO values/first
jets and overlap matrices in the mapped coordinates. The density tolerance
is unchanged; orbital signs and rotations within occupied spaces remain
handled by the separate cross-geometry subspace test.

No production AO ordering, SCF density, state selection, integral source or
Hessian changes. Never sort the production basis to make an oracle comparison
pass, infer state identity from energy alone, or weaken a density gate because
the coordinate systems were not aligned. Non-permutation changes of basis
would need their own independently checked transformation and cannot silently
reuse this helper.

## Evidence

The fast test covers interleaved SP contractions and stable radial ordering.
For the retained 12-AO SiH3 geometry, the independently evaluated first-jet map
error is below `9e-16` at 128 deterministic nonaxial points. Full final-state,
three-stencil curvature, fresh HVP and frequency acceptance still belongs to
explicit completed scientific receipts, not this mapping check. A separate
post-validation may consume the retained native matrix but must state that it
did not rerun the optimizer or fresh matrix, and must not relabel the failed
end-to-end test.

## Revisit when

The independent molecule builder preserves input shell order explicitly, or
another representation introduces normalization/component transforms beyond a
shell permutation. Retain actual AO/metric checks at that boundary.

## Fresh complete source-test acceptance

The separate `host-uks-minimal-saddle-v2-20261112` run now returns actual pytest
exit zero and one passing complete `sto-3g` parameter in 1918.19 seconds. It
reruns the native optimizer/full matrix, checks the actual aligned final density
(`5.803e-12` error under the unchanged `2e-7` gate), computes 144 new cold
displaced independent states for three full fourth-order matrices, and checks
two fresh public HVPs and all six unscaled modes. The root minimum is
`0.9988586907`; largest raw-matrix/frequency errors are `5.601e-9` and
`9.938e-6` cm^-1 under the unchanged `1e-5`/1 cm^-1 gates. The native search
uses three accepted attempts and the five original convergence thresholds.

This new source pytest is distinct from the original failed pytest, its
retained-matrix post-validation, and the separately accepted installed ELF
control. It does not qualify def2-SVP, Ni/ECP, scale or the requested Ni TS.
