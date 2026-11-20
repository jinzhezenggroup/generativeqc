# Decision: Share the linked first-order ERI leaf for stationary f shells

Status: implemented
Date: 2026-11-20

## Decision

The core that links the native host Hessian already owns a compiler-generated
factorized first-order ERI leaf. Use it for packaged stationary gradients of
quartets containing f. It returns all independent center partials in one
recurrence pass; J/K weights, radial normalization and physical scatter remain
at the existing stationary consumer. Keep packaged s/p/d kernels and the
three-direction f implementation for force-only cores and standalone targets
that do not link the factorized leaf.

The factorized entry requires a mixed-radix CCA component index through f. A
flat value-ERI helper bounded at d misidentifies f components; tests with signed
f contractions detected this in the original experiment. The selected leaf
determines the actual recurrence pass count, independent of weight channels.
The compile definition belongs to the core target even when a standalone
test compiles the same source file.

## Evidence and limits

- The sealed matched cold Ni/I RKS force pair recorded 123,992,467 versus
  79,755,235 passes and 522.31 versus 133.05 seconds end to end, with the
  same 52 SCF iterations and 56 Fock builds. Maximum independent force error
  remained `1.65e-13 Eh/Bohr`; these are single paired measurements.
- A separate sealed 25-atom mixed Ni/I28 UKS cold force run returned in
  6087.03 seconds, with 20,678,264 KiB peak RSS, 134 SCF iterations, 136 Fock
  builds, 17,319,947 shell submissions and 5,696,726,985 recurrence passes.
  It passed the independent same-geometry state/force comparison: occupied
  overlap `0.9999999999999162`, maximum force error `2.29e-12 Eh/Bohr` and
  energy error `4.25e-12 Eh`. This is a bounded completed scale endpoint,
  not a proven scale speed ratio, a practical Hessian, or an ECP saddle.
- The same frozen 25-atom force comparison passed from an isolated repaired
  wheel with no oracle modules or core override. `f-leaf-wheel/scale-mixed-installed`
  retains 6087.67 s, 20,492,140 KiB peak RSS, the same 134 SCF iterations and
  136 Fock builds, and an identical state/force snapshot hash. This is the
  installed public force receipt, not an installed Hessian or saddle receipt.
- The fold-spherical-cotangent and parallel schedules are separate, not
  part of this decision. Do not infer a composed performance result.

The source-bound paired and scale receipts are under
`/data/jzzeng/qualify-2844-2848-20261120/f-first-leaf-trial/`. For the scale
comparison use `scale-mixed-cca/validation.json` and
`scale-mixed-cca/source-bound-comparison.exit`; earlier comparator exits
reflect a missing oracle dependency or the wrong Python package, not the
scientific verdict. Supersedes
`../../proposed/2026-11-20-reuse-factorized-first-leaf-for-stationary-f.md`.
Related: #2846.
