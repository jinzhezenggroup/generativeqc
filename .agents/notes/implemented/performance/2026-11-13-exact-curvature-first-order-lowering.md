# Decision: lower only inactive mixed ERI fields, not curvature quartets

Status: implemented
Date: 2026-11-13

## Problem

The first-source translation elision does not apply to a complete curvature
quartet: even when its right integral direction vanishes, explicit moving D/W
weights still contract its value and first gradients. The native curvature
owner nevertheless computed a full gradient-JVP for every quartet in every
column, including quartets whose physical centers have identical motion. Sparse
Cartesian columns amplify this unnecessary second-order primitive work.

## Decision

When the symmetry-unique consumer observes exactly zero relative right motion
in all four independent mathematical slots, explicitly call the existing
canonical factorized value/gradient-only specialization. Retain the value and
all twelve first partials and supply a zero mixed field to the existing
contractions/scatter. All moving density and energy-weighted-density terms,
ordered weight-map incidences and owner-validation cadence remain unchanged.
Nonzero relative motion, however small, keeps the complete mixed primitive.

Build the first specialization in a separate translation unit from the same
`emit_factorized_second(output="first")` algebra. The native wrapper introduces
no recurrence, normalization, spin factor, numerical differentiation or oracle.
The generic generated operation never silently changes its own output ABI when
its numeric direction happens to be zero. The ordered schedule remains the
exhaustive full-mixed bounded comparison path.

Both first-only and mixed visits count against the unchanged ERI admission cap.
The internal cumulative work record exposes their partition, including failed
attempts. Public aggregate curvature ERI work still includes both operations;
do not report first-only work as absent. Numeric heap plans remain conservative;
there is no persistent geometry/jet cache or additional resident precondition.

## Rejected alternatives

- Omitting whole translation-zero curvature quartets loses first-gradient times
  moving-density terms and value/first consistency fields.
- A numerical equality tolerance would delete arbitrarily small real curvature
  directions. Use exact relative zero only.
- A second hand-derived first-gradient recurrence would duplicate the compiler's
  scientific owner. Reuse its already explicit first-order lowering instead.
- Reinterpreting a full-output primitive's zero-direction request as a different
  packed ABI would break the generic primitive contract.
- Larger work caps or source-only 32-site timings do not establish practical
  complete-electronic scale. Quartet/map work remains quartic.

## Invariants

- All finite output fields and D/W response contributions retain their original
  semantics for Cartesian and real-spherical AOs through f and both spins.
- Every physical-source incidence is validated and contracted; no whole quartet
  is screened or elided by this change.
- Capacity, aliasing, immutable-owner and transactional publication gates remain.
- Current chemistry must retain its actual producer identity. Old native Ni and
  running IHH receipts must not be restamped onto this changed build.
- Public molecular ECP saddle admission stays closed pending separate real-
  element state, raw-Hessian and mode qualification.

## Evidence

The first isolated source run fails compilation under `-Wall -Wextra -Werror`
because the existing first-only generated code did not mark its unused numeric
direction parameters. Preserve this attempt under
`build/qualification/host-curvature-first-lowering-20261113`.
The correction marks only those first-specialization parameters
`[[maybe_unused]]`; full-output mathematics and emitted full-source bytes are
unchanged. The first CPU build, initial source/codegen files and failure logs
remain distinct from subsequent validation.

Executable coverage includes independent libcint multistep primal controls,
all-spin/representation/angular schedule parity with explicit D/W motion and
common translation, tiny distinct motions, failure/recovery, and dense/sparse
3/8/32-site source columns. Final results and complete-endpoint qualification
must be read from actual terminal receipts, not inferred from this design.

The corrected production source identity is
`85765caef821f34958cf93d99139bc0051d34a1311bbc9df39629c19a94e5e20`;
the frozen core SHA256 is
`645e448f1f7c11d717def5dc3c4a1e5029d2eca3f862282f10e1755c566262b8`.
On that source, the independent source suite passes 119 tests, factorized
primitive coverage passes 28 tests, all seven targeted native/API/CLI groups
pass, and 123 targeted Python regressions pass with five expensive chemistry
cases deliberately deselected. Native/Python formatting, lint, the diff check
and the Sphinx warnings-as-errors build also pass. These results do not make
the earlier failed compilation attempts successful.

The separately executed real Cs/H RKS public Calculator endpoint passes its
unchanged canonical manifest gates. It retains one complete raw Hessian and
two HVPs, actual central states, and cold independent displaced states at all
three prescribed spacings. The maximum raw-Hessian errors are
`7.06688172591452e-7`, `7.320286671275156e-8` and
`9.014619412051061e-9` Eh/Bohr²; native raw asymmetry is
`5.601803743093825e-13` and the minimum independent displaced occupied-root
overlap is `0.9999970099622799`. The central energy, density and gradient errors
are respectively `1.7418965575499712e-13`, `1.5396389718702608e-10` and
`4.059974578751735e-12`. There is no displaced production SCF, runtime compile,
external production engine or post-hoc symmetrization.

The public Hessian endpoint takes `135.35577086545527` seconds, including its
cold primal; its six response columns report 76 iterations, 164 induced Fock
actions, 2,002,470 first-source and 4,398,324 curvature-source ERI primitive
visits. Each HVP additionally executes its own cold primal. The measured
Hessian-plus-two-HVP interval is `186.21813269797713` seconds; the complete
qualification process, including independent references, takes `4:16.01` and
peaks at 570,748 KiB RSS. This process RSS is not a derivative-owned memory
reservation or an isolated production-only peak. This is a checkout public
endpoint receipt, not installed-wheel, molecular-TS, Ni or 25--40-atom scale
qualification, and no before/after endpoint speedup is inferred.

A distinct snapshot-enabled qualifier is compiled against this frozen core;
its executable SHA256 is
`ef2308ddbc2f7ae5346f567df7ffc8eabb7bb900eb21fe27dca665e16e3d94ea`.
The old unsnapshotted producer is again refused before any independent oracle
SCF. Compiling this qualifier grants no real-chemistry acceptance, and the
original running IHH producer retains its older source/build identity.

The same frozen source also passes the real Cs/H cation doublet public UKS
endpoint, without changing its grid, basis, spin, manifest or reference policy.
Its three full-matrix errors are `5.296098789242532e-7`,
`5.8858029616803975e-8` and `5.2988538001841334e-9`; native raw asymmetry is
`1.9162449405030202e-13` and minimum displaced occupied-root overlap is
`0.9999967531795604`. The public Hessian takes `234.4989824993536` seconds;
the Hessian-plus-two-HVP interval is `320.49792462587357` seconds. Complete
qualification including references takes `6:46.90`, peak RSS 576,108 KiB.
These measurements likewise do not qualify installed or saddle consumers.

## Consequences

Sparse directions can avoid mixed-primitive work on inactive quartets without
discarding first-order response physics. The core includes an additional
generated first-specialization object. Enumeration, all compiler contractions,
SCF, response and remaining mixed primitive work are not removed; favorable
asymptotic or full molecular scale claims require separate evidence.

## Revisit when

Source-driven shell/radial or multi-column reuse, or a rigorously bounded
screening schedule, has independent numerical and complete-electronic endpoint
evidence. Preserve the ordered fallback and actual derivative-order accounting.

## References

- [First-source-only translation elision](2026-11-13-exact-first-eri-translation-elision.md).
- [Current curvature contract](../../../../docs/developer/dft_hessian.md).
- Issues #2844--#2848.
