# Proposal: analytic scalar-ECP UKS candidate and the unresolved spin-action gate

Status: proposed for full canonical/public UKS qualification; scoped CPU candidate accepted separately
Date: 2026-11-11

## Initial scope before the subsequent correction

The [native Gaussian ECP source port](../implemented/numerics/2026-11-11-native-gaussian-ecp-response.md)
is implemented and published. A subsequent working-tree prototype shares one
authenticated metadata predicate between `ECPSecondOrder` and the low-level
composite response admission. It permits UKS only for CPU, direct full-range
graphs whose ECP terms have power two and local/s/p/d projectors. The native
snapshot ABI still rechecks the complete geometry/operator and current token.
Other scalar-ECP UKS operators remain refused. Public ECP Hessian capability
is unchanged and closed. Do not publish this candidate as qualified yet.

## Fixtures and retained failures

The explicit synthetic Na/H valence basis has Na s/p shells with exponents
0.65/0.45, H s exponent 0.8, and a ten-electron Na core. Each local/s/p/d ECP
channel has signed power-two primitives `(0.67, -0.12)` and `(1.23, 0.081)`.
It is not a canonical def2 potential or a qualification of the requested
Ni/I/Cs systems. Both UKS occupied spaces are nonempty: `(2, 1)`.

Separated-H NaH2 gives different unseeded native/independent stationary roots;
its failed energy/density/force screen is preserved, including the independent
same-native-density energy/residual diagnostic. Removing Na p on that geometry
also fails. These attempts are not relabeled or replaced by the later fixture.

The distinct Na + bonded-H2 geometry in
`tests/python/test_analytic_ecp_uks_hessian.py` has independently matching SCFs
and first forces on the explicit 16x8x16 atomic quadrature. Neither solver is
seeded with the other's density. Public second-order capability remains empty.

## Original complete-matrix processes remain failed

The new test first stops at its strict arbitrary-AO spin-action comparison in
7.02 seconds. A second, separately named process defers that same assertion
until after the physical Hessian gates; it does not remove or loosen it. This
second process exits **one**, with one failed test in **213.73 seconds**.

Before that failure it computes an unsymmetrized complete 9x9 electronic
Hessian and two fresh HVPs, one unscaled unit-Na-x and one mixed direction.
The same single native owner supplies analytic ECP first/fixed-density second
sources, both spin responses and all physical components. Raw symmetry,
translation, H@v consistency, component reconstruction and response residual
gates pass. Native complete-matrix plus two-HVP time is 201.5755873 seconds,
including this process's JIT work and concurrent installed Ni work; this is
not an isolated performance claim.

Independent moving-grid gradients reconverged without density seeding give
complete raw matrix errors:

| Difference step (Bohr) | Maximum raw matrix error (Eh/Bohr²) |
| --- | --- |
| 0.003 | 4.838986592092809e-6 |
| 0.001 | 5.372590072827421e-7 |
| 0.0003 | 5.15986430804638e-8 |

All are below the unchanged `1e-5` molecular gate. Nevertheless the overall
test is **not accepted**: its arbitrary symmetric AO alpha-only/beta-only
induced-Fock errors are `[1.4666046155298318e-12, 0.0015167214396569761]`,
and the second fails the original `2e-9` diagnostic gate. No public or complete
UKS qualification follows from the preceding molecular checks alone.

## Numerical diagnosis, not a replacement qualification

Controlled independent derivative evaluations narrow the discrepancy:

| Independent response evaluation point | Beta-action max error |
| --- | --- |
| Independently reconverged state | 1.5167214396569761e-3 |
| Exact native orbitals/occupations, no SCF reseeding | 2.690736819204176e-4 |
| Exact native point features, no SCF reseeding | 3.715633283718489e-6 |

Changing only the independent Libxc density threshold between defaults,
`1e-15`, `1e-18`, `1e-24` and zero does not remove the beta-action failure.
Retain those diagnostic attempts. Initial PySCF customization harnesses fail
because the API requires an explicit per-component omega sequence and custom
names matching its lowercase DFT canonicalization; those harness failures are
not scientific passes and their original scripts/logs remain intact.

The same-point residual is dominated by near-node minority-spin samples, not
the largest far-tail pointwise kernels. At point 943, rho-alpha is
`0.11122928188611153`, rho-beta `7.4225911645623816e-9`, and its beta-AO(0,0)
native-minus-Libxc contribution is `3.946307376337172e-6`. Streaming versus
resident native point responses differ by at most `2.911e-11` in their weighted
coefficients, so simply disabling the bounded cache does not explain the gap.

As a **numerical diagnostic only**, mpmath differentiates the canonical primal
at 50 and 80 digits, independently of the compiler's derivative roots. This
shares the primal mathematics and is not an independent molecular oracle.
The dominant point's 50/80-digit contribution agrees to `7.025e-48`;
native-minus-high-precision is `-2.742299350310983e-9`, while Libxc evaluated at
the same features differs by `-3.949048905290064e-6`. Points 929, 1152 and 1404
give native high-precision errors below `6.315e-11`, with substantially larger
Libxc errors. These four samples do not qualify the complete arbitrary-AO
response, and even the dominant native sample does not pass `2e-9` by itself.
Do not use this observation to declare the original failed test successful.

A subsequent, separately frozen physical-subspace diagnostic applies all 13
unit occupied/virtual spin-density directions and projects both independent
and native responses into every output-spin OV block. Its predeclared `2e-9`
gate passes with maximum error `3.4010537174065444e-10`. This distinguishes
the physical coupled CPKS map from the ill-conditioned unrestricted AO probe;
it does not retroactively pass or qualify the latter. The complete diagnostic
receipt keeps `unconstrained_full_ao_qualified=false` and overall
`accepted=false`. Any qualification contract based on this distinction must
be explicit and justified rather than silently deleting the failed gate.

## Original evidence and next work

All original failed/accepted SCF screens, failed full tests, source snapshots,
raw XML properties and diagnostic scripts/results are retained under
`build/qualification/uks-analytic-ecp-20261111/` and `/tmp/gqc-uks-*`.
`spin-tail-point-audit.npz` retains complete ground fields, AO jets, native and
independent weighted kernels, the unscaled density direction and native state.
`spin-tail-point-audit.json` authenticates the source cohort/library and lists
the actual contributing points, enabling further diagnosis without another SCF.
The tested native library is the resource-cohort SHA256
`2436a788ed7b1c8f504872e4ffea7cee061389eeb6f1314abd7a93ee18bb42d7`.

The separate admission/source/fallback suite passes 165 tests in 3.55 seconds.
Its initial eight failures were manually constructed grid-only fixtures missing
the new analytic-lane field, not numerical failures; explicit initialization
restores their nonfinite/convergence rejection controls. That two-line fix is
published independently of this unqualified UKS candidate.

Next determine an independently justified stable fixed-state spin-response
contract, including state/feature conditioning and the full AO space, before
deciding whether compiler algebra/precision or a bounded admission/refusal
needs changing. Do not clip minority-spin tangents, loosen the physical gates,
drop the failed probe, or substitute reference derivatives into production.
Canonical def2 UKS/ECP, D3 total matrices, installed public endpoints, larger
Ni/I/Cs systems and the requested TS/frequency workflow remain separate gates.

## Frozen failed-candidate recovery

The final raw-retention repetition also exits one, in 214.00 seconds, with the
same strict spin-action failure. Its XML retains the complete native 9x9
matrix, both HVPs and all three independent matrices; these are additionally
extracted into `retained-complete-matrices.npz`. Physical matrix/HVP successes
do not change that process's failed status.

The member/hash-verified local recovery
`build/qualification/uks-analytic-ecp-20261111/failed-candidate-recovery.tar.gz`
contains 87 members / 5,368,936 bytes, SHA256
`434fc96a069ed7efed0f4baf95ffdd5ece7012433c52e9b1612834ed94f676bb`.
It freezes the dirty candidate sources, this note before this recovery footer,
all failed/accepted-scope diagnostic attempts, raw matrix/point inputs and
drivers. Its manifest explicitly sets `accepted=false`,
`unconstrained_full_ao_qualified=false` and `public_ecp_qualified=false`.
It links the earlier verified resource/library recovery rather than relabeling
or overwriting it. Retain this baseline before changing the validation contract
or implementing any subsequent numerical correction.

## Subsequent correction candidate, not a restamp

The separate [B88 stability and exact-input reference decision](../implemented/numerics/2026-11-11-b88-spin-node-response-stability.md)
records a newly diagnosed native cancellation error, its algebraic correction,
and an explicit full-AO/physical-OV/molecular validation contract. The original
failed receipts and recovery above remain unchanged. The subsequent complete
molecular process actually exits zero: five tests pass in 217.91 seconds,
including all 42 exact-input AO columns, 13 physical OV input columns / both
output spins, the complete raw 9x9 matrix and both fresh HVPs. The original
absolute numerical gates remain unchanged under the explicit reference
contract. The synthetic CPU candidate is qualified at that scope only;
canonical def2 and public ECP qualification remain open.

## Subsequent canonical Cs spin evidence

The separate [canonical Cs total-curvature decision](../implemented/numerics/2026-11-11-canonical-cs-spin-total-curvature.md)
now retains completed low-level CsH RKS and CsH+ UKS total raw matrices, two
fresh HVPs per state and all three independent moving-grid-gradient matrix
comparisons. This is a separate source-matched acceptance, not a restamp of
any failed synthetic process above. It resolves only the missing canonical Cs
spin evidence; public ECP, iodine, mixed Ni/ECP and scale qualification remain
separate gates.

## Subsequent iodine RKS acceptance and UKS reference failure

The separately declared canonical IH RKS process actually exits zero in
1631.3904 seconds: real-spherical def2-SVP / I def2-ECP28, native-default grid,
31 AOs and thirteen doubly occupied orbitals. Its complete total raw 6x6 matrix
and two fresh HVPs pass all unchanged gates. Total independent matrix errors
at `0.003`, `0.001`, `0.0003` Bohr are `1.3934244775803961e-6`,
`1.57589897020749e-7`, and `1.4713035709906785e-8`. Raw arrays and the scoped
receipt are in `build/qualification/canonical-i-rks-total-20261111/`.

The original IH+ doublet UKS process actually exits one before a successful
central-state screen: its independent PySCF SCF does not converge in the
original 240-cycle cap. No native Hessian is computed or accepted by that run.
The independent diagnostic then actually exits zero in 108.2896 seconds as a
**diagnostic only**, retaining both a fresh plain-240 and a fresh Newton-240
oracle, neither seeded from native density. Both fail convergence and the
original state gates. Energy errors near `2.8e-9` coexist with maximum density
errors above `0.325`; orbital-gradient norms remain `4.3e-8` and `1.14e-7`,
above `2e-9`. Nearly equal energies do not certify the same electronic root.
This is not the earlier arbitrary-AO Libxc conditioning problem, and neither
the energy agreement nor an oracle change can relabel this failed run.

The diagnostic's independent/native density, orbitals, occupations, energy
frontiers and full iteration traces are retained under
`build/qualification/ih-uks-independent-scf-diagnostic-20261111/`. The failed
original receipt, driver and log remain unchanged. A separate nonaxial IH2
doublet state screen is being attempted with the same potential, functional,
representation and native-default grid; it is a distinct declared validation
fixture, not a claim to fix the original IH+ reference or qualify public I UKS.

## Completed iodine cohort local recovery

The accepted low-level IH RKS run, failed original IH+ UKS run and diagnostic
plain/Newton reference failures now have a separate verified recovery:
`build/qualification/completed-i-h3-recovery-20261111/iodine-completed-cohorts.tar.gz`.
It contains 358 members and 46,712,338 bytes; archive SHA256 is
`e48d5df5a065f1188f079e4388807a45f31a5c7b3f59c40024860bc9c9245822`,
and manifest SHA256 is
`be9a930477129f0b541832468846bbd9d50fa18f83d209cd627d5482abdb2c19`.
Every archived member and retained original was rehashed, including the frozen
numerical sources and native library. The running distinct IH2 cohort is not
included, no failure is relabeled, and earlier recovery objects are untouched.
The generator `/tmp/gqc-completed-i-h3-recovery-20261111.py` is retained inside
the archive. These are ignored local recoveries, not released/published assets.

## Subsequent distinct IH2 doublet acceptance

The separately declared IH2 full-matrix process now actually exits zero. Its
raw 9x9 total matrix and two fresh HVPs pass the original three-step independent
gates; completed postrun verification and local recovery are recorded in
[the scoped IH2 evidence note](../implemented/numerics/2026-11-12-canonical-iodine-uks-total-curvature.md).
The original IH+ SCF failures and earlier recovery objects remain unchanged.
This low-level success does not open public/installed ECP, mixed/scale or TS
admission; the proposed broader gate remains unresolved.
