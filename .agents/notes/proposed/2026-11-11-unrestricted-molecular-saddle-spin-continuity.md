# Decision: preserve both occupied spin roots in molecular saddle correction

Status: proposed qualification; adapter changes are pending publication
Date: 2026-11-11

## Problem

The molecular adapter retained one restricted occupied subspace. Removing its
UKS admission refusal alone would either reject the unrestricted state later
or incorrectly compare only one spin. A continuous alpha root cannot certify
beta continuity. Rectangular singular-value decompositions can also hide a
changed occupied population.

## Decision

Retain a detached occupied coefficient block for each physical spin channel.
Require state and MethodIR spin identities to match, finite native
coefficients, and integer occupations: zero/two for RKS or zero/one for UKS.
At least one channel must contain an occupied orbital.

At every trial geometry, compare each channel to the **last accepted** state's
corresponding channel with the actual cross-geometry AO overlap. Each occupied
population must be unchanged and its smallest principal overlap must satisfy
the existing threshold. An empty channel contributes unit continuity only if
it remains empty. Occupied gauge rotations and signs do not change the test.
Rejected trials never replace the accepted root. Detached per-spin blocks
outlive electronic owners; observations expose both populations and both
overlaps instead of only a combined success scalar.

The native C++ corrector still owns steps, trust updates, mode tracking,
attempt limits and fresh final classification. The Python adapter does not
introduce a reference engine, displaced production SCFs, an external optimizer
or a preliminary electronic model. Electronic owners close before additive
two-body D3(BJ) curvature is published.

Admission still requires existing public forces and Hessian qualification.
ECP, DF, CUDA and unqualified PBE0 UKS remain refused. This work does not
provide a Python-free installed TS CLI or the requested Ni/ECP TS endpoint.

## Rejected alternatives

- Checking only alpha, a summed density, or one largest singular value can
  conceal a beta root change or the loss of one occupied orbital.
- Comparing rectangular subspaces without population checks changes the
  electronic-state contract rather than tracking its root.
- Tracking the last attempted rather than accepted state lets rejected
  geometries move the reference root.
- An occupied-space continuity check is not an electronic stability proof,
  global ground-state guarantee or unrestricted root-selection algorithm.

## Evidence and remaining gate

The current native/admission/lifetime and occupied-space suite passes 92 tests
in 2.38 seconds; the expensive new molecular oracle is explicitly skipped in
that invocation, not counted as accepted. Controls include both spin layouts,
both one-empty-channel arrangements, independent occupied gauge rotations,
single-orbital loss hidden by another continuous orbital, population changes,
actual nonidentity AO cross metrics and detached coefficient ownership.

The first unit invocation had an over-specific expected refusal message for
PBE0 UKS: the adapter correctly refused even earlier than the public-Hessian
guard. Its failed result is retained; only the assertion was corrected.

The first new H3 molecular invocation was terminated with actual wrapper exit
143 before a complete Hessian, after identifying a missing harness serializer
for immutable observation mappings. Its driver, pending receipt and partial
events are retained and are not scientific acceptance. A separate preflight
command then lacked the checkout Python path and exited before starting a new
molecular run. Neither failure changes the production method or any gate.

The corrected opt-in oracle uses H3 doublet, canonical real-spherical def2-SVP
H data, native-default grid and physical Gaussian/VWN-RPA B3LYP plus uncut
two-body D3(BJ). It requests a native maximum-100-attempt saddle run, fresh
final derivatives, unscaled frequencies, all raw final Hessian columns against
three independent moving-grid-gradient stencil sizes, and two fresh public
HVPs. Independent SCFs receive no native density. Its source/library manifest,
events and artifacts are under
`build/qualification/uks-molecular-saddle-20261111/`.

At this note's creation that molecular process is **still running**; no
converged saddle, final matrix, frequency acceptance or installed qualification
is claimed. Publication remains conditional on actual completed evidence.

## Subsequent actual H3 result

The corrected process actually exits one in 861.99 seconds. The native
corrector returns `target_not_negative` with a valid retained initial state,
zero attempted steps, no callback failures and no stationary verification.
The selected antisymmetric stretch has positive curvature
`0.12442943177985047` at the declared 1.7-Bohr spacing. It does not fabricate
a saddle or harmonics. All nine raw Hessian columns and the last geometry/
gradient are retained in `last-native.npz`; the maximum raw symmetry error is
`5.451264439848558e-13`, with converged response residuals.

This failed saddle-input expectation is not changed to acceptance. The separate
independent cold-SCF diagnostic actually exits zero in 293.2352 seconds. All
nine original raw matrix columns agree at the three original stencil sizes:
maximum errors `2.1321316101552412e-9`, `2.907328947032717e-9`, and
`1.110930722414416e-9`, below the unchanged `1e-5` molecular gate. Independent
reaction curvatures are positive at this same 1.7-Bohr input, matching the
retained native value. The nonstationary raw matrix's rotational negative
directions are not a verified vibrational saddle: the corrector removes five
rigid directions for stepping and finds four nonnegative active directions.
No frequencies are published for the refused nonstationary point.

The independent screen additionally matches cold native/reference states at
1.9, 2.1 and 2.3 Bohr. At 1.9 Bohr the antisymmetric stretch is genuinely
negative at all three independent steps, about `-0.10904343`. The new positive
qualification attempt therefore declares 1.9 Bohr explicitly, with unchanged
physical MethodIR, basis, grid, spin, convergence controls and derivative
gates. It is a separate running process, not a replayed success for the original
failed input. Original failed source bytes, receipt, full matrix and events
remain retained. The adapter remains unpublished pending real completed
positive molecular qualification.

## Subsequent actual 1.9-Bohr result

The separately screened 1.9-Bohr process actually exits one after 2292.26
seconds of pytest execution. It performs three complete raw Hessians and two
accepted native steps, reaching the symmetric 1.788455384-Bohr spacing. The
corrector returns `target_not_negative`, not convergence: the tracked reaction
curvature is now positive (`0.003700008661424156`), gradient maximum remains
`0.00040085183493926955`, and no stationary verification or harmonic result is
published. No electronic callback fails. Per-spin minimum occupied overlaps
remain above 0.999 throughout, and the last raw symmetry error is
`5.909725833697088e-13` with response residuals below `3.874e-12`.

The initial independent negative curvature was a necessary local-input screen,
not proof that this electronic branch contains a reachable first-order saddle.
Do not weaken the negative-curvature guard, change the input retrospectively or
use the two accepted steps as a successful molecular qualification. The full
receipt, geometry, gradient, raw matrix and frozen input/library manifest remain
under `build/qualification/uks-molecular-saddle-20261111/h3-1p9*`. The pending
UKS adapter still lacks a completed positive saddle/frequency control and is
not published on the strength of this failed attempt.

## Completed-cohort local recovery

The separately verified recovery
`build/qualification/completed-i-h3-recovery-20261111/h3-completed-failed-saddle-cohorts.tar.gz`
contains 379 members and 49,922,309 bytes; SHA256 is
`fefc5ef1cbc538dde08775f4648f6ac8b2f8f5698ad7262a7e6bcb83eb332780`.
Its manifest SHA256 is
`f42788a81732a28fe3e0a1ee4a082a32acd00c21ad6e6cae9e09a819f240bae5`.
Every member and original was rehashed after creation. Both failed input
attempts, the initial terminated harness, independent curvature diagnostic,
frozen sources and native library are retained with their distinct scopes.
This recovery includes no running scientific job and promotes no TS result.
The archive generator is `/tmp/gqc-completed-i-h3-recovery-20261111.py` and is
also retained in the archive; earlier recovery objects are unchanged.

## Separate SiH3 input characterization

A different all-electron doublet candidate uses the named real-spherical
def2-SVP Si/H pack and the same physical B3LYP/VWN-RPA plus two-body D3(BJ),
native-default grid and fixed derivative gates. This is not a rescue or
relabeling of either H3 attempt. Both independent input preparation and
finite-difference screening are validation-only, never production work.

The first independent planar SiH3 screen actually exits one after 65.65
seconds. A one-dimensional radial root with its normal along z leaves a
`7.575e-7` maximum Cartesian gradient, failing the unchanged `1e-7` stationary
input gate. A finite angular quadrature need not retain that triangle's
120-degree symmetry; a scalar radial root alone does not establish Cartesian
stationarity. The failed source/receipt remain under
`build/qualification/sih3-uks-independent-saddle-screen-20261111/` and
`/tmp/gqc-sih3-uks-saddle-input-screen-20261111.py`.

A separately declared orientation uses normal `[111]` and hydrogen directions
`[1,-1,0]`, `[0,1,-1]`, `[-1,0,1]`, so its D3 proper rotations are signed
coordinate permutations of the angular quadrature. No force gate is widened,
grid changed or derivative formula altered. The same-method independent
radial root is `2.7905333432392343` Bohr. Fresh native/reference states match
in energy/density/gradient to `1.137e-13`, `8.649e-13`, `7.007e-14`.
Independent inversion curvatures at `0.003`, `0.001`, `0.0003` Bohr are
`-0.07086923349314454`, `-0.07086923379849167`, `-0.07086923422885327`.
The independent twelve-column matrix is still being computed at this
checkpoint; one negative directional curvature is not yet a complete index
test. Its receipt remains explicitly diagnostic-only, not accepted as a
native molecular saddle or final frequencies.

The separately named opt-in
`tests/python/test_molecular_saddle_sih3_reference.py` declares an initial
2.80-Bohr triangle and a 0.02-Bohr silicon displacement, requiring native
correction, fresh stationary verification, full final raw matrices at all
three original independent steps, two fresh public HVPs, both occupied spin
counts `(9, 8)`, and unscaled final frequencies. Preparation/collection does
not count as its execution or acceptance. Original H3 source bytes stay frozen.

## Completed independent SiH3 screen and new native attempt

The symmetry-declared independent screen actually exits zero, recording
474.5203 seconds. All twelve raw columns at the explicitly declared
0.001-Bohr fourth-order stencil are retained; raw asymmetry is
`7.299295196050437e-10`. Independent harmonic analysis has six vibrational
modes and exactly one significant imaginary frequency (`-717.4542942109881`
cm^-1). This is an independent stationary-input characterization, **not** a
native final Hessian/frequency qualification; the receipt's accepted-saddle
flag remains false. Earlier H3 and z-normal SiH3 failures remain unchanged.

The separately declared native SiH3 test has now started with a finite
10,800-second outer limit. Its frozen manifest is
`build/qualification/uks-molecular-saddle-20261111/sih3-positive-manifest.json`;
the native library still has SHA256
`afb07c88c86bcfa2959cfa35bfcc138266c0d6fed8fe5abe16cac02b10ac0ec9`.
The test, adapter, common response/XC code and library must remain unchanged
during this attempt. At launch no native saddle, fresh final full matrix,
independent three-step comparison, HVP or frequency acceptance is claimed.

The two finished independent SiH3 input screens have a separate verified local
recovery, with no running native-job outputs:
`build/qualification/sih3-independent-screen-recovery-20261111/sih3-input-screens.tar.gz`.
It contains 32 members and 37,729,070 bytes; archive SHA256 is
`a01f17fe4d4c19e57f49ba01066e072d10e927909b1f303cb68b53078db25a4a`,
and manifest SHA256 is
`e3ebe88cbfd6fe08901dd874ddd5591fa4bb4522dacb8be821878527368afc28`.
Every archived member and original was rehashed. The generator
`/tmp/gqc-sih3-completed-screen-recovery-20261111.py` is included. Neither this
recovery nor its independent index test promotes native TS or installed CLI
capability.

## Revisit when

The complete molecular result is available, the public ECP domain is separately
qualified, or a scientifically validated unrestricted state-selection policy
requires more than local occupied-subspace continuity.

## References

- Issues #2844, #2847 and #2848; PR #3062.
- `docs/user/geometry_optimization.md`.
- `tests/python/test_molecular_saddle_spin_roots.py`.
- `tests/python/test_molecular_saddle_uks_reference.py`.
