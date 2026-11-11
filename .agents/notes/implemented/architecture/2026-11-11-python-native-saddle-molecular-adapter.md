# Decision: bind the native saddle owner to one fixed molecular RKS model

Status: implemented
Date: 2026-11-11

## Problem and decision

The native callback controls have independent light-molecule evidence, but a
scratch callback harness is not a public molecular endpoint. Expose that same
C++ owner through `generativeqc.saddle`, with a separately admitted molecular
adapter. Do not duplicate optimizer mathematics, replace the Python-free CLI
launcher, or silently use an external optimizer to fill the remaining CLI gap.

The generic `correct_saddle` binds real synchronous callbacks. The molecular
`optimize_saddle` requires the existing public Hessian/forces capability and an
additional narrow CPU, direct, all-electron, unpolarized full-range hybrid RKS
domain. A qualified D3(BJ) source stays in the same total MethodIR. The adapter
is not admission for UKS, ECP, Ni or the requested mixed-element TS task.

This supersedes only the absence-of-Python-molecular-adapter part of the
[native callback note](../numerics/2026-11-10-native-saddle-callback-and-rigid-subspace.md).
Its distinction between mathematical tests, independent molecular qualification
and the requested installed Ni/ECP CLI endpoint remains in force.

## Invariants and alternatives

- One prepared SCF supplies energy/gradient and the next raw Hessian at a given
  geometry. Re-running a public cold Hessian after separately computing forces
  would obscure work accounting and permit a different electronic root.
- Preserve a detached occupied subspace only when the native optimizer reports
  that geometry accepted. Compare each cold-SCF occupied subspace through native
  cross-geometry AO overlaps to that reference, not the last rejected trial.
  Minimum principal overlap is an explicit fail-closed local policy, invariant
  to rotations within the occupied space. It is not a stability certificate or
  an unrestricted root-tracking implementation.
- Freeze the calculator physical signature and total/electronic MethodIR
  identities, including across observer callbacks. No preliminary method change
  or implicit compatibility route is permitted.
- Release response, AO and prepared-batch owners before geometry-only D3
  publication. A correction failure cannot publish an electronic-only total.
- No Python exception, including an interrupt or observer exception, can unwind
  through a ctypes callback. Preserve diagnostic strings and native failure
  state; initial failure publishes no derivative arrays or numerical metrics.
- Keep final raw Hessians and all unscaled vibrational modes. Harmonic
  publication failure preserves the native result without molecular success.
  Partial failed work counts remain explicitly incomplete, never inferred.

## Validation boundary

Executable ABI tests cover native stepping, fresh final derivatives, capacity,
callback failures, cancellation at the final observation and detached outputs.
Adapter tests cover admission before SCF, model mutation, last-accepted root
reference across a rejected trial, and electronic cleanup before additive
publication. Synthetic lifecycle providers are not scientific qualification.
The existing independently checked ammonia callbacks supply prerequisites, not
an automatic receipt for this new public adapter or an installed TS CLI.

The live serial Ni full-Hessian experiment retains its original frozen numerical
sources and native binary. This new module does not edit those owners. Its
complete public molecular endpoint still needs source-matched real runs and
independent acceptance before claiming that endpoint qualified.

## Installed artifact and negative control

The implementation at `cf3f871af7adbcf7129b0d33c3d50b4c5f97481c` is built in
`build/wheel-public-saddle-cpu`, separate from the still-running original Ni
qualification library. The normal auditwheel repair vendors the pinned reviewed
OpenBLAS provider cohort. The initial configure failure (local OpenSSL discovery
prefix) and unrepaired-wheel import failure (unresolved OpenBLAS dependency) are
retained, not restamped as successful processes. The corrected build uses the
existing conda OpenSSL/LibArchive locations, verified sccache 0.16.0 and actual
launcher-bearing compiler commands. Observed build statistics report 275
requests, 120 hits and 155 misses; the build completes in 248.69 seconds.

The repaired local wheel SHA is
`b4dc4acadcb1b0046212d68b925cba29e94ea4591dcb71adbb198c4e973fba0c`;
the installed native library SHA is
`4f8ae2066cd4bf59f78a040123dacb811550eb56ac26efe12f7585dec0c6d2fb`,
with native source identity
`144eed9aeef931e89f2f924e21359ac4029a161a0e945553890e83657935ca1f`.
The installed Python modules equal committed source. Installed ABI/admission/
ownership/vibration tests pass (69 tests), as do the installed C SDK harness
(including exactly 100 rejected attempts) and C99 header compilation. These
are structural/algorithm checks, not molecular convergence qualification.

The **fresh installed public negative control** finishes with an actual zero
test/process exit. It runs from outside the checkout, pins the installed native
binary, and records hashes of the actual imported modules, input, driver and
independent oracle helper. The nonstationary pyramidal ammonia geometry returns
`target_not_negative`, zero attempts, index zero and no stationary verification;
it is not certified as an equilibrium minimum. One shared SCF takes nine
iterations, and the raw Hessian takes twelve nuclear response solves; maximum
true residual is `3.9409721688452136e-12`. The native public endpoint takes
505.2569066239521 seconds; the entire test, including independent validation,
takes 740.91 seconds with peak RSS 1,095,644 KiB.

The independent central density error is `1.4609424781042435e-12`, with orbital
gradient norm `3.2378883829013e-12`. All full raw matrices pass the predeclared
gates at fourth-order stencil sizes `[0.003, 0.001, 0.0003]` Bohr; maximum errors
are `[8.350564684178607e-11, 1.0198923650062142e-10,
2.4413220928762546e-9]`. No columns are omitted and no gate is widened.

Complete build recovery is retained locally in
`build/qualification/public-saddle-20261111/build-recovery.tar.gz`, 32 verified
members, SHA
`8fcb1c60ba77855c47b3e9ff0a841f69951a596bf23b57886623cefbbbffa15a`.
The separate completed-negative recovery is
`build/qualification/public-saddle-20261111/negative-recovery.tar.gz`, 139
verified members, SHA
`8ef39f8ef372037f3d17efd192b90a17fcec317345bc834e8801a127915e311c`.
It includes raw arrays, compiler artifacts, final exact-source precommit,
installed SDK evidence and the whole-process receipt. No external asset or
release is created.

At this evidence boundary, the installed positive public molecular saddle is
still running and is not accepted. Its first true occupied-subspace overlap
across a trial is `0.9989849540869761`; partial Hessians and this continuity
check are not final convergence. The original Ni full matrix also remains
unaccepted. Neither installed negative-control success nor passing ABI tests
closes the native TS CLI or the requested Ni/I/Cs production-domain work.

## Installed positive acceptance

The subsequent positive receipt supersedes only the pending-positive status
above. It uses the same installed `cf3f871af` artifact, frozen inputs, production
callables and independent helper, not a restamped callback pilot. The public
endpoint and the entire reference test both finish with actual zero exits:
`1 passed in 2110.89s`. Complete process wall time is 2112.13 seconds and peak
RSS 1,084,092 KiB, including independent validation. The public molecular
endpoint itself takes 1919.9627259047702 seconds.

Three native attempts are accepted. Five energy/gradient evaluations and five
fresh raw Hessians consume 42 SCF iterations, 54 Fock builds and 60 nuclear
response solves. Maximum true response residual is `2.9015392434227474e-12`;
minimum observed occupied-subspace overlap is `0.9989849540869761`. All five
reference controls pass: energy change `9.351452945338679e-11` Hartree, gradient
RMS `1.1840979854568566e-10` and maximum `2.3649827929350664e-10` Hartree/Bohr,
displacement RMS `6.5512233203232065e-6` and maximum `1.964847252659271e-5`
Angstrom. Fresh final native verification establishes index one, no ambiguous
active curvature, six geometric rigid modes and six retained vibrational modes.

Independent central density error is `5.471179065352771e-13` and orbital
gradient norm `9.371274234739576e-13`. Three independent full raw-matrix errors
are `[4.929251451457617e-10, 2.574847217384768e-9,
7.035446725200956e-9]`. Corresponding maximum frequency errors are
`[1.522399770692573e-6, 3.3213932852049766e-5,
5.155171254500601e-5]` cm⁻¹. All original gates pass without omitted columns,
widened thresholds or displaced production SCFs.

The final raw Hessian is retained. A separately labelled post-process harmonic
replay, not another optimization, reproduces all six unscaled frequencies:
`[-902.2167603047881, 1719.6442376211628, 1719.726965163753,
3725.992746207092, 4046.7137055084554, 4046.7345924537135]` cm⁻¹.
There is one negative mode and frequency scale factor one; no soft/negative
mode is discarded to obtain that classification.

Complete positive recovery is
`build/qualification/public-saddle-20261111/positive-recovery.tar.gz`, 132
member/hash-verified entries, 17,027,330 bytes, SHA
`58542ffd9b6b6d3832a0b7295666d47bd7ce30df4e05119e09c72c599e3a16af`.
The scope is the installed public all-electron B3LYP/VWN-RPA spherical STO-3G,
native-default-grid, no-D3 ammonia control. This accepts that fixed-model
public molecular workflow, not a native TS CLI, a certified equilibrium-minimum
negative control, or the requested Ni/I/Cs/ECP/D3/def2-SVP endpoint. The original
serial Ni matrix experiment remains separate and has not yet returned a result.

## Stationary-minimum control input

A separate minimum case now freezes independently prepared Cartesian coordinates
and a full-gradient gate of `1e-7` Hartree/Bohr. Its validation-only preparation
uses PySCF gradients and SciPy minimization, never a production TS optimizer.
The constrained solve reports success, but the full Cartesian BFGS solve reports
precision loss rather than success; the final independent maximum gradient is
`1.1319699733114419e-8`. Preserve both solver statuses: satisfying the physical
stationary-input gate is not a claim that the minimizer reports convergence or
that curvature has already been certified.

The public reference driver must separately require fresh native stationary
verification, index zero, zero optimizer attempts, all retained frequencies
positive and molecular `converged == False`, followed by three independent full
raw matrices and unscaled frequency agreement. These gates are frozen before
running the control. Adding the input alone does not qualify that endpoint; the
earlier nonstationary negative receipt remains a different test.

## Installed stationary-minimum acceptance

The separate public minimum control at
`5ecd28301d5048492985f0cdd044e4774be44147` completes with actual zero test and
process exits: `1 passed in 1102.58s`. It runs outside the checkout against a
fresh normally repaired wheel, including the subsequent input/harmonic guards,
with installed saddle modules verified against committed source. The wheel SHA
is `b0861e79ff0bb3b4a89b85cdeedd93aeaf5449c52d8278dcbdde2c716ec4f19a`;
installed native library SHA is
`3a942689b291d382fa30e7345e026f5ae5ce77db2a33dac2b379d75dc0868af6`.
The source identity is
`1e942c693a4be572d1777498eb4e73ff8de389916a18ab81734732e76712738f`.

Native fresh final verification returns `other_index`, index zero, zero
attempts, six rigid and six active modes, no ambiguous active curvature and
**molecular `converged == False`**. This is now a stationary minimum rejection
control, not a relabelled nonstationary input or a reported TS success. Two
energy/gradient evaluations and two fresh raw Hessians consume 18 SCF iterations,
22 Fock builds and 24 nuclear response solves. The native endpoint takes
871.7023754213005 seconds; complete process wall time is 1103.97 seconds with
peak RSS 1,082,360 KiB. Maximum true response residual is
`4.286388255605711e-12`; final maximum gradient is
`1.1319646775476144e-8` Hartree/Bohr, below the predeclared stationary-input gate.

Independent central density error is `2.544853217045784e-12` and orbital gradient
norm `5.335988115494629e-12`. All three full raw matrices pass the original
fourth-order `[0.003, 0.001, 0.0003]` Bohr gates: maximum errors are
`[1.007668393171457e-10, 2.79616794995885e-10, 2.051555036963748e-10]`.
Maximum unscaled frequency error is `3.766160716622835e-7` cm⁻¹. All six retained
frequencies are positive:
`[1355.0826394488868, 1937.65814014424, 1937.7612662190868,
3404.2146811967036, 3669.533056745566, 3669.6608098249953]` cm⁻¹.
No column, negative/soft mode or failed comparison is removed to pass a gate.

The same installed artifact passes 170 ABI/admission/dispatch tests, 81 public
contract/D3/vibration tests plus 17 subtests, nine low-level light-element UKS
and UKS/D3 tests, the native SDK saddle harness and C99 header compilation.
The updated strict documentation build and pre-commit checks pass. These UKS
tests retain their small custom-grid domain, not public/default-grid Ni or ECP
admission. Initial test-only import and scratch C-probe defects are retained
separately from the corrected successful runs.

Verified local recovery under
`build/qualification/stationary-minimum-20261111` consists of the 47-member
build/input archive (SHA
`b020b6a6fc042cf94e17dd1dbf110f0b0d63fc42e9124f43b089b60451087a35`) and
137-member completed-minimum archive (SHA
`41157ef851b8d17ded61a1172d13b2d4b649f8ac4545fd05fc4f373f9f73177b`).
The latter includes final raw arrays, independent full matrices, source hashes,
compiler artifacts, process receipts and low-level UKS regression evidence.
No release or external asset is published.

This does not restamp the original `cf3f871af` positive/negative receipts.
A fresh positive run against the updated `5ecd28301` wheel is still pending
at this evidence boundary. The original Ni full matrix is also still running;
its eleven frozen identities remain unchanged. Native TS CLI, public Ni/UKS/ECP,
mixed Ni/I/Cs and the requested D3/def2-SVP production TS endpoint remain open.

## Updated installed positive acceptance

The subsequently completed fresh positive against the updated `5ecd28301`
artifact also finishes with actual zero test and process exits:
`1 passed in 2111.80s`. This supersedes the pending-positive statement at the
previous evidence boundary, without restamping the original `cf3f871af` run.
The normally repaired wheel, native library and source identities are the same
as the separately accepted updated stationary-minimum control above. Every
input, driver, independent helper and imported production-source hash is checked
again against this run's own receipt; no production callable is replaced.

Native final status is `first_order_saddle`, with fresh stationary verification,
one negative active curvature and all five original convergence controls
satisfied. Three attempts are accepted. Five energy/gradient evaluations and
five fresh raw Hessians consume 42 SCF iterations, 54 Fock builds and 60 nuclear
response solves. Native endpoint time is 1920.5464496659115 seconds; complete
process wall time is 2113.05 seconds with peak RSS 1,091,612 KiB. These are
complete small-molecule endpoint measurements, not a Ni or scalability claim.

Independent final density error is `5.471179065352771e-13`; orbital gradient
norm is `9.371274234739576e-13`. All three independent complete raw-matrix
comparisons pass the unchanged fourth-order steps and gates. Their maximum
errors are `[4.929251451457617e-10, 2.574847217384768e-9,
7.035446725200956e-9]`; corresponding maximum frequency errors are
`[1.522399770692573e-6, 3.3213932852049766e-5,
5.155171254500601e-5]` cm⁻¹. The six unscaled final frequencies, with no
deleted soft or imaginary modes, are
`[-902.2167603047881, 1719.6442376211628, 1719.726965163753,
3725.992746207092, 4046.7137055084554, 4046.7345924537135]` cm⁻¹.

The separate ignored local recovery is
`build/qualification/installed-saddle-positive-20261111/positive-recovery.tar.gz`,
132 member/hash-verified entries, 17,025,859 bytes, SHA
`493c2eeaa6607d484ef0eb83f4e7fd9c6c06c04f83283c2d09fc0143a159cf7a`.
It preserves raw matrices, final state, progress events, frozen source hashes,
logs, process/JUnit receipts and the linked updated build/minimum identities.
Historical archives remain unchanged. Public UKS/ECP/Ni admission and installed
native TS CLI qualification remain false in this successful positive receipt;
the overall requested production-domain gates are not closed by this result.
