# Decision: opt-in common-Gram normalization for strict independent DIIS

Status: implemented for explicitly selected independent references only
Date: 2026-11-14

## Problem

The independent cold neutral-singlet linear H–Ni–I discovery at Ni–H 2.7 and
Ni–I 4.5 Bohr exhausted 240 PySCF SCF iterations with physical orbital-gradient
norm `3.331517953631639e-10`. A separately cold no-extra-diagonalization attempt
with a stricter `1e-11` threshold reached the identical energy and residual and
also failed. Extra final diagonalization is therefore not the demonstrated
cause of this particular stagnation. Both failures remain failures.

The installed upstream `pyscf.lib.diis.DIIS.extrapolate` applies an absolute
`1e-14` eigenvalue cutoff to the augmented affine DIIS matrix. Its constraint
entries are order one while the error Gram entries decrease quadratically with
SCF errors. Tiny physical errors can consequently disappear from that solve.
This identifies a numerical risk; it does not prove the sole cause of every
heavy-element SCF failure or establish a physical root match.

## Decision

`tests/python/independent_diis.py` supplies an explicit reference-only factory.
It divides the **entire** real positive-semidefinite error Gram matrix by one
common positive maximum before the small affine solve. Exact DIIS coefficients
are invariant under this common scaling. It does not normalize each error
individually, change physical Focks or occupations, replace integrals or XC,
seed another geometry, or weaken electronic-convergence certification.

The relative eigensystem pseudoinverse handles redundant history. History,
coefficient, affine residual, symmetry, finiteness and PSD bounds fail closed.
Exactly zero Gram entries permit a bounded affine average, not an SCF success
declaration. Each solve retains its original Gram scale and coefficient bound.

`independent_solver` and `_independent_gradient` accept the factory only when
explicitly supplied. Existing references keep their original default PySCF
DIIS policy. Production code does not import this helper. Actual independent
SCF convergence flags and recomputed physical orbital-gradient gates still
decide whether a state can be used.

## Evidence and limits

- 44 focused algebra/SCF/resource-policy tests pass. They cover common scales
  `1e-240` through `1e240`, redundant/zero/invalid Gram matrices, actual history
  extrapolation, cold RKS and UKS, physical-state agreement with default light
  references, and preservation of a genuinely unconverged state.
- The original 38-test run had one test-only UKS electron-count einsum error;
  both physical SCF assertions had already passed. That failed log is preserved;
  the corrected 38-test and expanded 44-test receipts are distinct.
- A separately predeclared cold normalized-DIIS linear H–Ni–I discovery converges
  its initial state in 60 cycles to actual physical gradient
  `6.192962666371274e-13`, without reusing either failed density. The same energy
  threshold, 240-cycle allowance and strict `1e-11` physical gate remain.
- This discovery evaluates 22 cold frames in 466.540 seconds, but exhausts its
  six optimizer evaluations with full nuclear gradient about `3.9574e-4`, above
  the `2e-7` stationary gate. Several neighboring cold states change occupied
  roots and acquire transverse gradients. Electronic convergence is therefore
  **not** stationary-fixture, stable-root, Hessian, vibration or TS acceptance.

Evidence: `build/qualification/host-nihi-stationary-discovery-20261114/`, with
separate original, `v2`, `v3`, and `diis-controls` receipts. The launch-time v3
control-test source is byte-reconstructed and verified against its frozen hash;
only tests were extended during discovery, not the executed DIIS algorithm.

## Rejected alternatives and invariants

- Do not raise physical residual gates or overwrite upstream convergence flags.
- Do not infer successful curvature from energy convergence, a common root
  label, a radial residual, or a nominal occupied/virtual gap alone.
- Do not silently make this strategy the default independent reference.
- Do not relabel earlier failed source or qualification receipts.
- Do not seed independent displaced states with native or neighboring states.
- Do not unblock either public ECP saddle guard on these controls.

## Revisit when

Upstream provides a qualified homogeneous DIIS solve, the reference needs
complex orbitals or more than 64 history vectors, or evidence supports a
different explicitly frozen convergence policy. Any heavy stationary fixture
still needs cold displaced-root matching, complete multistep independent raw
matrices, native state/H/HVP parity and physical-mass mode checks.

## Subsequent bounded bent hypothesis

A distinct cold bent neutral-singlet H–Ni–I discovery includes two bond lengths,
the bend, and an in-plane global orientation. The last parameter is deliberate:
fixed atom-centered discretized angular grids need not have exact continuous
rotational invariance. Restricting a bent search to internal coordinates alone
must not replace the full Cartesian-gradient gate.

The validation-only L-BFGS-B request returns `success=false`, `status=2`,
`ABNORMAL`, after 18 iterations and 47 objective requests (38 distinct cold
geometries, 533.975 seconds). Its actual candidate nevertheless passes the
separately declared **discovery** nuclear-gradient gate: maximum Cartesian
component `4.8763819794953554e-8` versus `2e-7`, with actual physical electronic
gradient `1.0740957125177575e-12`. This does not relabel optimizer success, a
minimum, a matched native state or a verified saddle. Full independent matrices
and new installed native endpoints have separate frozen acceptance and receipts.

The current expanded focused suite passes 62 tests with five opt-in heavy tests
skipped. Those skips are explicit: these controls do not substitute for the
actual new heavy calculations. Annotation/format-only changes to the helper
retain byte-verified original executed source and a separate one-off provenance
comparison; they do not restart the scientific requests or change the algorithm.

## Complete installed mixed-fixture acceptance

The separate installed producer actually completes in 2572.647 seconds, under
its frozen 3600-second allowance, with peak RSS 385900 KiB. Its supplemental
cold state, full raw Hessian and two independently executed HVP requests take
523.308, 552.467, 748.261 and 748.036 seconds respectively. The producer remains
unaccepted by design; a separate read-only `validation.json` accepts its outputs.

The independent request completes all 108 cold displaced states and all three
fourth-order raw matrices in 1518.288 seconds. Its minimum occupied-root overlap
is `0.9954149875343793` and maximum actual electronic gradient is
`7.556088644797674e-12`. The three installed raw-matrix errors are
`4.0549e-10`, `2.0219e-9`, `3.7041e-9`; both HVPs pass every independent matrix
action, with worst error `4.5295e-10`. Raw asymmetry/translation are
`5.5067e-12`/`5.9135e-12`. The installed occupied-root overlap is
`0.9999999999999967`, and density/energy errors are `4.5829e-12`/`7.0486e-12`.
All response residuals pass the original `2e-10` heavy gate with relative
`1e-11`, absolute `1e-12` solver settings. The full-H residual is `1.1820e-11`:
this is **not** absolute-only `1e-12` response qualification.

With explicit Ni-58/H-1/I-127 masses and the supplied-H geometric rigid-motion
convention, the three unscaled modes are 267.205358, 476.074725 and
2069.201413 cm^-1, with zero negative internal modes. Independently implemented
single-SVD mass-weighted null-space references match all mode characters, with
minimum mass-weighted overlap `0.9999999999999987`.

Do not hide the raw rotational residual: it is `0.0020083064790817544`, and the
three independent raw references reproduce it. No raw matrix is repaired.
Geometric projection is an explicit mode convention, not exact continuum
rotational-invariance certification or a proof of its particular cause.

The permanent opt-in `ni-i-stationary-rks` registry case reproduces the frozen
geometry, isotope masses, cold 240-cycle normalized-reference policy, fourth-
order stencils and mode gates. Its native allowance is 600 cycles; the actual
installed cold state needed 52 iterations/56 Fock builds. The expanded local
suite passes 63 tests with six explicitly skipped heavy cases; the accepted
installed calculation is not rerun merely to exercise the newly registered
pytest spelling. A separate read-only regression binding compares its current
mode helper and registry against the accepted actual outputs. Original executed
qualification sources/inputs are retained before adding that registry case.

Evidence: `build/qualification/host-nihi-stationary-20261114/`. The genuine
current-wheel core remains SHA-256
`64baa62a72d0e2cf8be8ba0342200136d141e8e42fe0e0b5717bd34ecae57225`, scientific
source `4879bbaf9f567b416a7d6ed70772ac8fddb7b059e4f324cba21aaa86fb197198`.
An attempted observational CPU perf sample was denied by host permissions; no
settings were changed and no scientific request was restarted for profiling.

This closes the small mixed Ni/iodine stationary raw-H/HVP/mode evidence gap
under the stated discretized-grid convention. It is not a representative Ni
reaction TS, a 25--40-atom complete electronic endpoint, or all-consumer release
qualification. Both public ECP saddle guards remain closed.
