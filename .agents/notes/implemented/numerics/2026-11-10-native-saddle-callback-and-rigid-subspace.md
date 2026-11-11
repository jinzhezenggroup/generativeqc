# Decision: expose the bounded native saddle owner without molecular admission

Status: implemented
Date: 2026-11-10

## Problem and decision

The internal corrector already owns rational eigenvector following, exact
accepted-step curvature refresh, five convergence tests and the 100-attempt
cap. Reimplementing those equations in a Python or CLI optimizer would create
another scientific owner without solving the missing molecular derivative
qualification. Export that owner through a separate versioned C header,
`generativeqc/saddle.h`, with synchronous borrowed callbacks and caller-owned
outputs. Keep electronic/method admission outside this mathematical API.

This supersedes the **no installed callback API / no rigid projection** portions
of the [earlier corrector note](2026-11-05-native-local-saddle-corrector.md).
Its distinction between algorithm qualification and a molecular TS task remains.

## Invariants

- Remove translations and rotations from geometry, not by deleting small
  Hessian eigenvalues. Two-pass orthogonalization builds an unweighted Cartesian
  vibrational subspace. Linear and nonlinear geometries have five and six
  rigid motions respectively. A rank change fails rather than silently changing
  the tracked target space. No mass-weighted frequency is inferred.
- Feed the reduced raw matrix to the existing shared native symmetric
  eigensolver and lift its eigenvectors back to Cartesian space. Preserve the
  unprojected raw Hessian in final and last-accepted outputs.
- Optional progress observations contain coherent accepted-state data, including
  after rejected attempts. Cancellation or observer failure clears stationary
  verification, including during the final observation. Do not consume another
  electronic callback after cancellation.
- Explicit ABI, coordinate, attempt and numerical-capacity admission precedes
  derivative work. The capacity bound excludes caller/provider memory and is
  not a process RSS receipt. Distinguish C execution success from optimizer
  convergence. Retain work counters and the last accepted state on failure.
- The callbacks must bind one fixed qualified physical model and state policy.
  A generic callback interface cannot qualify B3LYP/Ni/ECP chemistry, an
  installed molecular TS CLI task, or exactly-once total-method assembly.

## Evidence

`tests/native/test_saddle_optimizer.cpp` retains the original algorithm gates
and adds an independently constructed linear-triatomic spectrum
`[-6, 2, 6, 6]`, finite rotation/translation covariance, a minimum negative
control, raw-Hessian retention, progress cancellation and observer failure.
`tests/native/test_saddle_api.cpp` executes the C ABI's convergence, fresh-final
work counters, minimum classification, derivative failures, asymmetric raw
matrix rejection, last-valid-state outputs, cancellation, capacity/ABI/control
rejection and exactly 100 rejected attempts.

Both host executables pass. The C ABI harness also compiles against the header
and links/runs against the installed CPU wheel rather than the checkout
library. A separate C99 header compilation passes. Local builds use verified
sccache 0.16.0; cache hits are not claimed. The SDK harness initially needed an
explicit OpenBLAS dependency link-search path, which is an SDK test setup
correction, not a scientific fallback.

## Remaining work

Independent molecular saddle/minimum controls and a fixed-request installed
molecular TS API/CLI still require complete qualified derivative endpoints.
Neither the callback receipt nor these mathematical tests closes issue #2848.

## Bounded molecular callback pilot (2026-11-11)

A separate scratch pilot binds the actual native C corrector to one prepared
RKS B3LYP state per geometry. Energy/forces and raw Hessian callbacks borrow
the same prepared SCF owner, with no reference engine, finite differences or
external optimizer in the callback. It uses real-spherical STO-3G and the
native default grid on near-planar ammonia, not the requested def2-SVP/D3/Ni
domain. The five requested convergence controls and 100-attempt cap are fixed
before the run; rigid projection removes six geometric rigid modes.

The initial native SCF takes eight iterations. Its complete raw Hessian takes
the pipeline to 832.53 seconds, has maximum asymmetry `8.33501242e-13`, and
has one negative projected curvature, `-0.1285902654 Eh/Bohr^2`. The gradient
maximum is still `0.0176231067 Eh/Bohr`, so this is not a converged saddle.
The corrector requests a trial geometry and a fresh SCF/Hessian. The entire
process reaches its external 1,200-second time limit during the second Hessian,
after two complete energy/gradient calls, one complete Hessian, two attempted
Hessians and sixteen completed SCF iterations; peak RSS is 972,324 KiB.

The last coherent native progress observation retains the initial accepted
geometry with zero accepted steps and `stationary_verified=0`. The process
timeout is not a returned optimizer failure receipt and does not prove the
public task's failure-publication contract. No final Hessian, frequency result
or positive molecular-TS acceptance is inferred. Evidence is retained in
`/tmp/gqc-ammonia-native-saddle-pilot/`, its adjacent log/time files, and the
frozen script manifest. The negative pilot motivates endpoint work reuse;
it does not turn the callback ABI into an installed molecular TS API/CLI.

## Returned molecular controls and independent acceptance (2026-11-11)

The optimized production sources subsequently complete the same near-planar
ammonia callback calculation, but the original scratch harness exits **one**:
its final publisher supplies `seconds` twice after writing the returned native
receipt and arrays. Preserve this reporting failure rather than calling the
entire original process successful. Its result is separately requalified below;
neither that checker nor a corrected retry changes the failed process receipt.

The native C corrector itself returns first-order-saddle status with three
accepted attempts, five energy/gradient calls and five exact raw Hessians.
Those callbacks consume 42 native SCF iterations and 60 coupled response solves.
They retain Gaussian/VWN-RPA B3LYP RKS, spherical STO-3G, neutral singlet and the
native default grid at every geometry, without D3, an external optimizer or
displaced production SCFs. This small all-electron control is not the requested
def2-SVP/D3/Ni reaction domain. At the returned point:

- Energy change: `9.351452945338679e-11` Hartree.
- Gradient RMS / maximum: `1.1840979854568566e-10` /
  `2.3649827929350664e-10` Hartree/Bohr.
- Displacement RMS / maximum: `6.5512233203232065e-6` /
  `1.964847252659271e-5` Angstrom.
- One negative and zero near-zero active curvatures; six rigid and six active
  modes. The target curvature is approximately `-0.1021304176` Hartree/Bohr².

The original callback process takes 1907.61 seconds and peak RSS 985,708 KiB,
then fails only during post-receipt reporting. Its exact driver, manifest,
progress, returned arrays, process timing and explicit `reporting-failure.json`
remain under `/tmp/gqc-ammonia-native-saddle-production` and adjacent files.
Do not restamp it as a clean later-commit run or a zero-exit installed endpoint.

### Independent raw matrix and unscaled frequencies

The separate `/tmp/gqc-ammonia-molecular-acceptance.py` exits zero. Its gates
were frozen before completion of the native callback, in
`/tmp/gqc-ammonia-molecular-acceptance-gates.json`, SHA256
`fd170c4b1601f016dbb74c95e632947e71ba30fb12608d44fa1cdbab78f02bbd`.
It recomputes a fresh public Calculator final Hessian on the same physical graph
and geometry, then independently reconverges PySCF with the same raw basis and
moving quadrature. This is the checkout public Hessian, not an installed TS API.
The fresh endpoint takes 489.664640664123 seconds, including eight native SCF
iterations; its twelve response solves have maximum residual
`4.3982896162560947e-13`. The raw callback/public matrices agree within 2e-8,
with no post-hoc Hessian symmetrization or displaced production SCF.

The central independent density, energy and gradient maximum errors are
`5.471179065352771e-13`, `3.126388037344441e-13` Hartree and
`2.566002965664893e-14` Hartree/Bohr. The independent electronic orbital-gradient
norm is `9.371274234739576e-13`. For every raw column, independently converged
five-point moving-grid gradients at 0.003, 0.001 and 0.0003 Bohr give maximum
matrix errors `4.929251451457617e-10`, `2.574847217384768e-9` and
`7.035446725200956e-9`, below the preselected 1e-5 gate. Corresponding unscaled
frequency maximum errors are `1.522399770692573e-6`, `3.3213932852049766e-5`
and `5.155171254500601e-5` cm-1, below the 1 cm-1 gate.

The six retained signed frequencies are approximately `[-902.2167603,
1719.6442376, 1719.7269652, 3725.9927462, 4046.7137055, 4046.7345925]` cm-1.
No vibrational eigenvalue is deleted and the scale factor is one. The separate
checker process takes 1640.18 seconds and peak RSS 984,840 KiB, including
960.24 seconds waiting for the original callback; that wait is not additional
Hessian computation and these overlapping clocks must not be naively summed.

### Molecular nonnegative-target control and clean retries

A pyramidal ammonia geometry with N height 0.75 Bohr is an intentionally
nonstationary nonnegative-target control, not a certified equilibrium minimum.
The same native corrector returns target-not-negative with zero attempts,
zero negative curvatures and no stationary verification, preserving its raw
last-valid state. Its target curvature is `0.11773800223682922` Hartree/Bohr².
The original control harness also fails in the same post-receipt publisher.
It takes 502.20 seconds and peak RSS 991,152 KiB; retain that exit-one record.

The separate `/tmp/gqc-ammonia-nonnegative-independent-acceptance.py` exits zero
in 230.11 seconds with peak RSS 441,444 KiB. Its independently reconverged
central energy/gradient errors are `2.842170943040401e-14` Hartree and
`2.808864252301646e-13` Hartree/Bohr. At the same three steps, raw matrix errors
are `8.350564684178607e-11`, `1.0198923650062142e-10` and
`2.4413220928762546e-9`; both native and independent vibrational indices are zero.

The corrected scratch producer `/tmp/gqc-ammonia-native-callback-fixed.py`
changes only the reporting collision and retains each Hessian's explicit
response diagnostics. Fresh clean-5281dc3c7 positive and nonnegative retries
use the original geometries, five controls, 100-attempt cap and finite process
limits. The clean nonnegative-target retry returns the same rejection and
exits zero after 499.63 callback seconds. The separate acceptance checker
confirms that complete process and all three independent raw matrices. The
positive retry is still running at this checkpoint, not yet an accepted clean
whole-process receipt. `/tmp/gqc-ammonia-fixed-callback-acceptance.py` waits for
actual zero process exits and checks raw matrices against all three independent
matrices before accepting either complete retry.

The two compiled host saddle suites and 18 vibration regressions pass. A
verified ignored local archive,
`build/qualification/molecular-callback-controls-20261111/local-recovery.tar.gz`,
preserves 45 completed-record/script members in 48,029 bytes, SHA256
`74e220ec1173316ab2548adf50e61b1ebb1a5788127f9e1d2319fa4c99117df0`.
Its input-manifest SHA256 is
`db0a20deada29a77032285b8eecb5db11f9d532ac67984f363099eb127944bd8`.
The archive explicitly records original harness failures and pending clean
retries; it does not include incomplete live outputs or publish an external asset.

### Subsequent zero-exit complete retries

Both fresh clean-5281dc3c7 retry processes subsequently exit zero, and the
separate whole-process checker also exits zero with `accepted=true`. The
positive complete native callback takes 1903.76 seconds, peak RSS 989,676 KiB;
its callback clock is 1903.1845671329647 seconds. It retains three accepted
attempts, five energy/gradient calls, five fresh raw Hessians, 42 native SCF
iterations and 60 nuclear response solves. Every Hessian's response diagnostics
are retained; the maximum residual across all sixty solves is
`2.9015392434227474e-12`, below the frozen 2e-10 gate. All five convergence
controls, final index one and all six unscaled frequencies remain as reported
above. This is a genuinely completed callback process, not a relabeling of
the original reporting failure.

The clean nonnegative control takes 500.20 seconds, peak RSS 989,916 KiB;
its callback clock is 499.6268391869962 seconds. It consumes nine SCF iterations
and twelve nuclear response solves, with maximum response residual
`3.9409721688452136e-12`. The C call succeeds but returns explicit native
target-not-negative status, zero attempted steps and no stationary verification.
An expected negative-control rejection is not saddle convergence.

The independent checker first enforces a preselected 2e-12-Bohr geometry
correspondence before comparing either fresh raw matrix with all three retained
independently converged matrices. Both repeat exactly the independent matrix
errors listed above, and the positive independent frequency errors are
unchanged. Source/driver/library identities, raw symmetry/translation, actual
zero process exits and every per-column response residual pass. The checker's
1560.17 seconds mostly wait for the positive callback; do not add that wait to
the separately measured native endpoint or infer a paired speedup.

The separate completed-retry recovery archive is
`build/qualification/molecular-callback-clean-20261111/local-recovery.tar.gz`,
with 39 verified members, 56,391 bytes and SHA256
`2e553f97249abda1cc7694181b006258a63e45074ea63aaebb0f1f3dba3592e4`.
Its input-manifest SHA256 is
`780c1133a2a59278d4d2d32a60d60c3c48daa826d7c93cebed3842aa093c6f5d`.
It retains exact clean producer/acceptance scripts, logs, process timings,
response receipts, native and independent matrices, conventions and C ABI
sources, and links the distinct original-failure and native-binary archives.
No Release, external binary host or asset is published.

These molecular controls strengthen the native algorithm/reference evidence.
They do not deliver the fixed-request installed molecular TS API/CLI, qualify
Ni/ECP/D3 reaction optimization, or close #2848 or any of #2844–#2847.
