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
