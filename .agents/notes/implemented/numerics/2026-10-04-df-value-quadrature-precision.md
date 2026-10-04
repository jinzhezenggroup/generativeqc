# Decision: use strict common Rys coefficients for DF values

Status: implemented
Date: 2026-10-04

## Problem

The large supplied-orbital DF source probes in #1781 fail their original
`atol=rtol=3e-10` factor/block gates. Their retained lower-angular raw discrepancies
reach 2.76e-12 and are amplified by the orbital and auxiliary metric transforms.
The auxiliary-g polynomial path is not the sole source of these discrepancies.

A bounded independent host diagnosis identifies an inherited precision limit in
the default degree-13 Rys2–5 coefficients used by generic DF values. On normalized
s/s/d, p/p/d, d/d/d and f/f/f Cartesian z components at T≈2.053027104306768, emitted
production arithmetic differs from a 90-digit analytic center-derivative oracle
by +2.11825e-12, +2.45682e-12, +2.34777e-12 and -1.50142e-12. Replaying the returned
nodes/weights at 90 digits assigns essentially all these errors to quadrature;
argument rounding is below 6.75e-16 and remaining FP64 evaluation error below
1.85e-15. These are primitive accuracy reproductions, not the actual largest
Bvv elements of the molecular runs.

The independent reference also has finite error. The documented normalized
s/s/s example in #1782 has analytic value 5.55819383594290434659468908658;
emitted host arithmetic gives 5.558193835942903 and PySCF 2.14.0 gives
5.558193835942818. Therefore matching that reference more closely is not a
universal accuracy criterion in an ill-conditioned transformed frame.

## Decision

Extend the existing common Rys high-accuracy option from three/four roots to
two through five roots. Regenerate degree-17 coefficients at 90 decimal digits
with the existing offline Stieltjes/Jacobi generator. Select that option for
generic DF value emission. Preserve the evaluator, root counts, interval width,
small/large branches, FP64 arithmetic and the existing Rys3/4 strict coefficients.
No independent scientific recurrence or runtime high-precision dependency is
introduced. Default Direct emitters and the complete derivative header remain
byte-identical to their previous versions.

This replaces the demonstrated primitive approximation floor. It does not
relax the source factor/block, energy, residual or symmetry gates. It does not
constitute a passing large-factor qualification or a performance promotion.

## Cost and scope

Each interpolated node/weight series executes eight paired Clenshaw steps
instead of six. The generated strict tables cover only intervals up to their
existing asymptotic switch, so the four DF tables hold 11,448 doubles rather
than 15,680: 91,584 versus 125,440 coefficient bytes. This is static table
accounting, not a measured endpoint timing or total device-memory claim.
No owner allocation, lease, screening, source traversal or tensor size changes.

The shared generator is deterministic; existing strict Rys3/4 coefficient tuples
are unchanged. Its full four-table regeneration took approximately 7.3 seconds
on one validation CPU core. That is generation cost, not a runtime speedup.

## Validation

`tests/python/test_df_value_precision.py` compiles the actual generated value
consumer and checks every defining moment against independent incomplete-gamma
integrals at 90 digits, including interpolation boundaries, small/large branches,
random arguments, extreme arguments and strided output sentinels. The unchanged
5e-14 moment criterion applies to all four root counts. Independent analytic
center derivatives check the four normalized primitive failures at 3e-14
absolute tolerance. All eight numerical tests fail with the prior emitted
header and pass with the strict coefficients. Compatibility assertions preserve
all default Direct emitter hashes and the derivative header hash
`71430c3d8929868c10156d36578f719bbd56d595b35c4ae56131930395f4ce04`.

The focused host/compiler/auxiliary-g/value-candidate selection passes 162 tests;
57 explicit CUDA cases skip locally. Host compilation used ccache. A host check
does not establish device execution or exact new-head CUDA compilation.

## Remaining molecular qualification

The actual large Bvv error needs the already retained frozen ethane230 diagnostic:
the exact orbital matrix C, native/reference raw tensors and metrics, shared
inverse-root matrix and retained rank/cutoff, basis/geometry/ordering metadata,
and original elementwise masks with worst indices and values. Apply the same
transform order and pair projection on both sides. No fresh RHF/CC solve is
needed for this attribution step. If only selected weighted contributions are
exported, omitted contributions need a rigorous tail bound below the gate.

Replacing the metric eigensolver alone, pair-symmetrizing factors, or observing
agreement in a scalar total energy cannot close the original failed factor
gates. Retain those failures and original source/binary identities until the
fixed-frame comparison is requalified. The new coefficients address the
confirmed primitive error component without claiming that all reference and
transform-conditioning errors disappear.
