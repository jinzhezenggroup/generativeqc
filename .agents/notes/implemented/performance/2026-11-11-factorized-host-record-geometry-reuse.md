# Decision: bind factorized host geometry once per primitive component page

Status: implemented
Date: 2026-11-11

## Problem

The frozen nuclear RHS of the asymmetric NiH3 UKS HVP reaches the raw-first
factorized ERI consumer before the observed response solve. The previous host
schedule recomputed Gaussian-pair geometry and Boys moments for every selected
Cartesian component, even though these inputs depend only on the primitive
record and quartet angular bound. A 1,800-second complete diagnostic did not
return an HVP; this is not evidence of a response-solver failure.

The investigation and frozen molecular gates are preserved in
`../../proposed/2026-11-11-ni-uks-hvp-first-integral-work.md`. This decision
implements only its record-local geometry reuse candidate.

## Decision

The compiler-generated host `Program` binds a full-range four-center geometry
once per primitive record in a component page. The native streaming template
passes that immutable binding to each component evaluation. No geometry is
retained after the record, shared across calls, or keyed by object identity.
The existing 128-record and 64-component bounds and publication buffer remain.

The canonical factorized primitive accepts a borrowed geometry only on CPU,
for the full-range four-center family. Without it, the original binder remains
the owner for every supported family and radial operator. CUDA emission and
its callable contract are unchanged. The compiler continues to own geometry
construction and recurrence mathematics; the native template owns iteration,
validation and atomic publication, not an independent scientific recurrence.

`geometry_evaluations` reports actual primitive-record bindings for successful
host invocations. Record and component work counts remain separately visible.
The numeric lease continues to exclude generated scalar/jet stack temporaries,
compiler processes and library mappings; this change does not claim a new
whole-process memory bound.

## Invariants and alternatives

- Preserve independent mathematical slots at repeated molecular centers,
  Cartesian/spherical weights, derivative signs and accumulation order.
- Preserve late-error atomic publication and finite compiler/cache ownership.
- Do not screen primitive records, change precision or loosen scientific gates.
- Do not introduce a persistent geometry cache with unchecked input lifetime.
- First-order-only jet lowering and sharing raw sources between spin RHS
  consumers are distinct, unimplemented opportunities, not part of this change.

## Evidence

The final changed-domain run passes 28 tests in 558.99 seconds:
`test_factorized_host.py`, `test_factorized_second.py` and
`test_global_hybrid_uks_hessian.py`. They include independent libcint primitive
first/HVP checks, up to 64 selected components including ffff, and complete raw
light-element coupled UKS Hessians against independently converged multistep
moving-grid gradients. Existing input, capacity and failure-publication checks
remain. Compiler structure passes with 603 modules and zero dependency errors.
The packaging/source-identity domain adds 61 passing tests and one explicitly
skipped test. CI-pinned ty 0.0.82 and Ruff 0.17.4 checks pass, as does the
changed native header's clang-format check. These checks do not restamp the
earlier installed-wheel numerical receipts.

A scratch before/after schedule probe uses 16 primitive records and 64 selected
components of `(f,f,p,s)`. Raw first and weighted HVP arrays are bitwise equal.
The new invocation reports 16 geometry bindings, 1,024 component evaluations,
one chunk and a 45,208-byte explicit numeric lease. Median raw-first time is
0.0063128797 to 0.0047946554 seconds; weighted-HVP time is 0.0063104248 to
0.0048581958 seconds. These are local schedule measurements, **not molecular
endpoint speedup or Ni HVP qualification**.

An opt-in scratch comparison confirms unchanged emitted CUDA bytes. No source
fingerprint test is added. Local evidence is retained in
`/tmp/gqc-geometry-final-tests.{log,xml}`, its adjacent `*-time.txt`, and
`/tmp/gqc-factorized-geometry-benchmark/{before,after}.{json,npz}`.

The complete Ni HVP diagnostic reaches its finite 3,600-second limit without
returning an HVP, with peak process RSS 2,034,812 KiB. Its frozen dirty-source
provenance and negative timeout receipt are retained under
`/tmp/gqc-continue-ni-hvp-hoisted-geometry/`; it must not be attributed to the
clean parent commit. No response iteration count, residual or unique failure
cause can be inferred from its last `native-hvp-start` observation. Geometry
reuse alone is not a qualified Ni endpoint. Public Ni/ECP curvature admission
is unchanged. The separate scalar-direction/shared-spin candidate remains
proposed in `../../proposed/2026-11-11-factorized-directional-first-and-shared-spin-rhs.md`.

## Revisit when

Complete endpoint measurements show the next dominant repeated source work.
Any first-only or cross-spin reuse change needs independent numerical tests,
explicit semantic work counts and its own complete endpoint receipt. A faster
component probe alone cannot justify opening a public scientific domain.
