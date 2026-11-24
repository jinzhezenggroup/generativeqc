# Decision: Stream explicit CPU LDA/PBE RKS Hessian solves natively

Status: implemented
Date: 2026-11-24

## Decision

Expose a dense-only sequential operation beside the automatic and recycled
operations. It consumes the same accepted native token and numerical resources,
but skips optional concurrent response admission. The caller's block size is
only a logical partition: no basis or initial guess is shared between columns,
and no live block of response vectors is required. Geometry-only source reuse
is independent of the solver choice and remains available under its existing
limits. The C ABI uses a separate symbol, so its availability can be checked
without changing the layout or semantics of existing options or receipts.

## Boundaries

The existing LDA/PBE RKS method and physical-state limits still apply. Automatic
independent batching and true block GMRES are not sequential solves. Recycling
has a different initial guess, so it is not an acceptable substitute either.
Keep the Cartesian compatibility owner for old libraries, block GMRES and
other unlinked schedules; do not reuse it after a linked sequential execution
fails native admission or convergence. The C++ wrapper and native CLI expose
the same streamed operation without adding another numerical owner. The CLI
flag is incompatible with recycling, HVP and CUDA.

## Acceptance

The native bridge test covers unchanged output and receipt on short capacity,
wrong geometry and unconverged CPKS. The Python moving-gradient controls compare
raw Cartesian and real-spherical LDA/PBE and linked PBE-D3(BJ) matrices against
independently reconverged forces; a complete raw result must preserve the full
response count, independent partition semantics and physical residual gate.

References: #2227, #932; `docs/developer/dft_hessian.md`.
