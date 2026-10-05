# Decision: borrow the symmetric density views in shared checked lowering

Status: implemented
Date: 2026-10-05

## Problem

The molecular COSX derivative still owned the dense symmetric-density projection
loop after ordinary projection moved to shared lowering. Its existing scalar
program computes `(AO * 0.5) * (D_rc + D_cr)` inside each increasing reduction
step. Materializing the symmetric density, scaling the final contraction, or
auditing only the final matrix could change rounding and intermediate failures.

## Decision

Describe the dense equation with existing TensorIR add, transpose and einsum
nodes. A bounded common recognizer verifies the original half-scaled graph,
square readonly source, exclusive virtual intermediates, strict FP64 precision
and the matching original scalar program. Both graph identities, scalar input
roles and the source/sum/transpose hashes enter the canonical request.

The existing prepared executor borrows both read directions from one density
allocation. Its compiler-bound scalar helper owns the arithmetic, half-factor
placement and finite/error checks. A native descriptor flag describes this
proved input-view recipe; it does not select a provider. Invalid coefficient,
shape, publication and helper/view combinations reject before enqueue. Optional
provider recipes remain explicitly unsupported for the complete checked region.

Molecular preparation now has four immutable ordinary/symmetric full/tail sites,
with a 128 KiB host reservation. Device storage and its bounded peak are
unchanged: no density copies, symmetric matrix, packing buffer or numeric cache
is added. Existing enclosing Fock admission reads the updated reservation.

## Rejected alternatives

- Symmetrize a new numeric density matrix before an ordinary GEMM.
- Move the half factor to final publication or reassociate the scalar program.
- Leave the method-owned dense loop behind a shared API wrapper.
- Permit a library that cannot implement the per-step checked recipe.
- Infer density symmetry and silently omit one read direction.

## Invariants and evidence

Asymmetric density is essential to the independent long-double oracle. Native
full/tail tests additionally compare every output bit to the retained incumbent
CUDA scalar-helper traversal. Zero AO with overflowing density-pair sums must
still publish zero and a sticky error; helper/view mismatch and ordinary
execution cannot bypass the checked binding. Real molecular response diagnostics
cover all four sites and complete analytic CPU derivative, changed-geometry,
spherical basis, translation and lifetime gates remain required.

Source/binary manifests, compiler commands, ccache statistics, finite Slurm
allocations and qualification logs live in ignored `.artifacts/1884-cosx-symmetric/`.
This is a consumer/ownership migration without a performance promotion.

## Consequences and revisit

Composed ESP response, bidirectional potential, molecular pullbacks and point
reductions still require subsequent shared-region migrations. New provider
recipes must preserve the complete declared checked contract; mathematical
equivalence alone does not prove the original scalar execution semantics.

## References

#1884, #1886, #1970, #1971 and
[checked derivative sites](2026-10-05-cosx-checked-derivative-sites.md).
