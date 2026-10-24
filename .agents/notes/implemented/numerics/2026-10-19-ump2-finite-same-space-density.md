# Decision: finite UMP2 same-space density from the existing integral cotangent

Status: implemented
Date: 2026-10-19

## Problem

The canonical same-space stationarity equation has the form
`G[p,q]-G[q,p] = -2*(eps[p]-eps[q])*D[p,q]`. Its right-hand side
vanishes at an exact occupied/occupied or virtual/virtual degeneracy even when
the finite offdiagonal density is nonzero. Checking that the numerator is small
and dropping that density therefore cannot certify the nuclear force. A full
antisymmetric Lagrangian residual cannot detect the discarded degenerate term.
A symmetric STO-3G H6 ladder exposed a physical force error and dependence on
rotations of the same occupied reference, despite converged SCF/response and
small final stationarity residuals.

## Decision and proof

Keep `unrestricted_energy_program` as the only energy-equation owner. Write its
scalar channel energy as `k*A*A/D`, with `A=g-x` for same spin and `A=g` for
opposite spin. The inverse-denominator derivative identity
`delta(D^-1) = -D^-1 * delta(D) * D^-1` gives finite occupied-leg contractions
`-k*t[p]*t[q]` and virtual-leg contractions `+k*t[p]*t[q]`, where `t=A/D`.
These remain finite as a same-space energy gap tends to zero; the excitation
denominators retain their existing admission threshold.

The existing physical integral cotangent Gamma includes both ordered exchange
scatters. For the current energy owner it equals t for same spin and 2t for
opposite spin. Consequently each of the four legs contributes a Gram contraction
with coefficient -1/4 (occupied) or +1/4 (virtual). Both same-spin occurrences
must be added; there is no extra restricted-spin occupation factor. Diagonal
entries reproduce the original epsilon cotangents, and nondegenerate symmetric
offdiagonals equal half the old single-triangle gap quotient.

Production does not restate those scientific coefficients. The generator obtains
the physical-bar normalization from the existing VJP at unit g, x=-g and D=-1.
The common TensorIR JVP polarizes each quadratic epsilon cotangent using
`B(u,v) = DQ(u)[v]/2`. The resulting bounded native helper consumes two retained
physical integral bars and returns the four leg contributions. This derivation
continues to refer to the existing energy/VJP hashes; no amplitude or density
energy equation is introduced.

The native runtime performs only structural leg indexing and accumulation. It
fills symmetric offdiagonals in the existing one-electron/Fock matrices and
preserves the original generated epsilon-diagonal summation. It does not add
rank-two or rank-four state. The original same-space compatibility gate remains:
a gap at most 1e-9 and antisymmetric gradient above 1e-9 rejects after finite
density assembly, without dividing or discarding density. The separate final
1e-7 global residual, occupied/virtual response, reference identities and
force/Pulay consumers retain their existing thresholds and semantics.

## Resource contract

Reserve the sum of all emitted bilinear helper capacities plus four caller-owned
doubles, using checked arithmetic. The initial emitted capacities are 608, 608
and 520 bytes; derive them from generated plans rather than pinning a constant.
This conservative sum is valid even though only one helper runs at a time.

Independently, the one-electron derivative consumer retains C center Jets with C
components for C=3*atom_count. The force plan now reserves C*C+16*C doubles with
checked arithmetic and preserves all previous allowances. The validated native
force source admits s/p/d/f shells; a direct request for a higher-angular-
momentum recurrence fallback fails before the derivative consumer. Neither
change is a process-RSS or speed claim.

## Rejected alternatives

- Dividing an antisymmetric orbital gradient by an arbitrarily clipped gap changes
  the mathematics and does not recover the finite limit.
- A blanket rejection of same-space degeneracy would exclude already successful
  Li and Be virtual-p cases. It also avoids rather than computes the finite limit.
- An additional handwritten amplitude/energy implementation would create a second
  normalization owner and increase opportunities for same/opposite-spin drift.
- Retaining another rank-four tensor is unnecessary: the existing physical ijab
  cotangents already contain the required amplitudes.

## Evidence and invariants

Permanent controls cover compiled helper polarization/signs, repeated same-spin
legs and empty channels against independent four-spin enumeration, native H6
alpha-only/beta-only/both-spin occupied-gauge invariance, independent PySCF
analytic gradients, reconverged multistep differences, degenerate-virtual Li/Be,
existing H2/H3/OH/NH2 cases, source failure/recovery and inclusive numeric budgets.
The sparse-basis regression keeps 600 nuclei but one AO, and requires a 20 MiB
force request to fail before publication while energy-only replay succeeds.

One-off design controls compared 1,080 nondegenerate offdiagonals to the original
quotient and 6,000 compiled bilinear evaluations to polarization of the exact
existing native VJP. Four H6 gauges agreed within 9e-16; all 18 gradient components
agreed with independent analytic gradients within 1.8e-10 using partial-integral
central differences. These controls motivate the implementation; the final
candidate native library must separately pass the executable regression suite.

## Revisit when

A compiler-owned tensor/GEMM lowering can reduce contraction work while retaining
the same energy owner, numerical gates, complete-endpoint resource plan and
source-matched endpoint evidence. No performance qualification is claimed here.

## References

- PR #2619 and issue #1823
- `docs/developer/ump2_adjoints.md`
- `tests/native/test_ump2_contract.cpp`
- `tests/python/test_ump2_adjoint.py`
- `tests/python/test_ump2_forces.py`
