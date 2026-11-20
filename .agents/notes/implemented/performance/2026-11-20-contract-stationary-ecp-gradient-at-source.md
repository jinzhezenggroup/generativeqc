# Decision: Contract stationary Gaussian ECP forces at source

Status: implemented
Date: 2026-11-20

## Problem

The frozen 25-atom Ni/I28 two-spin-block workflow (226 spherical AOs,
multiplicity 3) passed native UKS SCF in 134 iterations and 136 physical Fock
builds but failed before its hcore force stage. The 64 MiB publication/source
budget was charged for a raw checked-grid ECP derivative inventory even when
the operator selected analytic Gaussian jets. Simply using the analytic raw
matrix bound or raising the limit would retain coordinate-by-AO derivative
storage solely to contract it immediately.

## Decision

Contract ordered AO weights in the existing analytic order-one pair traversal,
using both weights for an off-diagonal pair and the physical AO/ECP-center
scatter shared with the raw derivative. Retain only compensated coordinate
accumulators and the result. The planner charges that schedule and its existing
kernel-block/metadata workspace. Keep the checked two-grid matrix path and its
budget for other already-supported operators; raw exports retain their ABI and
existing resource contract. The ionic tail is still a separate source.

## Evidence

The compiled test checks nonsymmetric ordered weights, multiple ECP and AO
centers, and both Cartesian and spherical representations against the raw
analytic derivative and independent quadrature. Exact-capacity refusal/recovery
and the checked-grid fallback remain tested. Admitting hcore does not establish
complete 25-atom force timing or independent endpoint agreement.

## Independent-reference caution

The archived independent PySCF Newton calculations used `canonicalization=False`.
Their converged orbitals were retained unrotated, but the returned orbital
energies belonged to the rotated orbitals. That is not an energy-weighted density
for the nuclear Pulay term. For verification only, rebuild the physical Fock and
canonicalize within each occupied/virtual subspace without changing the density,
occupations, method or grid. Check against `D F D` in each spin block. The sealed
25-atom all-electron two-step actions initially disagreed by 11.08 Eh/Bohr²;
replaying both gradients with the corrected Pulay term reduced the discrepancy
to 4.21e-10 Eh/Bohr². The original mismatch is not native curvature evidence.
