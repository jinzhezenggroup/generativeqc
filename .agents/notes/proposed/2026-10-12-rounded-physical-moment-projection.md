# Decision: retain work branches for latent physical-moment projections

Status: proposed correction to PR #2312
Date: 2026-10-12

## Problem

The [exact physical kinetic predicate](2026-10-12-exact-physical-kinetic-point-predicate.md)
mistook unchanged rounded work coordinates for unprojected physical moments.
Independently rounded AO contractions can be slightly outside the exact per-spin
kinetic cone, even though the rounded FHC cap equals the rounded spin sigma.
The exact total comparison then discards a legal inward derivative. At retained
RKS point 1234, the exact total sigma/cap gap is 5.773972489594029e-18 and the
half-vtau changes from -0.16758689734565194 to zero. A 1728-point replay changes
228 derivatives and shifts the XC potential by 0.0010995401772939048.

## Decision

Require both unchanged work coordinates and exact per-spin feasibility before
replacing the work-domain comparison. Reuse the bounded binary64 comparator
with the other spin zeroed, without a tolerance or a separate numerical algebra.
If either raw spin is infeasible, retain the original work branch: a projection
can be latent in rounded coordinates. If both spins are feasible, Cauchy-Schwarz
implies total feasibility, including equality, so use the inclusive inward side.
The comparator primitive continues to distinguish exact adjacent exterior inputs;
that distinction does not assign a new derivative contract to projected work.

Keep floors, FHC/cross-spin clamps, the normal-product guard, branch payloads and
already-differentiated work coefficients unchanged. Scalar-feature bulk programs
retain their incumbent contract. This does not solve empty-beta continuation.

## Rejected Alternatives

- A numerical epsilon conflates exact feasibility with projection semantics.
- Classifying exact raw total moments alone loses valid rounded-AO derivatives.
- Matching raw Libxc vxc fails existing decimal ties and molecular inward paths.
- Replacing the whole ratio perturbs minority/empty-spin coefficients.

## Evidence

The two retained AO/feature points in `rks-moment-rounding.json` are inputs,
not reference derivatives. They check virtual occupation and fixed-electron
occupied-to-virtual transfer against independently extrapolated pinned Libxc
energies. Both controls fail against the e41a762 qualified CPU binary; the worst
local response error is about 0.14177328. The correction passes these controls,
the decimal ties, existing work-domain coefficients and independent Fraction
primitive controls: 28 focused host cases. Broader host validation passes 88
cases with 34 explicit skips (33 opt-in CUDA cases and one historical migration
audit). Compiler structure checks 525 modules with zero dependency errors;
changed-file hooks pass.

Fresh Slurm job 7661 on node1/main, one RTX 5090, finite ten-minute allocation
and scheduler visibility 1 passes 14 CPU/CUDA point controls. A repeat under
memcheck passes with zero errors. Both backends also replay all 1728 retained
RKS points, assemble their XC potential and independently check virtual and
fixed-electron inward directions with PySCF AO/Libxc energies. CUDA replay is
repeated under memcheck. These repeats are not additional independent gates.
The qualified source manifest is
`99f986daa1556f05bf3d576a7c2d854ba675c07ade81e5bc1ce64d11adfeb381`;
local receipts are under `.cache/1118-runtime/physical-kinetic-repair/`.
The final note refresh is requalified separately; this source identity is not
relabeled. The complete sm_120 point probe uses 255 registers, 800 stack bytes
and 192/200 spill store/load bytes, not isolated-comparator resources.

## Invariants and Revisit Conditions

Do not label captured-input point replay as new SCF or analytic-force qualification.
Revisit this policy only with independent full-Fock inward evidence for projected
moments and empty/floored-spin domains. A faster comparator must preserve exact
binary64 extremes, adjacent sides, decimal ties and rounded-AO inward controls.
